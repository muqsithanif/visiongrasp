"""3D Visualization & animation renderer for robot kinematics, camera perception, and trajectory."""
from typing import List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for CLI and headless execution
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

from core.kinematics import ManipulatorKinematics
from core.trajectory import TrajectoryPoint, ExecutionPhase


class SimulationVisualizer:
    """Renders 3D perspective scenes of manipulator arm, camera ray, workspace objects, and trajectory."""

    def __init__(self, kinematics: ManipulatorKinematics):
        self.kinematics = kinematics

    def get_link_positions(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute 3D coordinates of all arm joints from base to end-effector."""
        full_angles = np.zeros(len(self.kinematics.chain.links))
        full_angles[1:7] = joint_angles[:6]

        pts = []
        current_trans = np.eye(4)
        for i, link in enumerate(self.kinematics.chain.links):
            link_trans = link.get_link_frame_matrix(full_angles[i])
            current_trans = current_trans @ link_trans
            pts.append(current_trans[:3, 3])

        return np.array(pts)

    def plot_frame(
        self,
        ax: Axes3D,
        trajectory_pt: TrajectoryPoint,
        objects_pos: List[Tuple[np.ndarray, str, str]],  # [(pos, label, color_code)]
        bins_pos: List[Tuple[np.ndarray, str]],          # [(pos, color_code)]
        camera_pos: np.ndarray,
        trail_pts: Optional[np.ndarray] = None,
    ) -> None:
        """Draw a single 3D scene frame onto a matplotlib 3D axis."""
        ax.cla()

        # Set coordinate bounds
        ax.set_xlim([-0.3, 0.7])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.05, 0.85])
        ax.set_xlabel("X (m)", fontsize=8, labelpad=2)
        ax.set_ylabel("Y (m)", fontsize=8, labelpad=2)
        ax.set_zlabel("Z (m)", fontsize=8, labelpad=2)
        ax.tick_params(labelsize=7)

        # Draw industrial table surface
        tx = np.array([0.1, 0.65, 0.65, 0.1])
        ty = np.array([-0.45, -0.45, 0.45, 0.45])
        tz = np.array([0.0, 0.0, 0.0, 0.0])
        ax.plot_trisurf(tx, ty, tz, color="#c8ced4", alpha=0.45, shade=True)

        # Draw sorting bins
        for b_pos, b_col in bins_pos:
            bx, by, bz = b_pos
            ax.bar3d(bx - 0.04, by - 0.04, bz, 0.08, 0.08, 0.04, color=b_col, alpha=0.3, edgecolor="black", linewidth=0.5)
            ax.text(bx, by, bz + 0.06, "Bin", color="black", fontsize=7, ha="center")

        # Draw workpieces
        for o_pos, o_lbl, o_col in objects_pos:
            ox, oy, oz = o_pos
            ax.bar3d(ox - 0.02, oy - 0.02, oz - 0.02, 0.04, 0.04, 0.04, color=o_col, alpha=0.85, edgecolor="black")

        # Draw Overhead Camera
        cx, cy, cz = camera_pos
        ax.scatter([cx], [cy], [cz], color="#333333", s=80, marker="s", label="RGB-D Camera")
        ax.text(cx, cy, cz + 0.04, "RGB-D Cam", fontsize=7, ha="center", weight="bold")

        # Draw optical ray from camera to targeted pick point
        if len(objects_pos) > 0:
            target_obj_pos = objects_pos[0][0]
            ax.plot(
                [cx, target_obj_pos[0]],
                [cy, target_obj_pos[1]],
                [cz, target_obj_pos[2]],
                color="#00aa00", linestyle=":", linewidth=1.2, alpha=0.7, label="Vision Ray"
            )

        # Draw trajectory history trail
        if trail_pts is not None and len(trail_pts) > 1:
            ax.plot(trail_pts[:, 0], trail_pts[:, 1], trail_pts[:, 2], color="#ff6600", linestyle="--", linewidth=1.5, alpha=0.65)

        # Draw Robot Arm Skeleton
        link_pts = self.get_link_positions(trajectory_pt.joint_angles)
        ax.plot(link_pts[:, 0], link_pts[:, 1], link_pts[:, 2], "-o", color="#1f77b4", linewidth=4.5, markersize=6, label="UR5 Arm")

        # Draw End-Effector Gripper
        ee_pos = trajectory_pt.ee_position
        gripper_col = "#d62728" if trajectory_pt.gripper_closed else "#2ca02c"
        ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]], color=gripper_col, s=90, marker="D",
                   label=f"Gripper ({'CLOSED' if trajectory_pt.gripper_closed else 'OPEN'})")

        # Title & Phase annotation
        phase_name = trajectory_pt.phase.name.replace("_", " ")
        ax.set_title(
            f"Autonomous Vision-Guided Pick-and-Place\nPhase: {phase_name} | t = {trajectory_pt.time_s:.2f}s",
            fontsize=9, pad=10
        )
        ax.view_init(elev=28, azim=-55)

    def render_gif(
        self,
        trajectory: List[TrajectoryPoint],
        objects_pos: List[Tuple[np.ndarray, str, str]],
        bins_pos: List[Tuple[np.ndarray, str]],
        camera_pos: np.ndarray,
        output_path: str,
        fps: int = 15,
        step_stride: int = 2,
    ) -> None:
        """Render and export animated GIF of the full pick-and-place sequence."""
        fig = plt.figure(figsize=(7, 6), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        sub_traj = trajectory[::step_stride]
        trail_pts: List[np.ndarray] = []

        # Track workpiece position dynamically: when gripper is closed, object travels with gripper
        initial_workpiece_pos = objects_pos[0][0].copy()
        current_objects = [(o_pos.copy(), lbl, col) for o_pos, lbl, col in objects_pos]

        def update(frame_idx: int):
            pt = sub_traj[frame_idx]
            trail_pts.append(pt.ee_position)

            if pt.gripper_closed and pt.phase in [ExecutionPhase.CLOSE_GRIPPER, ExecutionPhase.LIFT_POST_GRASP, ExecutionPhase.TRANSIT_BIN, ExecutionPhase.DESCEND_PLACE]:
                # Workpiece moves with gripper tip
                current_objects[0] = (pt.ee_position.copy(), current_objects[0][1], current_objects[0][2])
            elif pt.phase in [ExecutionPhase.OPEN_GRIPPER, ExecutionPhase.ASCEND_RETURN, ExecutionPhase.HOME] and len(trail_pts) > 10:
                # Placed in bin
                bin_pos = bins_pos[0][0].copy()
                bin_pos[2] += 0.02
                current_objects[0] = (bin_pos, current_objects[0][1], current_objects[0][2])

            self.plot_frame(
                ax=ax,
                trajectory_pt=pt,
                objects_pos=current_objects,
                bins_pos=bins_pos,
                camera_pos=camera_pos,
                trail_pts=np.array(trail_pts),
            )

        anim = FuncAnimation(fig, update, frames=len(sub_traj), interval=1000.0 / fps)
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close(fig)
