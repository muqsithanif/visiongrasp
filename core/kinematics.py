"""Manipulator kinematics solver: 6-DoF Forward & Inverse Kinematics with joint limit enforcement."""
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import ikpy.chain
from scipy.spatial.transform import Rotation


class ManipulatorKinematics:
    """UR5 6-DoF robotic arm kinematics engine."""

    # Default home joint angles (radians) in ready pose
    HOME_JOINTS = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

    def __init__(self, urdf_path: Optional[str] = None):
        if urdf_path is None:
            # Default to bundled URDF
            root_dir = Path(__file__).resolve().parent.parent
            urdf_path = str(root_dir / "data" / "urdf" / "ur5.urdf")

        # Active mask: ignore base fixed link and ee fixed link
        # Chain has 8 links: [base_fixed, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, ee_fixed]
        active_mask = [False, True, True, True, True, True, True, False]
        self.chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=active_mask,
        )
        self.num_active_joints = 6

        # Standard UR5 workspace limits (in meters)
        self.reach_min = 0.15
        self.reach_max = 0.85
        self.z_min = -0.05
        self.z_max = 0.80

    def is_reachable(self, target_pos: np.ndarray) -> bool:
        """Check if 3D position is within geometric spherical workspace."""
        pos = np.asarray(target_pos)[:3]
        radius = float(np.linalg.norm(pos))
        if radius < self.reach_min or radius > self.reach_max:
            return False
        if pos[2] < self.z_min or pos[2] > self.z_max:
            return False
        return True

    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector 3D position and 3x3 rotation matrix from 6 active joint angles."""
        full_angles = np.zeros(len(self.chain.links))
        full_angles[1:7] = joint_angles[:6]

        transform_mat = self.chain.forward_kinematics(full_angles)
        position = transform_mat[:3, 3]
        rotation = transform_mat[:3, :3]
        return position, rotation

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
        max_iterations: int = 40,
    ) -> Tuple[np.ndarray, float]:
        """Solve inverse kinematics for target position [x, y, z] and optional orientation.

        Returns:
            (joint_angles, position_error_meters)
        """
        target_pos = np.asarray(target_pos)[:3]
        if not self.is_reachable(target_pos):
            raise ValueError(f"Target position {target_pos} is outside reachable workspace.")

        # Default orientation: gripper pointing straight down towards table
        # Tool Z-axis points towards -Z of base frame
        if target_orientation is None:
            # Gripper pointing down: rotation matrix
            target_rot = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ], dtype=np.float64)
        else:
            target_rot = target_orientation

        # Construct 4x4 target transformation matrix
        target_frame = np.eye(4, dtype=np.float64)
        target_frame[:3, :3] = target_rot
        target_frame[:3, 3] = target_pos

        initial_full = np.zeros(len(self.chain.links))
        if initial_guess is not None:
            initial_full[1:7] = initial_guess[:6]
        else:
            initial_full[1:7] = self.HOME_JOINTS

        # Solve via ikpy numerical solver
        solution_full = self.chain.inverse_kinematics_frame(
            target_frame,
            initial_position=initial_full,
        )

        active_solution = solution_full[1:7]
        # Verify forward kinematics error
        achieved_pos, _ = self.forward_kinematics(active_solution)
        err = float(np.linalg.norm(achieved_pos - target_pos))

        # If 6D solver error is higher than 1.5cm, fallback to position-only IK
        if err > 0.015:
            solution_pos = self.chain.inverse_kinematics(
                target_position=target_pos,
                initial_position=initial_full,
            )
            active_solution = solution_pos[1:7]
            achieved_pos, _ = self.forward_kinematics(active_solution)
            err = float(np.linalg.norm(achieved_pos - target_pos))

        return active_solution, err

    def get_joint_trajectory_fk(self, joint_trajectory: np.ndarray) -> np.ndarray:
        """Compute end-effector 3D trajectory positions [N, 3] for a sequence of joint angles [N, 6]."""
        ee_positions = []
        for joints in joint_trajectory:
            pos, _ = self.forward_kinematics(joints)
            ee_positions.append(pos)
        return np.array(ee_positions)
