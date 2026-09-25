"""Render the simulated workspace, detect the parts, and plan a pick-and-place cycle.

Reports how far each detection is from the part's true position in the
simulated scene, and how closely the IK solutions reach their targets.
Writes samples/detection_annotated.jpg, samples/demo.gif and results/summary.json.
"""
import json
import time
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2

from core.camera import CameraModel, SyntheticWorkspace, SyntheticObject
from core.perception import VisionPerception
from core.kinematics import ManipulatorKinematics
from core.trajectory import PickAndPlaceStateMachine
from core.visualizer import SimulationVisualizer


def main():
    print("visiongrasp: running the pick-and-place pipeline...")

    # 1. Initialize System Components
    print("\n[1/5] Initializing Camera Model & Kinematics Engine...")
    camera = CameraModel()
    kinematics = ManipulatorKinematics()
    perception = VisionPerception(camera)
    state_machine = PickAndPlaceStateMachine(kinematics)
    visualizer = SimulationVisualizer(kinematics)

    # 2. Setup Synthetic Industrial Table Workspace
    print("[2/5] Synthesizing Industrial Scene with Workpieces & Bins...")
    workspace = SyntheticWorkspace(camera, table_z=0.0)

    # Ground truth objects placed on table
    objects = [
        SyntheticObject(
            name="red_cylinder",
            color_name="red_block",
            bgr_color=(40, 40, 220),
            hsv_lower=np.array([0, 100, 100]),
            hsv_upper=np.array([10, 255, 255]),
            position_base=np.array([0.42, -0.15, 0.02]),  # 2cm block height
            dimensions=(0.04, 0.04, 0.04),
            yaw_deg=25.0,
        ),
        SyntheticObject(
            name="blue_cube",
            color_name="blue_block",
            bgr_color=(220, 60, 40),
            hsv_lower=np.array([100, 100, 100]),
            hsv_upper=np.array([130, 255, 255]),
            position_base=np.array([0.48, 0.12, 0.02]),
            dimensions=(0.04, 0.04, 0.04),
            yaw_deg=-15.0,
        ),
    ]

    # Target sorting bins
    bins_pos = [
        (np.array([0.25, 0.32, 0.0]), "#ffaaaa"),   # Red bin
        (np.array([0.25, -0.32, 0.0]), "#aaccff"),  # Blue bin
    ]

    t0_synth = time.perf_counter()
    rgb, depth = workspace.generate_scene(objects)
    synth_time = (time.perf_counter() - t0_synth) * 1000.0
    print(f"      Rendered 640x480 RGB-D frames in {synth_time:.2f} ms")

    # 3. Execute Perception Pipeline
    print("[3/5] Running Real-Time Vision Perception Pipeline...")
    t0_perc = time.perf_counter()
    detections = perception.detect(rgb, depth)
    perc_time = (time.perf_counter() - t0_perc) * 1000.0
    print(f"      Detected {len(detections)} target workpieces in {perc_time:.2f} ms:")

    # The camera sees each part's top face. The grasp point is half a part
    # height below it; the height comes from the part catalogue, not from
    # the simulator's ground truth.
    part_half_height = 0.02
    truth = {o.color_name: o for o in objects}
    detection_errors = []
    for i, det in enumerate(detections):
        pos = det.position_3d_base
        gt = truth[det.class_name]
        xy_err_mm = float(np.linalg.norm(pos[:2] - gt.position_base[:2]) * 1000.0)
        z_err_mm = float((pos[2] - part_half_height - gt.position_base[2]) * 1000.0)
        yaw_err = float((det.yaw_deg - gt.yaw_deg + 45.0) % 90.0 - 45.0)    # square parts repeat every 90 deg
        detection_errors.append({"part": det.class_name, "xy_mm": round(xy_err_mm, 2),
                                 "z_mm": round(z_err_mm, 2), "yaw_deg": round(yaw_err, 2)})
        print(f"      [{i+1}] {det.class_name:<12} | 3D Base Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m | Yaw: {det.yaw_deg:.1f} deg"
              f" | error vs scene: xy {xy_err_mm:.2f} mm, z {z_err_mm:+.2f} mm, yaw {yaw_err:+.2f} deg")

    # 4. Plan the cycle for the first detection, into the bin for its colour
    bin_for = {"red_block": bins_pos[0][0], "blue_block": bins_pos[1][0]}
    target = detections[0]
    target_pos = target.position_3d_base.copy()
    target_pos[2] -= part_half_height
    destination_bin = bin_for[target.class_name].copy()
    destination_bin[2] = 0.02

    print(f"\n[4/5] Planning Smooth Quintic Trajectory...")
    print(f"      Pick Target : [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m")
    print(f"      Place Target: [{destination_bin[0]:.3f}, {destination_bin[1]:.3f}, {destination_bin[2]:.3f}] m")

    t0_plan = time.perf_counter()
    trajectory = state_machine.plan_cycle(target_pos, destination_bin, pick_yaw_deg=target.yaw_deg)
    plan_time = (time.perf_counter() - t0_plan) * 1000.0
    total_duration = trajectory[-1].time_s

    print(f"      Generated {len(trajectory)} trajectory waypoints ({total_duration:.2f} s execution) in {plan_time:.2f} ms")

    # IK accuracy at the four key poses: position error, how far the gripper
    # tilts from pointing straight down, and at the pick poses how far its
    # yaw is from the detected part yaw.
    hover = state_machine.hover_height
    ik_rows = []
    for name, goal, yaw in [("pre-pick", target_pos + [0, 0, hover], target.yaw_deg), ("pick", target_pos, target.yaw_deg),
                            ("pre-place", destination_bin + [0, 0, hover], 0.0), ("place", destination_bin, 0.0)]:
        c, s_ = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        goal_rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])
        q, pos_err = kinematics.solve_ik(goal, target_orientation=goal_rot)
        _, rot = kinematics.forward_kinematics(q)
        tilt = float(np.degrees(np.arccos(np.clip(-rot[2, 2], -1.0, 1.0))))
        yaw_err = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])) - yaw)
        ik_rows.append({"pose": name, "position_mm": round(pos_err * 1000.0, 3),
                        "tilt_deg": round(tilt, 3), "yaw_error_deg": round((yaw_err + 180.0) % 360.0 - 180.0, 3)})
    worst_pos = max(r["position_mm"] for r in ik_rows)
    worst_tilt = max(r["tilt_deg"] for r in ik_rows)
    worst_yaw = max(abs(r["yaw_error_deg"]) for r in ik_rows)
    print(f"      IK at key poses: position error up to {worst_pos:.3f} mm, tilt up to {worst_tilt:.3f} deg, "
          f"yaw error up to {worst_yaw:.3f} deg")

    # 5. Export Diagnostic & Showcase Artifacts
    print("\n[5/5] Exporting Showcase Artifacts...")
    samples_dir = Path(__file__).resolve().parent.parent / "samples"
    samples_dir.mkdir(exist_ok=True)

    # Save perception visual frame
    vis_annotated = perception.draw_annotations(rgb, detections)
    annotated_path = samples_dir / "detection_annotated.jpg"
    cv2.imwrite(str(annotated_path), vis_annotated)
    print(f"      Saved annotated perception frame -> {annotated_path}")

    # Save animated GIF demonstration
    gif_path = samples_dir / "demo.gif"
    print(f"      Rendering 3D kinematic trajectory to {gif_path} (step stride = 3)...")
    t0_render = time.perf_counter()
    visualizer.render_gif(
        trajectory=trajectory,
        objects_pos=[(target_pos, target.class_name, "#d62728")],
        bins_pos=bins_pos,
        camera_pos=camera.t_base_cam[:3, 3],
        output_path=str(gif_path),
        fps=15,
        step_stride=3,
    )
    render_time = time.perf_counter() - t0_render
    print(f"      Animation rendered in {render_time:.2f} s")

    summary = {
        "scene": "simulated: rendered RGB-D frame, 640x480, two 4 cm blocks",
        "detections": detection_errors,
        "ik_residual_at_key_poses": ik_rows,
        "trajectory": {"points": len(trajectory), "duration_s": round(total_duration, 2)},
        "timing_ms_this_machine": {"perception": round(perc_time, 1), "planning": round(plan_time, 1)},
    }
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Done. Artifacts saved to: {samples_dir}")


if __name__ == "__main__":
    main()
