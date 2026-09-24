"""Full end-to-end execution demo: Scene generation -> Perception -> Kinematics -> Trajectory -> GIF Export."""
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
    print("==================================================================")
    print("    Vision-Guided Robotic Pick-and-Place Demonstration")
    print("==================================================================")

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

    for i, det in enumerate(detections):
        pos = det.position_3d_base
        print(f"      [{i+1}] {det.class_name:<12} | 3D Base Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m | Yaw: {det.yaw_deg:.1f} deg")

    # 4. Plan Pick-and-Place Trajectory for First Detected Object
    target = detections[0]
    target_pos = target.position_3d_base.copy()
    target_pos[2] = 0.02  # Object surface grasp height
    destination_bin = bins_pos[0][0].copy()
    destination_bin[2] = 0.02

    print(f"\n[4/5] Planning Smooth Quintic Trajectory...")
    print(f"      Pick Target : [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m")
    print(f"      Place Target: [{destination_bin[0]:.3f}, {destination_bin[1]:.3f}, {destination_bin[2]:.3f}] m")

    t0_plan = time.perf_counter()
    trajectory = state_machine.plan_cycle(target_pos, destination_bin)
    plan_time = (time.perf_counter() - t0_plan) * 1000.0
    total_duration = trajectory[-1].time_s

    print(f"      Generated {len(trajectory)} trajectory waypoints ({total_duration:.2f} s execution) in {plan_time:.2f} ms")

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

    print("\n==================================================================")
    print("Demo Execution Completed Successfully!")
    print(f"Artifacts ready in: {samples_dir}")
    print("==================================================================")


if __name__ == "__main__":
    main()
