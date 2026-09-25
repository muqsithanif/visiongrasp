# visiongrasp

Vision-guided pick-and-place for a UR5 arm, run in a simulated workspace. An overhead RGB-D camera finds coloured parts on the table, their pixel positions are turned into 3D coordinates in the robot's base frame, inverse kinematics finds joint angles for a gripper pointing straight down and turned to the part's yaw, and the arm moves on smooth quintic trajectories.

![Pick and place cycle](samples/demo.gif)

*The red block is found by the camera, picked with the gripper turned to its yaw, and placed in the red bin.*

---

## From pixel to robot coordinates

```mermaid
flowchart LR
    RGBD["RGB-D frame"] --> Seg["HSV segmentation"]
    Seg --> Part["Centroid, median depth, min-area rectangle"]
    Part --> Base["Deproject to the base frame: position and yaw"]
    Base --> IK["IK: gripper down, turned to the part's yaw"]
    IK --> Traj["Quintic joint trajectories"]
```

The camera is fixed above the table, looking straight down. This is an eye-to-hand setup with a known camera pose, not a camera on the gripper. A detection at pixel (u, v) with depth Z becomes a point in the camera frame through the intrinsic matrix, and then a point in the robot's base frame through the camera's pose:

```
P_cam  = Z · K⁻¹ · [u, v, 1]ᵀ
P_base = T_base_cam · [P_cam, 1]ᵀ
```

The camera's image axes are swapped relative to the base axes: image x runs along base +Y, and image y along base +X. That matters for orientation. An earlier version reported the angle of the part in the image as its yaw, which was 40–60° off. Yaw is now measured by deprojecting a second point along the part's edge and taking the direction in base X–Y.

## Perception

`core/perception.py` segments each colour in HSV and cleans the mask with morphological opening and closing. For each part it then:

- takes the centroid,
- takes the median depth over the whole part mask, so one bad pixel at an edge does not move it, and
- fits a minimum-area rectangle for orientation.

The camera sees the top face. The grasp point is half a part height below it, and that height comes from the part catalogue, not from the simulator.

## Motion

`core/kinematics.py` solves IK with ikpy on the UR5 URDF, constraining both position and orientation. An earlier version called ikpy in a way that silently ignored the orientation, which left the gripper tilted by 98° at the pick pose and up to 167° at the others. `core/trajectory.py` interpolates each segment in joint space with the quintic `s(τ) = 10τ³ − 15τ⁴ + 6τ⁵`. This starts and ends every segment at zero velocity and zero acceleration. The cycle runs home → above the part → grasp → lift → above the bin → release → home.

---

## Results

These come from the simulated scene, a rendered 640×480 RGB-D frame with two 4 cm blocks, and are saved in `results/summary.json`.

| | Red block | Blue block |
|---|---:|---:|
| Position error, horizontal | 0.73 mm | 0.74 mm |
| Position error, height | 0.00 mm | 0.00 mm |
| Yaw error | −1.8° | +1.0° |

At the four key poses of the cycle, IK reaches the target with position error below 0.001 mm and gripper tilt below 0.001°.

These errors are against a clean rendered frame with no lens distortion, depth noise or lighting changes, so they measure the geometry, not how the pipeline would do on a real camera.

## Limits

- **Everything is simulated**: the scene, the camera and the arm. There is no physics, so grasp success is not modelled.
- **Parts are found by colour threshold.** That works for the distinct colours used here, not for mixed or textured parts.
- **`ros2/` has thin rclpy wrappers** for the perception and planning steps. They have not been run against a live ROS 2 graph.

## Run it

```bash
pip install -r requirements.txt
python scripts/run_demo.py
pytest -q
```

## Tests

There are seventeen tests. These are the ones worth naming:

- **Yaw is reported in the robot base frame.** Five part angles, each within 3° of the truth, taken modulo 90° because the parts are square.
- **IK keeps the gripper pointing down.**
- **Deprojection round-trips**, and points behind the camera are rejected.
- **Joint motion stays continuous** through the cycle, and the gripper stays closed while carrying the part.
