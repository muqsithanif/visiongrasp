"""Overhead RGB-D Camera model, calibration transforms, and synthetic workspace generator."""
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import cv2


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


class CameraModel:
    """Pinhole camera model with rigid 3D spatial transformation (base <-> camera)."""

    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        camera_pose_in_base: Optional[np.ndarray] = None,
    ):
        # Default: 640x480 overhead camera with 60 deg horizontal FoV
        if intrinsics is None:
            width, height = 640, 480
            fov_x = np.deg2rad(60.0)
            fx = (width / 2.0) / np.tan(fov_x / 2.0)
            fy = fx
            cx = width / 2.0
            cy = height / 2.0
            intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
        self.intrinsics = intrinsics

        # Default camera position: overhead at [0.4, 0.0, 0.95], looking straight down
        # Robot base frame: X forward, Y left, Z up
        # Standard camera optical frame: X right, Y down, Z forward (pointing down)
        if camera_pose_in_base is None:
            # R_base_cam: optical X points along +Y, optical Y points along +X, optical Z points along -Z
            r_mat = np.array([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0]
            ], dtype=np.float64)
            t_vec = np.array([0.45, 0.0, 0.90], dtype=np.float64)
            camera_pose_in_base = np.eye(4, dtype=np.float64)
            camera_pose_in_base[:3, :3] = r_mat
            camera_pose_in_base[:3, 3] = t_vec

        self.t_base_cam = camera_pose_in_base.copy()
        # Inverse transform: camera coordinates to base coordinates
        self.t_cam_base = np.linalg.inv(self.t_base_cam)

    def project_base_to_pixel(self, p_base: np.ndarray) -> Tuple[int, int, float]:
        """Project a 3D point in robot base coordinates into (u, v) pixel and camera Z-depth."""
        p_base_homo = np.append(p_base[:3], 1.0)
        p_cam = self.t_cam_base @ p_base_homo
        z_cam = p_cam[2]
        if z_cam <= 1e-4:
            raise ValueError(f"Point {p_base} is behind or at camera plane (z={z_cam:.4f})")

        p_norm = p_cam[:3] / z_cam
        p_pix = self.intrinsics.matrix @ p_norm
        u = int(np.round(p_pix[0]))
        v = int(np.round(p_pix[1]))
        return u, v, float(z_cam)

    def deproject_pixel_to_base(self, u: float, v: float, depth: float) -> np.ndarray:
        """De-project 2D pixel (u, v) and camera depth into 3D robot base frame coordinates."""
        k_inv = np.linalg.inv(self.intrinsics.matrix)
        ray_cam = k_inv @ np.array([u, v, 1.0], dtype=np.float64)
        p_cam = ray_cam * depth
        p_cam_homo = np.append(p_cam, 1.0)
        p_base = self.t_base_cam @ p_cam_homo
        return p_base[:3]


@dataclass
class SyntheticObject:
    name: str
    color_name: str
    bgr_color: Tuple[int, int, int]
    hsv_lower: np.ndarray
    hsv_upper: np.ndarray
    position_base: np.ndarray  # [x, y, z] in base frame
    dimensions: Tuple[float, float, float]  # [width, length, height] in meters
    yaw_deg: float  # rotation around Z in base frame


class SyntheticWorkspace:
    """Generates synthetic RGB-D frames of industrial table and workpieces for simulation and testing."""

    def __init__(self, camera: CameraModel, table_z: float = 0.0):
        self.camera = camera
        self.table_z = table_z

    def generate_scene(self, objects: List[SyntheticObject]) -> Tuple[np.ndarray, np.ndarray]:
        """Render RGB image and float32 Depth map (in meters)."""
        w, h = self.camera.intrinsics.width, self.camera.intrinsics.height
        rgb = np.full((h, w, 3), (210, 210, 215), dtype=np.uint8)  # Industrial steel table background

        # Compute depth of the flat table plane
        # Camera is looking down from Z=0.90 to table Z=0.0 -> depth is approximately 0.90m
        _, _, table_depth = self.camera.project_base_to_pixel(np.array([0.45, 0.0, self.table_z]))
        depth = np.full((h, w), table_depth, dtype=np.float32)

        # Draw grid lines on table surface for visual texture
        for gx in np.linspace(0.2, 0.7, 6):
            u1, v1, _ = self.camera.project_base_to_pixel(np.array([gx, -0.35, self.table_z]))
            u2, v2, _ = self.camera.project_base_to_pixel(np.array([gx, 0.35, self.table_z]))
            cv2.line(rgb, (u1, v1), (u2, v2), (185, 185, 190), 1)

        for gy in np.linspace(-0.35, 0.35, 8):
            u1, v1, _ = self.camera.project_base_to_pixel(np.array([0.2, gy, self.table_z]))
            u2, v2, _ = self.camera.project_base_to_pixel(np.array([0.7, gy, self.table_z]))
            cv2.line(rgb, (u1, v1), (u2, v2), (185, 185, 190), 1)

        # Render each object
        for obj in objects:
            pos = obj.position_base
            dx, dy, dz = obj.dimensions
            top_z = pos[2] + dz / 2.0
            rad = np.deg2rad(obj.yaw_deg)

            # Local corners on top surface
            hx, hy = dx / 2.0, dy / 2.0
            corners_local = np.array([
                [-hx, -hy],
                [hx, -hy],
                [hx, hy],
                [-hx, hy]
            ])
            rot_z = np.array([
                [np.cos(rad), -np.sin(rad)],
                [np.sin(rad), np.cos(rad)]
            ])
            corners_rot = (rot_z @ corners_local.T).T
            corners_base = np.zeros((4, 3))
            corners_base[:, 0] = pos[0] + corners_rot[:, 0]
            corners_base[:, 1] = pos[1] + corners_rot[:, 1]
            corners_base[:, 2] = top_z

            poly_pts = []
            top_depths = []
            for c_pt in corners_base:
                u, v, z_d = self.camera.project_base_to_pixel(c_pt)
                poly_pts.append([u, v])
                top_depths.append(z_d)

            poly_np = np.array(poly_pts, dtype=np.int32)
            cv2.fillPoly(rgb, [poly_np], obj.bgr_color)
            cv2.polylines(rgb, [poly_np], True, (20, 20, 20), 2)

            # Fill depth map inside polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly_np], 255)
            avg_top_depth = float(np.mean(top_depths))
            depth[mask == 255] = avg_top_depth

        return rgb, depth
