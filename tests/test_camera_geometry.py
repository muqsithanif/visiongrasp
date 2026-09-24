"""Unit tests for pinhole camera model, intrinsics, and 3D coordinate transformations."""
import numpy as np
import pytest
from core.camera import CameraModel, CameraIntrinsics


def test_camera_intrinsics_matrix():
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    mat = intrinsics.matrix
    assert mat.shape == (3, 3)
    assert mat[0, 0] == 500.0
    assert mat[1, 1] == 500.0
    assert mat[0, 2] == 320.0
    assert mat[1, 2] == 240.0
    assert mat[2, 2] == 1.0


def test_extrinsic_orthonormality():
    camera = CameraModel()
    rot = camera.t_base_cam[:3, :3]
    # Check R * R^T = I
    identity = np.eye(3)
    np.testing.assert_allclose(rot @ rot.T, identity, atol=1e-7)
    # Check det(R) = 1 (valid right-handed rotation)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-7)


def test_projection_deprojection_roundtrip():
    """Verify projecting 3D point to 2D pixel + depth, then deprojecting recovers the exact original 3D point."""
    camera = CameraModel()
    original_points = [
        np.array([0.35, -0.15, 0.05]),
        np.array([0.50, 0.20, 0.02]),
        np.array([0.40, 0.00, 0.10]),
    ]

    for pt in original_points:
        u, v, depth = camera.project_base_to_pixel(pt)
        recovered_pt = camera.deproject_pixel_to_base(u, v, depth)
        np.testing.assert_allclose(recovered_pt, pt, atol=2e-3, err_msg=f"Failed roundtrip for {pt}")


def test_behind_camera_rejection():
    camera = CameraModel()
    # Point high above the camera (camera is at Z=0.90 looking down)
    point_above = np.array([0.45, 0.0, 1.50])
    with pytest.raises(ValueError, match="behind or at camera plane"):
        camera.project_base_to_pixel(point_above)
