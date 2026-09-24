"""Unit tests for perception pipeline: object segmentation, centroid extraction, and 3D positioning."""
import numpy as np
import pytest
from core.camera import CameraModel, SyntheticWorkspace, SyntheticObject
from core.perception import VisionPerception


@pytest.fixture
def setup_scene():
    camera = CameraModel()
    workspace = SyntheticWorkspace(camera, table_z=0.0)
    target_obj = SyntheticObject(
        name="test_red_block",
        color_name="red_block",
        bgr_color=(30, 30, 220),
        hsv_lower=np.array([0, 100, 100]),
        hsv_upper=np.array([10, 255, 255]),
        position_base=np.array([0.45, -0.10, 0.02]),
        dimensions=(0.04, 0.04, 0.04),
        yaw_deg=0.0,
    )
    rgb, depth = workspace.generate_scene([target_obj])
    return camera, target_obj, rgb, depth


def test_object_detection_and_classification(setup_scene):
    camera, target_obj, rgb, depth = setup_scene
    perception = VisionPerception(camera)
    detections = perception.detect(rgb, depth)

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "red_block"
    assert det.confidence > 0.0


def test_3d_position_estimation_accuracy(setup_scene):
    camera, target_obj, rgb, depth = setup_scene
    perception = VisionPerception(camera)
    detections = perception.detect(rgb, depth)

    assert len(detections) == 1
    estimated_pos = detections[0].position_3d_base
    gt_pos = target_obj.position_base

    # Check X and Y accuracy within 5 mm
    assert np.isclose(estimated_pos[0], gt_pos[0], atol=0.005)
    assert np.isclose(estimated_pos[1], gt_pos[1], atol=0.005)
    # Check top surface height
    assert estimated_pos[2] > gt_pos[2]  # Top surface is higher than center
