"""Unit tests for UR5 forward and inverse kinematics."""
import numpy as np
import pytest
from core.kinematics import ManipulatorKinematics


@pytest.fixture
def kinematics():
    return ManipulatorKinematics()


def test_home_forward_kinematics(kinematics):
    pos, rot = kinematics.forward_kinematics(kinematics.HOME_JOINTS)
    assert pos.shape == (3,)
    assert rot.shape == (3, 3)
    # Check that home pose is above table
    assert pos[2] > 0.10


def test_workspace_reachability(kinematics):
    # Reachable table workspace coordinate
    assert kinematics.is_reachable(np.array([0.45, 0.10, 0.15])) is True
    # Too far (out of reach > 0.85m)
    assert kinematics.is_reachable(np.array([1.20, 0.00, 0.15])) is False
    # Below floor boundary
    assert kinematics.is_reachable(np.array([0.40, 0.00, -0.20])) is False


def test_inverse_kinematics_accuracy(kinematics):
    targets = [
        np.array([0.40, 0.15, 0.10]),
        np.array([0.45, -0.15, 0.05]),
        np.array([0.35, 0.25, 0.12]),
    ]
    for target in targets:
        joint_sol, err = kinematics.solve_ik(target)
        assert err < 0.015, f"IK error too large ({err*1000:.1f} mm) for target {target}"
        fk_pos, _ = kinematics.forward_kinematics(joint_sol)
        np.testing.assert_allclose(fk_pos, target, atol=0.015)
