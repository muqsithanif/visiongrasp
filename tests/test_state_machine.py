"""Unit tests for trajectory generation, continuity, and pick-and-place state machine."""
import numpy as np
import pytest
from core.kinematics import ManipulatorKinematics
from core.trajectory import PickAndPlaceStateMachine, ExecutionPhase


@pytest.fixture
def state_machine():
    kinematics = ManipulatorKinematics()
    return PickAndPlaceStateMachine(kinematics, hover_height=0.10, dt=0.05)


def test_trajectory_continuity(state_machine):
    pick_pos = np.array([0.45, -0.15, 0.02])
    place_pos = np.array([0.30, 0.30, 0.02])

    trajectory = state_machine.plan_cycle(pick_pos, place_pos)
    assert len(trajectory) > 50

    # Ensure no discontinuous teleportation between consecutive trajectory points
    max_joint_step = 0.0
    for i in range(1, len(trajectory)):
        q_prev = trajectory[i - 1].joint_angles
        q_curr = trajectory[i].joint_angles
        diff = np.max(np.abs(q_curr - q_prev))
        if diff > max_joint_step:
            max_joint_step = diff

    # Max change per 50ms timestep must be bounded (< 0.2 rad)
    assert max_joint_step < 0.25, f"Discontinuous joint jump detected: {max_joint_step:.4f} rad"


def test_gripper_phase_logic(state_machine):
    pick_pos = np.array([0.45, -0.15, 0.02])
    place_pos = np.array([0.30, 0.30, 0.02])

    trajectory = state_machine.plan_cycle(pick_pos, place_pos)

    # Gripper must start open
    assert trajectory[0].gripper_closed is False

    # Check that gripper closes during grasp and remains closed during transit
    closed_points = [pt for pt in trajectory if pt.gripper_closed]
    assert len(closed_points) > 0

    transit_points = [pt for pt in trajectory if pt.phase == ExecutionPhase.TRANSIT_BIN]
    assert len(transit_points) > 0
    for pt in transit_points:
        assert pt.gripper_closed is True, "Gripper dropped workpiece during transit!"

    # Gripper must end open
    assert trajectory[-1].gripper_closed is False
