"""Trajectory planning & finite state machine for pick-and-place manipulation."""
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from core.kinematics import ManipulatorKinematics


class ExecutionPhase(Enum):
    HOME = auto()
    APPROACH_PRE_GRASP = auto()
    DESCEND_GRASP = auto()
    CLOSE_GRIPPER = auto()
    LIFT_POST_GRASP = auto()
    TRANSIT_BIN = auto()
    DESCEND_PLACE = auto()
    OPEN_GRIPPER = auto()
    ASCEND_RETURN = auto()
    COMPLETED = auto()


@dataclass
class TrajectoryPoint:
    time_s: float
    joint_angles: np.ndarray  # [6]
    ee_position: np.ndarray   # [3]
    phase: ExecutionPhase
    gripper_closed: bool


class TrajectoryPlanner:
    """Generates minimum-jerk smooth trajectories using quintic polynomial blending."""

    @staticmethod
    def quintic_blend(q0: np.ndarray, q1: np.ndarray, steps: int) -> np.ndarray:
        """Interpolate between q0 and q1 using 5th-order polynomial (zero vel and acc at boundaries)."""
        tau = np.linspace(0.0, 1.0, steps)
        # s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        # Shape: [steps, num_joints]
        return np.outer(1.0 - s, q0) + np.outer(s, q1)


class PickAndPlaceStateMachine:
    """Coordinates the end-to-end pick-and-place lifecycle for a targeted workpiece."""

    def __init__(
        self,
        kinematics: ManipulatorKinematics,
        hover_height: float = 0.12,  # 12cm clearance above target
        dt: float = 0.05,            # 20 Hz interpolation rate
    ):
        self.kinematics = kinematics
        self.hover_height = hover_height
        self.dt = dt

    def plan_cycle(
        self,
        pick_pos: np.ndarray,
        place_pos: np.ndarray,
        time_per_segment: float = 1.0,
    ) -> List[TrajectoryPoint]:
        """Generate smooth trajectory for a full pick-and-place cycle.

        Waypoints:
        1. HOME
        2. Pre-Grasp: (pick_x, pick_y, pick_z + hover)
        3. Grasp:     (pick_x, pick_y, pick_z)
        4. Post-Grasp:(pick_x, pick_y, pick_z + hover)
        5. Pre-Place: (place_x, place_y, place_z + hover)
        6. Place:     (place_x, place_y, place_z)
        7. Post-Place:(place_x, place_y, place_z + hover)
        8. HOME
        """
        pick_p = np.asarray(pick_pos)[:3]
        place_p = np.asarray(place_pos)[:3]

        pre_pick_p = pick_p.copy()
        pre_pick_p[2] += self.hover_height

        post_pick_p = pre_pick_p.copy()

        pre_place_p = place_p.copy()
        pre_place_p[2] += self.hover_height

        post_place_p = pre_place_p.copy()

        # Solve IK for all critical waypoints
        q_home = self.kinematics.HOME_JOINTS.copy()
        q_pre_pick, _ = self.kinematics.solve_ik(pre_pick_p, initial_guess=q_home)
        q_pick, _ = self.kinematics.solve_ik(pick_p, initial_guess=q_pre_pick)
        q_post_pick = q_pre_pick.copy()

        q_pre_place, _ = self.kinematics.solve_ik(pre_place_p, initial_guess=q_post_pick)
        q_place, _ = self.kinematics.solve_ik(place_p, initial_guess=q_pre_place)
        q_post_place = q_pre_place.copy()

        # Segments definition: (start_q, end_q, phase, gripper_closed, duration_s)
        segments = [
            (q_home, q_pre_pick, ExecutionPhase.APPROACH_PRE_GRASP, False, time_per_segment * 1.2),
            (q_pre_pick, q_pick, ExecutionPhase.DESCEND_GRASP, False, time_per_segment * 0.8),
            (q_pick, q_pick, ExecutionPhase.CLOSE_GRIPPER, True, 0.4),  # Dwell to grip
            (q_pick, q_post_pick, ExecutionPhase.LIFT_POST_GRASP, True, time_per_segment * 0.8),
            (q_post_pick, q_pre_place, ExecutionPhase.TRANSIT_BIN, True, time_per_segment * 1.5),
            (q_pre_place, q_place, ExecutionPhase.DESCEND_PLACE, True, time_per_segment * 0.8),
            (q_place, q_place, ExecutionPhase.OPEN_GRIPPER, False, 0.4),  # Dwell to release
            (q_place, q_post_place, ExecutionPhase.ASCEND_RETURN, False, time_per_segment * 0.8),
            (q_post_place, q_home, ExecutionPhase.HOME, False, time_per_segment * 1.2),
        ]

        full_trajectory: List[TrajectoryPoint] = []
        current_time = 0.0

        for q_start, q_end, phase, grip, duration in segments:
            steps = max(2, int(np.round(duration / self.dt)))
            interpolated_joints = TrajectoryPlanner.quintic_blend(q_start, q_end, steps)

            for i in range(steps):
                q = interpolated_joints[i]
                ee_pos, _ = self.kinematics.forward_kinematics(q)
                full_trajectory.append(
                    TrajectoryPoint(
                        time_s=current_time,
                        joint_angles=q,
                        ee_position=ee_pos,
                        phase=phase,
                        gripper_closed=grip,
                    )
                )
                current_time += self.dt

        return full_trajectory
