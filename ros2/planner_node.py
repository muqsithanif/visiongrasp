"""ROS 2 Node for pick-and-place trajectory generation and execution coordination."""
from typing import List, Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from geometry_msgs.msg import Pose
    from std_msgs.msg import String
    HAVE_ROS2 = True
except ImportError:
    HAVE_ROS2 = False
    class Node:
        def __init__(self, name: str):
            self.name = name

from core.kinematics import ManipulatorKinematics
from core.trajectory import PickAndPlaceStateMachine, TrajectoryPoint


class MotionPlannerNode(Node):
    """ROS 2 Node planning smooth arm trajectories for target grasp positions."""

    def __init__(self):
        super().__init__("motion_planner_node")
        self.kinematics = ManipulatorKinematics()
        self.state_machine = PickAndPlaceStateMachine(self.kinematics)

        if HAVE_ROS2:
            self.joint_traj_pub = self.create_publisher(
                JointTrajectory, "/arm_controller/joint_trajectory", 10
            )
            self.get_logger().info("MotionPlannerNode ready.")

    def plan_and_execute(self, pick_pos: np.ndarray, place_pos: np.ndarray) -> List[TrajectoryPoint]:
        """Compute full waypoint trajectory for given pick and place coordinates."""
        return self.state_machine.plan_cycle(pick_pos, place_pos)
