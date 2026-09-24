"""ROS 2 Node for real-time camera processing and 3D workpiece pose publishing."""
import json
from typing import List, Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseArray, Pose
    from std_msgs.msg import Header
    HAVE_ROS2 = True
except ImportError:
    HAVE_ROS2 = False
    # Mock base class when ROS 2 environment is not directly active
    class Node:
        def __init__(self, name: str):
            self.name = name

from core.camera import CameraModel, CameraIntrinsics
from core.perception import VisionPerception, DetectedObject


class PerceptionNode(Node):
    """ROS 2 Node that receives RGB-D image streams and publishes detected 3D grasp poses."""

    def __init__(self):
        super().__init__("perception_node")
        self.camera_model = CameraModel()
        self.perception = VisionPerception(self.camera_model)
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None

        if HAVE_ROS2:
            self.create_subscription(Image, "/camera/color/image_raw", self._on_rgb, 10)
            self.create_subscription(Image, "/camera/depth/image_raw", self._on_depth, 10)
            self.pose_pub = self.create_publisher(PoseArray, "/vision/target_poses_3d", 10)
            self.get_logger().info("PerceptionNode initialized. Listening to camera streams.")

    def _on_rgb(self, msg) -> None:
        pass

    def _on_depth(self, msg) -> None:
        pass

    def process_frame(self, rgb: np.ndarray, depth: np.ndarray) -> List[DetectedObject]:
        """Core detection routine, independent of middleware transport."""
        return self.perception.detect(rgb, depth)
