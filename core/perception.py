"""Vision perception pipeline: 2D segmentation, depth extraction, and 3D grasp pose estimation."""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

from core.camera import CameraModel


@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    pixel_centroid: Tuple[int, int]
    depth_m: float
    position_3d_base: np.ndarray  # [x, y, z] in meters
    yaw_deg: float  # orientation angle around base Z axis
    contour: np.ndarray
    bbox_xywh: Tuple[int, int, int, int]


class VisionPerception:
    """Perception engine extracting oriented 3D grasp targets from RGB-D frames."""

    def __init__(self, camera: CameraModel):
        self.camera = camera
        # Default color filter ranges in HSV space
        self.color_ranges: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
            "red_block": [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255])),
            ],
            "blue_block": [
                (np.array([100, 100, 100]), np.array([130, 255, 255])),
            ],
            "green_block": [
                (np.array([35, 80, 80]), np.array([85, 255, 255])),
            ],
        }

    def register_color_class(
        self,
        name: str,
        hsv_lower: np.ndarray,
        hsv_upper: np.ndarray,
    ) -> None:
        """Register or override a target color class."""
        self.color_ranges[name] = [(hsv_lower, hsv_upper)]

    def detect(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        min_area: int = 150,
    ) -> List[DetectedObject]:
        """Process RGB and Depth frames, returning list of detected 3D oriented objects."""
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        detections: List[DetectedObject] = []

        for class_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                sub_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, sub_mask)

            # Morphological cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                moments = cv2.moments(cnt)
                if moments["m00"] == 0:
                    continue
                uc = int(np.round(moments["m10"] / moments["m00"]))
                vc = int(np.round(moments["m01"] / moments["m00"]))

                # Oriented bounding box for grasp angle
                rect = cv2.minAreaRect(cnt)
                (rx, ry), (width, height), angle = rect

                # Normalize orientation angle
                if width < height:
                    angle += 90.0

                # Sample median depth from contour mask (eliminates noisy edge outliers)
                cnt_mask = np.zeros(depth_image.shape, dtype=np.uint8)
                cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
                valid_depths = depth_image[cnt_mask == 255]
                valid_depths = valid_depths[np.isfinite(valid_depths) & (valid_depths > 0.05)]

                if len(valid_depths) == 0:
                    continue
                depth_val = float(np.median(valid_depths))

                # De-project centroid to 3D base coordinates
                pos_3d = self.camera.deproject_pixel_to_base(uc, vc, depth_val)

                # Camera to base yaw alignment
                # For our top-down camera with optical X -> +Y and optical Y -> +X,
                # image plane rotation maps to base Z rotation:
                yaw_base = float(angle) % 180.0
                if yaw_base > 90.0:
                    yaw_base -= 180.0

                x, y, w, h = cv2.boundingRect(cnt)

                detections.append(
                    DetectedObject(
                        class_name=class_name,
                        confidence=min(1.0, float(area) / 2000.0),
                        pixel_centroid=(uc, vc),
                        depth_m=depth_val,
                        position_3d_base=pos_3d,
                        yaw_deg=yaw_base,
                        contour=cnt,
                        bbox_xywh=(x, y, w, h),
                    )
                )

        # Sort detections from left to right (X coordinate)
        detections.sort(key=lambda d: d.position_3d_base[0])
        return detections

    def draw_annotations(
        self,
        rgb_image: np.ndarray,
        detections: List[DetectedObject],
    ) -> np.ndarray:
        """Draw bounding boxes, centroids, and estimated 3D world coordinates on image."""
        vis = rgb_image.copy()
        for det in detections:
            cv2.drawContours(vis, [det.contour], -1, (0, 255, 0), 2)
            uc, vc = det.pixel_centroid
            cv2.circle(vis, (uc, vc), 4, (0, 0, 255), -1)

            # Coordinate annotation text
            x, y, z = det.position_3d_base
            text1 = f"{det.class_name}"
            text2 = f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}m"
            text3 = f"Yaw:{det.yaw_deg:.1f}deg"

            bx, by, _, _ = det.bbox_xywh
            cv2.putText(vis, text1, (bx, max(15, by - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2)
            cv2.putText(vis, text1, (bx, max(15, by - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.putText(vis, text2, (bx, max(30, by - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
            cv2.putText(vis, text3, (bx, max(42, by + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        return vis
