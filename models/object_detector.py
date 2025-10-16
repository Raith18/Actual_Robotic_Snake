"""
YOLOv8 Object Detection and Tracking Module
Handles real-time object detection, tracking, and 3D integration.
"""

import cv2
import numpy as np
import torch
import time
import logging
from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict, deque
import yaml
from ultralytics import YOLO

class ObjectDetector:
    """
    YOLOv8-based object detection and tracking with 3D integration.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        Initialize YOLOv8 object detector.

        Args:
            config_path: Path to system configuration file
        """
        self.config = self._load_config(config_path)
        self.yolo_config = self.config['yolo']

        # Load YOLOv8 model
        self.model_path = self.yolo_config['model_size']
        self.model = None

        # Detection parameters
        self.confidence_threshold = self.yolo_config['confidence_threshold']
        self.iou_threshold = self.yolo_config['iou_threshold']
        self.max_detections = self.yolo_config['max_detections']

        # Tracking parameters
        self.track_objects = self.yolo_config['track_objects']
        self.track_buffer = self.yolo_config['track_buffer']

        # Object tracking data
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 0
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.track_buffer))

        # 3D integration
        self.object_depths: Dict[int, float] = {}
        self.object_point_clouds: Dict[int, np.ndarray] = {}

        # Performance tracking
        self.detection_times = []
        self.tracking_times = []
        self.max_performance_history = 100

        # Setup logging
        self.logger = logging.getLogger('ObjectDetector')

        # Initialize model
        self._load_model()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {
                'yolo': {
                    'model_size': 'yolov8n.pt',
                    'confidence_threshold': 0.5,
                    'iou_threshold': 0.45,
                    'max_detections': 100,
                    'track_objects': True,
                    'track_buffer': 30
                }
            }

    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            self.logger.info(f"Loading YOLOv8 model: {self.model_path}")

            # Check if model file exists locally
            if not self._model_exists():
                self.logger.warning(f"Model {self.model_path} not found locally, downloading...")
                self._download_model()

            # Load model
            self.model = YOLO(self.model_path)

            # Move to GPU if available
            if torch.cuda.is_available() and self.config['performance']['enable_cuda']:
                self.model.to('cuda')
                self.logger.info("YOLOv8 model loaded on GPU")
            else:
                self.logger.info("YOLOv8 model loaded on CPU")

        except Exception as e:
            self.logger.error(f"Error loading YOLOv8 model: {e}")
            self.model = None

    def _model_exists(self) -> bool:
        """Check if model file exists."""
        import os
        return os.path.exists(self.model_path)

    def _download_model(self):
        """Download YOLOv8 model if not available."""
        try:
            # This will download the model automatically when YOLO() is called
            # with a model name that doesn't exist locally
            pass
        except Exception as e:
            self.logger.error(f"Error downloading model: {e}")

    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in image using YOLOv8.

        Args:
            image: Input RGB image

        Returns:
            List of detection results
        """
        if self.model is None or image is None:
            return []

        start_time = time.time()

        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )

            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        detection = self._process_detection(boxes[i], result.names)
                        if detection:
                            detections.append(detection)

            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)

            if len(self.detection_times) > self.max_performance_history:
                self.detection_times.pop(0)

            self.logger.debug(f"Detected {len(detections)} objects in {detection_time:.3f}s")

            return detections

        except Exception as e:
            self.logger.error(f"Error in object detection: {e}")
            return []

    def _process_detection(self, box, class_names: Dict[int, str]) -> Optional[Dict[str, Any]]:
        """Process single detection result."""
        try:
            # Extract box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            # Create detection dictionary
            detection = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_names.get(class_id, f'class_{class_id}'),
                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                'width': x2 - x1,
                'height': y2 - y1,
                'area': (x2 - x1) * (y2 - y1)
            }

            return detection

        except Exception as e:
            self.logger.error(f"Error processing detection: {e}")
            return None

    def track_objects(self, detections: List[Dict[str, Any]], image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Track objects across frames.

        Args:
            detections: Current frame detections
            image: Current frame image

        Returns:
            Tracked detections with track IDs
        """
        if not self.track_objects or not detections:
            return detections

        start_time = time.time()

        try:
            # Simple tracking based on IoU and center distance
            current_centers = np.array([d['center'] for d in detections])

            # Update existing tracks
            for track_id, track_data in self.tracks.items():
                track_center = track_data['last_center']
                track_bbox = track_data['last_bbox']

                # Calculate distances to current detections
                if len(current_centers) > 0:
                    distances = np.linalg.norm(current_centers - track_center, axis=1)
                    ious = self._calculate_ious(track_bbox, [d['bbox'] for d in detections])

                    # Find best match
                    combined_scores = distances * 0.7 + (1 - ious) * 100  # Weighted combination
                    best_match_idx = np.argmin(combined_scores)

                    if combined_scores[best_match_idx] < 50:  # Threshold for matching
                        # Update track
                        matched_detection = detections[best_match_idx]
                        matched_detection['track_id'] = track_id

                        # Update track data
                        self.tracks[track_id].update({
                            'last_center': matched_detection['center'],
                            'last_bbox': matched_detection['bbox'],
                            'frames_since_update': 0,
                            'class_id': matched_detection['class_id'],
                            'class_name': matched_detection['class_name']
                        })

                        # Add to track history
                        self.track_history[track_id].append({
                            'center': matched_detection['center'],
                            'bbox': matched_detection['bbox'],
                            'timestamp': time.time()
                        })

                        # Remove from current detections to avoid double assignment
                        detections.pop(best_match_idx)
                        current_centers = np.delete(current_centers, best_match_idx, axis=0)

            # Create new tracks for unmatched detections
            for detection in detections:
                track_id = self.next_track_id
                detection['track_id'] = track_id

                self.tracks[track_id] = {
                    'last_center': detection['center'],
                    'last_bbox': detection['bbox'],
                    'frames_since_update': 0,
                    'class_id': detection['class_id'],
                    'class_name': detection['class_name'],
                    'created_at': time.time()
                }

                self.track_history[track_id].append({
                    'center': detection['center'],
                    'bbox': detection['bbox'],
                    'timestamp': time.time()
                })

                self.next_track_id += 1

            # Update frame counters
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['frames_since_update'] += 1

                # Remove old tracks
                if self.tracks[track_id]['frames_since_update'] > self.track_buffer:
                    del self.tracks[track_id]
                    del self.track_history[track_id]

            tracking_time = time.time() - start_time
            self.tracking_times.append(tracking_time)

            if len(self.tracking_times) > self.max_performance_history:
                self.tracking_times.pop(0)

            return detections

        except Exception as e:
            self.logger.error(f"Error in object tracking: {e}")
            return detections

    def _calculate_ious(self, bbox1: List[float], bboxes2: List[List[float]]) -> np.ndarray:
        """Calculate IoU between one bbox and multiple bboxes."""
        try:
            # Convert to numpy arrays
            b1 = np.array(bbox1)
            b2 = np.array(bboxes2)

            # Calculate intersection coordinates
            x1 = np.maximum(b1[0], b2[:, 0])
            y1 = np.maximum(b1[1], b2[:, 1])
            x2 = np.minimum(b1[2], b2[:, 2])
            y2 = np.minimum(b1[3], b2[:, 3])

            # Calculate intersection area
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

            # Calculate union area
            b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
            b2_area = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
            union = b1_area + b2_area - intersection

            # Calculate IoU
            ious = intersection / union
            ious[union == 0] = 0  # Handle zero union case

            return ious

        except Exception as e:
            self.logger.error(f"Error calculating IoU: {e}")
            return np.zeros(len(bboxes2))

    def integrate_with_depth(self,
                           detections: List[Dict[str, Any]],
                           depth_map: np.ndarray,
                           points_3d: np.ndarray) -> List[Dict[str, Any]]:
        """
        Integrate object detections with depth information.

        Args:
            detections: Object detections
            depth_map: Depth map
            points_3d: 3D point cloud

        Returns:
            Detections with depth information
        """
        try:
            for detection in detections:
                # Get object center in image coordinates
                center_x, center_y = detection['center']

                # Get depth at center point
                depth_y = int(center_y)
                depth_x = int(center_x)

                if 0 <= depth_y < depth_map.shape[0] and 0 <= depth_x < depth_map.shape[1]:
                    object_depth = depth_map[depth_y, depth_x]
                    detection['depth'] = float(object_depth)

                    # Calculate 3D position
                    fx, fy, cx, cy = 500.0, 500.0, depth_map.shape[1]/2, depth_map.shape[0]/2
                    x_3d = (depth_x - cx) * object_depth / fx
                    y_3d = (depth_y - cy) * object_depth / fy
                    z_3d = object_depth

                    detection['position_3d'] = [float(x_3d), float(y_3d), float(z_3d)]

                    # Extract object point cloud
                    object_points = self._extract_object_point_cloud(
                        detection, depth_map, points_3d
                    )
                    detection['point_cloud'] = object_points
                else:
                    detection['depth'] = 0.0
                    detection['position_3d'] = [0.0, 0.0, 0.0]
                    detection['point_cloud'] = np.array([]).reshape(0, 3)

            return detections

        except Exception as e:
            self.logger.error(f"Error integrating with depth: {e}")
            return detections

    def _extract_object_point_cloud(self,
                                 detection: Dict[str, Any],
                                 depth_map: np.ndarray,
                                 points_3d: np.ndarray) -> np.ndarray:
        """Extract point cloud points belonging to detected object."""
        try:
            x1, y1, x2, y2 = detection['bbox']

            # Create mask for object region
            mask = np.zeros(depth_map.shape, dtype=bool)
            mask[int(y1):int(y2), int(x1):int(x2)] = True

            # Find 3D points in object region
            # This is a simplified approach - in practice, you'd project properly
            height, width = depth_map.shape
            object_points = []

            for y in range(int(y1), int(y2)):
                for x in range(int(x1), int(x2)):
                    if 0 <= y < height and 0 <= x < width:
                        # Find corresponding 3D point (simplified)
                        depth = depth_map[y, x]
                        if depth > 0:
                            fx, fy, cx, cy = 500.0, 500.0, width/2, height/2
                            x_3d = (x - cx) * depth / fx
                            y_3d = (y - cy) * depth / fy
                            z_3d = depth
                            object_points.append([x_3d, y_3d, z_3d])

            return np.array(object_points)

        except Exception as e:
            self.logger.error(f"Error extracting object point cloud: {e}")
            return np.array([]).reshape(0, 3)

    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detections on image.

        Args:
            image: Input image
            detections: Object detections

        Returns:
            Image with drawn detections
        """
        try:
            annotated_image = image.copy()

            for detection in detections:
                # Draw bounding box
                bbox = detection['bbox']
                track_id = detection.get('track_id', -1)

                # Choose color based on class or track ID
                if track_id >= 0:
                    color = self._get_track_color(track_id)
                else:
                    color = (0, 255, 0)  # Green for untracked

                cv2.rectangle(annotated_image,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            color, 2)

                # Draw label
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                if track_id >= 0:
                    label += f" ID:{track_id}"

                cv2.putText(annotated_image, label,
                          (bbox[0], bbox[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw center point
                center = detection['center']
                cv2.circle(annotated_image, (int(center[0]), int(center[1])), 3, color, -1)

                # Draw depth if available
                if 'depth' in detection:
                    depth_text = f"Depth: {detection['depth']:.2f}m"
                    cv2.putText(annotated_image, depth_text,
                              (bbox[0], bbox[3] + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return annotated_image

        except Exception as e:
            self.logger.error(f"Error drawing detections: {e}")
            return image

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID."""
        np.random.seed(track_id)
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
        return colors[track_id % len(colors)]

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection and tracking statistics."""
        return {
            'num_active_tracks': len(self.tracks),
            'avg_detection_time': sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0.0,
            'avg_tracking_time': sum(self.tracking_times) / len(self.tracking_times) if self.tracking_times else 0.0,
            'total_detections': len(self.detection_times)
        }

    def clear_tracks(self):
        """Clear all object tracks."""
        self.tracks.clear()
        self.track_history.clear()
        self.next_track_id = 0
        self.logger.info("Object tracks cleared")

    def get_track_history(self, track_id: int) -> List[Dict[str, Any]]:
        """Get tracking history for specific track ID."""
        if track_id in self.track_history:
            return list(self.track_history[track_id])
        return []

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_path': self.model_path,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'confidence_threshold': self.confidence_threshold,
            'tracking_enabled': self.track_objects,
            'num_classes': len(self.model.names) if self.model else 0
        }