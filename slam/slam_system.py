"""
SLAM System Module
Implements Simultaneous Localization and Mapping for monocular depth estimation.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, Any, List
from collections import deque
import yaml
from scipy.spatial.transform import Rotation

class SLAMSystem:
    """
    Monocular depth-based SLAM system with keyframe management and pose estimation.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        Initialize SLAM system.

        Args:
            config_path: Path to system configuration file
        """
        self.config = self._load_config(config_path)
        self.slam_config = self.config['slam']

        # Feature detection and matching
        self.feature_detector = self.slam_config['feature_detector']
        self.num_features = self.slam_config['num_features']

        # Keyframe management
        self.keyframe_distance = self.slam_config['keyframe_distance']
        self.keyframe_angle = self.slam_config['keyframe_angle']

        # Bundle adjustment
        self.bundle_adjustment = self.slam_config['bundle_adjustment']
        self.ba_iterations = self.slam_config['ba_iterations']

        # Camera pose and trajectory
        self.current_pose: Optional[np.ndarray] = None
        self.pose_history: List[np.ndarray] = []
        self.trajectory_points: List[np.ndarray] = []

        # Keyframes storage
        self.keyframes: List[Dict[str, Any]] = []
        self.keyframe_descriptors: List[np.ndarray] = []

        # Map management
        self.map_points: Optional[np.ndarray] = None
        self.map_colors: Optional[np.ndarray] = None
        self.map_point_ids: List[int] = []

        # Feature detector
        self.orb = None
        self.bf_matcher = None

        # Tracking state
        self.tracking_state = "NOT_INITIALIZED"
        self.last_keyframe_idx = -1

        # Performance tracking
        self.tracking_times = []
        self.mapping_times = []

        # Setup logging
        self.logger = logging.getLogger('SLAMSystem')

        # Initialize components
        self._initialize_feature_detector()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {
                'slam': {
                    'feature_detector': 'ORB',
                    'num_features': 1000,
                    'keyframe_distance': 0.1,
                    'keyframe_angle': 10,
                    'bundle_adjustment': True,
                    'ba_iterations': 10
                }
            }

    def _initialize_feature_detector(self):
        """Initialize feature detection and matching components."""
        try:
            if self.feature_detector == 'ORB':
                self.orb = cv2.ORB_create(nfeatures=self.num_features)
                self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                self.logger.warning(f"Unsupported feature detector: {self.feature_detector}")
                self.orb = cv2.ORB_create(nfeatures=self.num_features)
                self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            self.logger.info("Feature detector initialized")

        except Exception as e:
            self.logger.error(f"Error initializing feature detector: {e}")

    def initialize(self, first_frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> bool:
        """
        Initialize SLAM system with first frame.

        Args:
            first_frame: First RGB frame
            depth_map: Optional depth map

        Returns:
            True if initialization successful
        """
        try:
            # Extract features from first frame
            keypoints, descriptors = self._extract_features(first_frame)

            if len(keypoints) < 10:
                self.logger.error("Insufficient features for initialization")
                return False

            # Create initial keyframe
            initial_keyframe = {
                'frame_id': 0,
                'pose': np.eye(4),  # Identity pose
                'keypoints': keypoints,
                'descriptors': descriptors,
                'image': first_frame.copy(),
                'depth_map': depth_map.copy() if depth_map is not None else None,
                'timestamp': time.time()
            }

            self.keyframes.append(initial_keyframe)
            self.keyframe_descriptors.append(descriptors)

            # Set initial pose
            self.current_pose = np.eye(4)
            self.pose_history.append(self.current_pose.copy())
            self.trajectory_points.append(self.current_pose[:3, 3])

            # Initialize map with first frame points
            if depth_map is not None:
                self._initialize_map(first_frame, depth_map, keypoints, self.current_pose)

            self.tracking_state = "INITIALIZED"
            self.last_keyframe_idx = 0

            self.logger.info("SLAM system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing SLAM: {e}")
            return False

    def _extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extract features from image."""
        try:
            if self.orb is None:
                return [], np.array([])

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Extract features
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            return keypoints, descriptors

        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return [], np.array([])

    def _initialize_map(self, image: np.ndarray, depth_map: np.ndarray,
                       keypoints: List[cv2.KeyPoint], pose: np.ndarray):
        """Initialize 3D map with first frame."""
        try:
            map_points = []
            map_colors = []
            point_ids = []

            fx, fy, cx, cy = 500.0, 500.0, image.shape[1]/2, image.shape[0]/2

            for i, kp in enumerate(keypoints):
                x, y = kp.pt

                if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                    depth = depth_map[int(y), int(x)]

                    if depth > 0:
                        # Convert to 3D point
                        X = (x - cx) * depth / fx
                        Y = (y - cy) * depth / fy
                        Z = depth

                        point_3d = np.array([X, Y, Z])
                        point_color = image[int(y), int(x)] / 255.0

                        map_points.append(point_3d)
                        map_colors.append(point_color)
                        point_ids.append(i)

            if map_points:
                self.map_points = np.array(map_points)
                self.map_colors = np.array(map_colors)
                self.map_point_ids = point_ids

                self.logger.info(f"Initialized map with {len(map_points)} points")

        except Exception as e:
            self.logger.error(f"Error initializing map: {e}")

    def track_frame(self, current_frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> bool:
        """
        Track current frame and update pose.

        Args:
            current_frame: Current RGB frame
            depth_map: Current depth map

        Returns:
            True if tracking successful
        """
        if self.tracking_state == "NOT_INITIALIZED":
            return False

        start_time = time.time()

        try:
            # Extract features from current frame
            keypoints, descriptors = self._extract_features(current_frame)

            if len(keypoints) < 10:
                self.logger.warning("Insufficient features for tracking")
                return False

            # Match features with last keyframe
            matches = self._match_features(descriptors)

            if len(matches) < 10:
                self.logger.warning("Insufficient matches for pose estimation")
                return False

            # Estimate pose using PnP or similar
            success, pose = self._estimate_pose(keypoints, matches, current_frame.shape)

            if success:
                self.current_pose = pose
                self.pose_history.append(pose.copy())
                self.trajectory_points.append(pose[:3, 3])

                # Check if we should create a new keyframe
                if self._should_create_keyframe():
                    self._create_keyframe(current_frame, depth_map, keypoints, descriptors)

                # Update map with new observations
                if depth_map is not None:
                    self._update_map(current_frame, depth_map, keypoints, pose)

                tracking_time = time.time() - start_time
                self.tracking_times.append(tracking_time)

                return True
            else:
                self.logger.warning("Pose estimation failed")
                return False

        except Exception as e:
            self.logger.error(f"Error in frame tracking: {e}")
            return False

    def _match_features(self, descriptors: np.ndarray) -> List[cv2.DMatch]:
        """Match features with keyframes."""
        try:
            if len(self.keyframes) == 0 or descriptors is None:
                return []

            # Match with last keyframe
            last_kf = self.keyframes[-1]
            last_descriptors = last_kf['descriptors']

            if last_descriptors is None or len(last_descriptors) == 0:
                return []

            # Match descriptors
            matches = self.bf_matcher.match(descriptors, last_descriptors)

            # Filter by distance
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:min(100, len(matches))]  # Keep best matches

            return good_matches

        except Exception as e:
            self.logger.error(f"Error matching features: {e}")
            return []

    def _estimate_pose(self, keypoints: List[cv2.KeyPoint], matches: List[cv2.DMatch],
                      image_shape: Tuple[int, int]) -> Tuple[bool, np.ndarray]:
        """Estimate camera pose using matched features."""
        try:
            if len(matches) < 10:
                return False, np.eye(4)

            # Get matched keypoints
            query_kp = [keypoints[match.queryIdx].pt for match in matches]
            train_kp = [self.keyframes[-1]['keypoints'][match.trainIdx].pt for match in matches]

            # Convert to numpy arrays
            query_pts = np.array(query_kp, dtype=np.float32)
            train_pts = np.array(train_kp, dtype=np.float32)

            # Camera intrinsic matrix
            fx, fy, cx, cy = 500.0, 500.0, image_shape[1]/2, image_shape[0]/2
            K = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float32)

            # Get 3D points for matched features (simplified - assumes known depth)
            object_points = []
            image_points = []

            for i, match in enumerate(matches):
                train_idx = match.trainIdx
                if train_idx < len(self.map_points):
                    # Use existing 3D point
                    object_points.append(self.map_points[train_idx])
                    image_points.append(query_pts[i])

            if len(object_points) < 4:
                return False, np.eye(4)

            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)

            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, K, None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # Convert to pose matrix
                R, _ = cv2.Rodrigues(rvec)
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = tvec.flatten()

                return True, pose
            else:
                return False, np.eye(4)

        except Exception as e:
            self.logger.error(f"Error estimating pose: {e}")
            return False, np.eye(4)

    def _should_create_keyframe(self) -> bool:
        """Check if current frame should be a keyframe."""
        try:
            if len(self.keyframes) == 0:
                return True

            # Check distance from last keyframe
            last_pose = self.keyframes[-1]['pose']
            distance = np.linalg.norm(self.current_pose[:3, 3] - last_pose[:3, 3])

            if distance > self.keyframe_distance:
                return True

            # Check rotation from last keyframe
            last_rotation = Rotation.from_matrix(last_pose[:3, :3])
            current_rotation = Rotation.from_matrix(self.current_pose[:3, :3])
            rotation_diff = (last_rotation.inv() * current_rotation).magnitude()

            if np.degrees(rotation_diff) > self.keyframe_angle:
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking keyframe condition: {e}")
            return False

    def _create_keyframe(self, frame: np.ndarray, depth_map: np.ndarray,
                        keypoints: List[cv2.KeyPoint], descriptors: np.ndarray):
        """Create new keyframe."""
        try:
            keyframe = {
                'frame_id': len(self.keyframes),
                'pose': self.current_pose.copy(),
                'keypoints': keypoints,
                'descriptors': descriptors,
                'image': frame.copy(),
                'depth_map': depth_map.copy() if depth_map is not None else None,
                'timestamp': time.time()
            }

            self.keyframes.append(keyframe)
            self.keyframe_descriptors.append(descriptors)
            self.last_keyframe_idx = len(self.keyframes) - 1

            self.logger.info(f"Created keyframe {len(self.keyframes)}")

        except Exception as e:
            self.logger.error(f"Error creating keyframe: {e}")

    def _update_map(self, frame: np.ndarray, depth_map: np.ndarray,
                   keypoints: List[cv2.KeyPoint], pose: np.ndarray):
        """Update 3D map with new observations."""
        try:
            if self.map_points is None:
                return

            # Add new points from current frame
            new_points = []
            new_colors = []
            new_ids = []

            fx, fy, cx, cy = 500.0, 500.0, frame.shape[1]/2, frame.shape[0]/2

            for i, kp in enumerate(keypoints):
                x, y = kp.pt

                if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                    depth = depth_map[int(y), int(x)]

                    if depth > 0:
                        # Transform to world coordinates
                        point_camera = np.array([
                            (x - cx) * depth / fx,
                            (y - cy) * depth / fy,
                            depth,
                            1
                        ])

                        point_world = pose @ point_camera
                        point_3d = point_world[:3]
                        point_color = frame[int(y), int(x)] / 255.0

                        new_points.append(point_3d)
                        new_colors.append(point_color)
                        new_ids.append(len(self.map_point_ids) + len(new_points) - 1)

            # Add new points to map
            if new_points:
                if self.map_points is None:
                    self.map_points = np.array(new_points)
                    self.map_colors = np.array(new_colors)
                else:
                    self.map_points = np.vstack([self.map_points, new_points])
                    self.map_colors = np.vstack([self.map_colors, new_colors])

                self.map_point_ids.extend(new_ids)

        except Exception as e:
            self.logger.error(f"Error updating map: {e}")

    def get_current_pose(self) -> Optional[np.ndarray]:
        """Get current camera pose."""
        return self.current_pose.copy() if self.current_pose is not None else None

    def get_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get camera trajectory."""
        return self.pose_history, self.trajectory_points

    def get_map(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get current 3D map."""
        return self.map_points.copy() if self.map_points is not None else None, \
               self.map_colors.copy() if self.map_colors is not None else None

    def get_tracking_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            'tracking_state': self.tracking_state,
            'num_keyframes': len(self.keyframes),
            'num_map_points': len(self.map_point_ids) if self.map_point_ids else 0,
            'avg_tracking_time': sum(self.tracking_times) / len(self.tracking_times) if self.tracking_times else 0.0,
            'current_pose': self.current_pose.tolist() if self.current_pose is not None else None
        }

    def reset(self):
        """Reset SLAM system."""
        self.current_pose = None
        self.pose_history.clear()
        self.trajectory_points.clear()
        self.keyframes.clear()
        self.keyframe_descriptors.clear()
        self.map_points = None
        self.map_colors = None
        self.map_point_ids.clear()
        self.tracking_state = "NOT_INITIALIZED"
        self.last_keyframe_idx = -1
        self.tracking_times.clear()
        self.mapping_times.clear()

        self.logger.info("SLAM system reset")

    def save_map(self, filename: str):
        """Save current map to file."""
        try:
            map_data = {
                'points': self.map_points.tolist() if self.map_points is not None else [],
                'colors': self.map_colors.tolist() if self.map_colors is not None else [],
                'point_ids': self.map_point_ids,
                'keyframes': len(self.keyframes),
                'trajectory': [pose.tolist() for pose in self.pose_history]
            }

            import json
            with open(filename, 'w') as f:
                json.dump(map_data, f, indent=2)

            self.logger.info(f"Map saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error saving map: {e}")

    def load_map(self, filename: str) -> bool:
        """Load map from file."""
        try:
            import json
            with open(filename, 'r') as f:
                map_data = json.load(f)

            self.map_points = np.array(map_data['points'])
            self.map_colors = np.array(map_data['colors'])
            self.map_point_ids = map_data['point_ids']
            self.pose_history = [np.array(pose) for pose in map_data['trajectory']]

            self.logger.info(f"Map loaded from {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading map: {e}")
            return False