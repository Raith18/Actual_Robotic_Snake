"""
Advanced Point Cloud Processing Module
Implements sophisticated 3D point cloud processing algorithms including
voxel grid downsampling, statistical outlier removal, and Kalman filtering.
"""

import numpy as np
import open3d as o3d
import time
import logging
from typing import Optional, Tuple, Dict, Any, List
from scipy.spatial import KDTree
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import yaml

class PointCloudProcessor:
    """
    Advanced point cloud processing with multiple filtering and optimization techniques.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        Initialize point cloud processor.

        Args:
            config_path: Path to system configuration file
        """
        self.config = self._load_config(config_path)
        self.pc_config = self.config['point_cloud']

        # Kalman filters for temporal smoothing
        self.kalman_filters: Dict[str, KalmanFilter] = {}
        self.filter_states: Dict[str, np.ndarray] = {}

        # Processing history for temporal consistency
        self.processing_history: List[Dict[str, Any]] = []
        self.max_history = 100

        # Performance tracking
        self.processing_times = []

        # Setup logging
        self.logger = logging.getLogger('PointCloudProcessor')

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {
                'point_cloud': {
                    'voxel_size': 0.01,
                    'outlier_neighbors': 50,
                    'outlier_std_ratio': 1.0,
                    'kalman_process_noise': 0.01,
                    'kalman_measurement_noise': 0.1
                }
            }

    def process_point_cloud(self,
                          points: np.ndarray,
                          colors: Optional[np.ndarray] = None,
                          enable_voxel_filter: bool = True,
                          enable_outlier_removal: bool = True,
                          enable_kalman_filter: bool = True,
                          enable_normal_estimation: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply comprehensive point cloud processing pipeline.

        Args:
            points: Input point cloud
            colors: Point colors
            enable_voxel_filter: Apply voxel grid filtering
            enable_outlier_removal: Remove statistical outliers
            enable_kalman_filter: Apply Kalman filtering
            enable_normal_estimation: Estimate surface normals

        Returns:
            Tuple of processed (points, colors)
        """
        if points is None or len(points) == 0:
            return None, None

        start_time = time.time()
        original_points = points.copy()

        try:
            processed_points = points
            processed_colors = colors

            # Step 1: Voxel grid downsampling
            if enable_voxel_filter:
                processed_points = self.apply_adaptive_voxel_filter(processed_points)
                if processed_colors is not None and len(processed_points) < len(colors):
                    # Downsample colors to match filtered points
                    indices = np.random.choice(len(colors), len(processed_points), replace=True)
                    processed_colors = colors[indices]

            # Step 2: Statistical outlier removal
            if enable_outlier_removal:
                processed_points = self.remove_adaptive_outliers(processed_points)

            # Step 3: Kalman filtering for temporal smoothing
            if enable_kalman_filter:
                processed_points = self.apply_temporal_kalman_filter(processed_points)

            # Step 4: Normal estimation (if requested)
            if enable_normal_estimation:
                normals = self.estimate_adaptive_normals(processed_points)
                # Store normals for later use
                self.current_normals = normals

            # Store processing result
            processing_result = {
                'timestamp': time.time(),
                'original_points': len(original_points),
                'processed_points': len(processed_points) if processed_points is not None else 0,
                'processing_time': time.time() - start_time
            }
            self.processing_history.append(processing_result)

            if len(self.processing_history) > self.max_history:
                self.processing_history.pop(0)

            self.processing_times.append(time.time() - start_time)

            self.logger.debug(f"Processed point cloud: {len(original_points)} -> {len(processed_points)} points")

            return processed_points, processed_colors

        except Exception as e:
            self.logger.error(f"Error in point cloud processing: {e}")
            return original_points, colors

    def apply_adaptive_voxel_filter(self, points: np.ndarray, base_voxel_size: float = None) -> np.ndarray:
        """
        Apply adaptive voxel grid filtering based on point density.

        Args:
            points: Input point cloud
            base_voxel_size: Base voxel size (uses config default if None)

        Returns:
            Filtered point cloud
        """
        if base_voxel_size is None:
            base_voxel_size = self.pc_config['voxel_size']

        try:
            # Calculate point density
            if len(points) < 1000:
                voxel_size = base_voxel_size * 0.5  # Smaller voxels for sparse clouds
            elif len(points) > 50000:
                voxel_size = base_voxel_size * 2.0  # Larger voxels for dense clouds
            else:
                voxel_size = base_voxel_size

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Apply voxel grid filter
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

            return np.asarray(downsampled_pcd.points)

        except Exception as e:
            self.logger.error(f"Error in adaptive voxel filtering: {e}")
            return points

    def remove_adaptive_outliers(self, points: np.ndarray) -> np.ndarray:
        """
        Remove outliers using adaptive statistical methods.

        Args:
            points: Input point cloud

        Returns:
            Filtered point cloud
        """
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Calculate optimal parameters based on point cloud size
            num_points = len(points)
            if num_points < 1000:
                neighbors = max(10, num_points // 10)
                std_ratio = 1.5
            elif num_points > 10000:
                neighbors = 100
                std_ratio = 0.8
            else:
                neighbors = self.pc_config['outlier_neighbors']
                std_ratio = self.pc_config['outlier_std_ratio']

            # Remove outliers
            filtered_pcd, inlier_mask = pcd.remove_statistical_outlier(
                nb_neighbors=neighbors,
                std_ratio=std_ratio
            )

            return np.asarray(filtered_pcd.points)

        except Exception as e:
            self.logger.error(f"Error removing adaptive outliers: {e}")
            return points

    def apply_temporal_kalman_filter(self, points: np.ndarray) -> np.ndarray:
        """
        Apply Kalman filtering for temporal smoothing of point cloud motion.

        Args:
            points: Current point cloud

        Returns:
            Temporally smoothed point cloud
        """
        try:
            if len(points) == 0:
                return points

            # Calculate current centroid and bounding box
            current_centroid = np.mean(points, axis=0)
            current_bbox = self._calculate_bounding_box(points)

            # Initialize Kalman filters if needed
            filter_keys = ['centroid_x', 'centroid_y', 'centroid_z', 'bbox']
            for key in filter_keys:
                if key not in self.kalman_filters:
                    self._initialize_kalman_filter(key)

            # Apply Kalman filtering to centroid
            filtered_centroid = np.zeros(3)
            for i, key in enumerate(['centroid_x', 'centroid_y', 'centroid_z']):
                kf = self.kalman_filters[key]
                kf.predict()
                kf.update([current_centroid[i]])
                filtered_centroid[i] = kf.x[0]

            # Apply Kalman filtering to bounding box
            bbox_kf = self.kalman_filters['bbox']
            bbox_kf.predict()
            current_bbox_flat = current_bbox.flatten()
            bbox_kf.update(current_bbox_flat)
            filtered_bbox_flat = bbox_kf.x
            filtered_bbox = filtered_bbox_flat.reshape(current_bbox.shape)

            # Apply corrections to points
            centroid_correction = filtered_centroid - current_centroid
            corrected_points = points + centroid_correction

            # Apply bounding box constraints
            corrected_points = self._apply_bbox_constraints(corrected_points, filtered_bbox)

            return corrected_points

        except Exception as e:
            self.logger.error(f"Error in temporal Kalman filtering: {e}")
            return points

    def _initialize_kalman_filter(self, filter_key: str):
        """Initialize Kalman filter for specific measurement type."""
        try:
            if filter_key.startswith('centroid'):
                # 1D Kalman filter for centroid coordinates
                kf = KalmanFilter(dim_x=1, dim_z=1)
                kf.F = np.array([[1.0]])  # State transition
                kf.H = np.array([[1.0]])  # Measurement matrix
                kf.Q = Q_discrete_white_noise(dim=1, dt=1.0, var=self.pc_config['kalman_process_noise'])
                kf.R = np.array([[self.pc_config['kalman_measurement_noise']]])
                kf.P = np.eye(1)
                kf.x = np.array([0.0])  # Initial state

            elif filter_key == 'bbox':
                # 6D Kalman filter for bounding box (min_x, min_y, min_z, max_x, max_y, max_z)
                kf = KalmanFilter(dim_x=6, dim_z=6)
                kf.F = np.eye(6)  # State transition
                kf.H = np.eye(6)  # Measurement matrix
                kf.Q = Q_discrete_white_noise(dim=6, dt=1.0, var=self.pc_config['kalman_process_noise'])
                kf.R = np.eye(6) * self.pc_config['kalman_measurement_noise']
                kf.P = np.eye(6)
                kf.x = np.zeros(6)  # Initial state

            self.kalman_filters[filter_key] = kf

        except Exception as e:
            self.logger.error(f"Error initializing Kalman filter {filter_key}: {e}")

    def _calculate_bounding_box(self, points: np.ndarray) -> np.ndarray:
        """Calculate axis-aligned bounding box of point cloud."""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        return np.array([min_coords, max_coords])

    def _apply_bbox_constraints(self, points: np.ndarray, target_bbox: np.ndarray) -> np.ndarray:
        """Apply bounding box constraints to prevent drift."""
        try:
            # Simple constraint application - clamp points to reasonable bounds
            min_bounds = target_bbox[0] - 0.1  # Small tolerance
            max_bounds = target_bbox[1] + 0.1

            constrained_points = np.clip(points, min_bounds, max_bounds)

            return constrained_points

        except Exception as e:
            self.logger.error(f"Error applying bbox constraints: {e}")
            return points

    def estimate_adaptive_normals(self, points: np.ndarray, search_radius: float = None) -> np.ndarray:
        """
        Estimate surface normals with adaptive search radius.

        Args:
            points: Input point cloud
            search_radius: Search radius (adaptive if None)

        Returns:
            Surface normals
        """
        try:
            # Calculate adaptive search radius based on point density
            if search_radius is None:
                if len(points) < 1000:
                    search_radius = 0.05
                elif len(points) > 10000:
                    search_radius = 0.2
                else:
                    search_radius = 0.1

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Estimate normals
            pcd.estimate_normals(search_param=o3d.geometry.KdTreeSearchParamRadius(radius=search_radius))

            return np.asarray(pcd.normals)

        except Exception as e:
            self.logger.error(f"Error estimating adaptive normals: {e}")
            return np.zeros((len(points), 3))

    def apply_bilateral_filter(self, points: np.ndarray, sigma_s: float = 0.03, sigma_r: float = 0.05) -> np.ndarray:
        """
        Apply bilateral filtering to preserve edges while smoothing.

        Args:
            points: Input point cloud
            sigma_s: Spatial sigma
            sigma_r: Range sigma

        Returns:
            Filtered point cloud
        """
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Apply bilateral filter
            filtered_pcd = pcd.bilateral_filter(sigma_s=sigma_s, sigma_r=sigma_r)

            return np.asarray(filtered_pcd.points)

        except Exception as e:
            self.logger.error(f"Error in bilateral filtering: {e}")
            return points

    def compute_curvature(self, points: np.ndarray, neighbors: int = 16) -> np.ndarray:
        """
        Compute curvature values for each point.

        Args:
            points: Input point cloud
            neighbors: Number of neighbors for curvature estimation

        Returns:
            Curvature values
        """
        try:
            # Create KDTree for neighbor search
            tree = KDTree(points)

            curvatures = np.zeros(len(points))

            for i, point in enumerate(points):
                # Find nearest neighbors
                distances, indices = tree.query(point, k=neighbors + 1)

                if len(indices) < 4:  # Need at least 4 points for plane fitting
                    curvatures[i] = 0.0
                    continue

                neighbor_points = points[indices[1:]]  # Exclude the point itself

                # Fit plane to neighbors
                centroid = np.mean(neighbor_points, axis=0)
                centered_points = neighbor_points - centroid

                # SVD for plane normal
                _, _, Vt = np.linalg.svd(centered_points)
                normal = Vt[-1, :]

                # Project points onto plane
                projections = neighbor_points - np.dot(neighbor_points - centroid, normal)[:, np.newaxis] * normal

                # Compute curvature as variance of distances to plane
                distances_to_plane = np.linalg.norm(neighbor_points - projections, axis=1)
                curvatures[i] = np.var(distances_to_plane)

            return curvatures

        except Exception as e:
            self.logger.error(f"Error computing curvature: {e}")
            return np.zeros(len(points))

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not self.processing_times:
            return {
                'avg_processing_time': 0.0,
                'total_processed': 0,
                'processing_history_length': len(self.processing_history)
            }

        return {
            'avg_processing_time': sum(self.processing_times) / len(self.processing_times),
            'total_processed': len(self.processing_times),
            'processing_history_length': len(self.processing_history)
        }

    def clear_history(self):
        """Clear processing history."""
        self.processing_history.clear()
        self.processing_times.clear()
        self.kalman_filters.clear()
        self.filter_states.clear()
        self.logger.info("Processing history cleared")

    def reset_kalman_filters(self):
        """Reset all Kalman filters."""
        self.kalman_filters.clear()
        self.filter_states.clear()
        self.logger.info("Kalman filters reset")