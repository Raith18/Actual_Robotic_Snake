"""
3D Point Cloud Generation Module
Converts depth maps to 3D point clouds with advanced processing capabilities.
"""

import numpy as np
import cv2
import open3d as o3d
import time
import logging
from typing import Optional, Tuple, Dict, Any, List
from scipy.spatial.transform import Rotation
import yaml

class PointCloudGenerator:
    """
    Generates and processes 3D point clouds from depth maps.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        Initialize point cloud generator.

        Args:
            config_path: Path to system configuration file
        """
        self.config = self._load_config(config_path)
        self.pc_config = self.config['point_cloud']

        # Camera intrinsic parameters
        self.fx = 500.0  # Focal length x
        self.fy = 500.0  # Focal length y
        self.cx = 320.0  # Principal point x
        self.cy = 240.0  # Principal point y

        # Point cloud storage
        self.current_points: Optional[np.ndarray] = None
        self.current_colors: Optional[np.ndarray] = None
        self.point_cloud_history: List[Tuple[np.ndarray, np.ndarray]] = []

        # Performance tracking
        self.generation_times = []
        self.max_history = 50

        # Setup logging
        self.logger = logging.getLogger('PointCloudGenerator')

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

    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """
        Set camera intrinsic parameters.

        Args:
            fx: Focal length x
            fy: Focal length y
            cx: Principal point x
            cy: Principal point y
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def generate_point_cloud(self,
                           depth_map: np.ndarray,
                           rgb_image: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate 3D point cloud from depth map and RGB image.

        Args:
            depth_map: Depth map (H, W)
            rgb_image: RGB image (H, W, 3)
            mask: Optional mask for filtering points

        Returns:
            Tuple of (points, colors) arrays
        """
        if depth_map is None or rgb_image is None:
            return None, None

        start_time = time.time()

        try:
            height, width = depth_map.shape[:2]

            # Create meshgrid of pixel coordinates
            u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

            # Calculate 3D coordinates
            x = (u_coords - self.cx) * depth_map / self.fx
            y = (v_coords - self.cy) * depth_map / self.fy
            z = depth_map

            # Stack to create point cloud
            points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

            # Get colors from RGB image
            colors = rgb_image.reshape(-1, 3) / 255.0

            # Apply mask if provided
            if mask is not None:
                valid_mask = mask.ravel() & (z.ravel() > 0) & (z.ravel() < 10.0)
            else:
                valid_mask = (z.ravel() > 0) & (z.ravel() < 10.0)

            # Filter valid points
            points = points[valid_mask]
            colors = colors[valid_mask]

            # Store current point cloud
            self.current_points = points
            self.current_colors = colors

            # Add to history
            self.point_cloud_history.append((points.copy(), colors.copy()))
            if len(self.point_cloud_history) > self.max_history:
                self.point_cloud_history.pop(0)

            generation_time = time.time() - start_time
            self.generation_times.append(generation_time)

            self.logger.debug(f"Generated point cloud: {len(points)} points in {generation_time:.3f}s")

            return points, colors

        except Exception as e:
            self.logger.error(f"Error generating point cloud: {e}")
            return None, None

    def apply_voxel_grid_filter(self,
                              points: np.ndarray,
                              voxel_size: float = None) -> np.ndarray:
        """
        Apply voxel grid downsampling to point cloud.

        Args:
            points: Input point cloud
            voxel_size: Size of voxels (uses config default if None)

        Returns:
            Downsampled point cloud
        """
        if voxel_size is None:
            voxel_size = self.pc_config['voxel_size']

        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Apply voxel grid filter
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

            return np.asarray(downsampled_pcd.points)

        except Exception as e:
            self.logger.error(f"Error in voxel grid filtering: {e}")
            return points

    def remove_outliers(self,
                       points: np.ndarray,
                       neighbors: int = None,
                       std_ratio: float = None) -> np.ndarray:
        """
        Remove statistical outliers from point cloud.

        Args:
            points: Input point cloud
            neighbors: Number of neighbors to consider
            std_ratio: Standard deviation ratio threshold

        Returns:
            Filtered point cloud
        """
        if neighbors is None:
            neighbors = self.pc_config['outlier_neighbors']
        if std_ratio is None:
            std_ratio = self.pc_config['outlier_std_ratio']

        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Remove outliers
            filtered_pcd, inlier_mask = pcd.remove_statistical_outlier(
                nb_neighbors=neighbors,
                std_ratio=std_ratio
            )

            return np.asarray(filtered_pcd.points)

        except Exception as e:
            self.logger.error(f"Error removing outliers: {e}")
            return points

    def apply_kalman_filter(self,
                           current_points: np.ndarray,
                           previous_points: np.ndarray,
                           process_noise: float = None,
                           measurement_noise: float = None) -> np.ndarray:
        """
        Apply Kalman filtering to smooth point cloud motion.

        Args:
            current_points: Current point cloud
            previous_points: Previous point cloud for motion estimation
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance

        Returns:
            Filtered point cloud
        """
        if process_noise is None:
            process_noise = self.pc_config['kalman_process_noise']
        if measurement_noise is None:
            measurement_noise = self.pc_config['kalman_measurement_noise']

        try:
            from filterpy.kalman import KalmanFilter

            if len(current_points) == 0 or len(previous_points) == 0:
                return current_points

            # Simple Kalman filter for point cloud centroid
            kf = KalmanFilter(dim_x=3, dim_z=3)

            # State transition matrix
            kf.F = np.eye(3)

            # Measurement matrix
            kf.H = np.eye(3)

            # Covariances
            kf.Q = np.eye(3) * process_noise
            kf.R = np.eye(3) * measurement_noise
            kf.P = np.eye(3)

            # Calculate centroids
            current_centroid = np.mean(current_points, axis=0)
            previous_centroid = np.mean(previous_points, axis=0)

            # Initialize with previous state
            kf.x = previous_centroid

            # Predict and update
            kf.predict()
            kf.update(current_centroid)

            # Apply correction to all points
            correction = kf.x - current_centroid
            filtered_points = current_points + correction

            return filtered_points

        except Exception as e:
            self.logger.error(f"Error in Kalman filtering: {e}")
            return current_points

    def estimate_normals(self, points: np.ndarray, radius: float = 0.1) -> np.ndarray:
        """
        Estimate surface normals for point cloud.

        Args:
            points: Input point cloud
            radius: Search radius for normal estimation

        Returns:
            Surface normals
        """
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Estimate normals
            pcd.estimate_normals(search_param=o3d.geometry.KdTreeSearchParamRadius(radius=radius))

            return np.asarray(pcd.normals)

        except Exception as e:
            self.logger.error(f"Error estimating normals: {e}")
            return np.zeros((len(points), 3))

    def compute_point_cloud_stats(self, points: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics for point cloud.

        Args:
            points: Input point cloud

        Returns:
            Dictionary with statistics
        """
        if len(points) == 0:
            return {
                'num_points': 0,
                'mean_x': 0.0, 'mean_y': 0.0, 'mean_z': 0.0,
                'std_x': 0.0, 'std_y': 0.0, 'std_z': 0.0,
                'min_x': 0.0, 'max_x': 0.0,
                'min_y': 0.0, 'max_y': 0.0,
                'min_z': 0.0, 'max_z': 0.0
            }

        return {
            'num_points': len(points),
            'mean_x': float(np.mean(points[:, 0])),
            'mean_y': float(np.mean(points[:, 1])),
            'mean_z': float(np.mean(points[:, 2])),
            'std_x': float(np.std(points[:, 0])),
            'std_y': float(np.std(points[:, 1])),
            'std_z': float(np.std(points[:, 2])),
            'min_x': float(np.min(points[:, 0])),
            'max_x': float(np.max(points[:, 0])),
            'min_y': float(np.min(points[:, 1])),
            'max_y': float(np.max(points[:, 1])),
            'min_z': float(np.min(points[:, 2])),
            'max_z': float(np.max(points[:, 2]))
        }

    def transform_point_cloud(self,
                            points: np.ndarray,
                            transformation_matrix: np.ndarray) -> np.ndarray:
        """
        Transform point cloud using 4x4 transformation matrix.

        Args:
            points: Input point cloud
            transformation_matrix: 4x4 transformation matrix

        Returns:
            Transformed point cloud
        """
        try:
            # Add homogeneous coordinates
            points_homo = np.ones((len(points), 4))
            points_homo[:, :3] = points

            # Apply transformation
            transformed_homo = (transformation_matrix @ points_homo.T).T

            return transformed_homo[:, :3]

        except Exception as e:
            self.logger.error(f"Error transforming point cloud: {e}")
            return points

    def merge_point_clouds(self,
                          point_clouds: List[Tuple[np.ndarray, np.ndarray]],
                          max_points: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge multiple point clouds into one.

        Args:
            point_clouds: List of (points, colors) tuples
            max_points: Maximum number of points to keep

        Returns:
            Tuple of merged (points, colors)
        """
        try:
            all_points = []
            all_colors = []

            for points, colors in point_clouds:
                all_points.append(points)
                all_colors.append(colors)

            if not all_points:
                return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

            # Concatenate all point clouds
            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors)

            # Limit number of points
            if len(merged_points) > max_points:
                indices = np.random.choice(len(merged_points), max_points, replace=False)
                merged_points = merged_points[indices]
                merged_colors = merged_colors[indices]

            return merged_points, merged_colors

        except Exception as e:
            self.logger.error(f"Error merging point clouds: {e}")
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    def save_point_cloud(self, points: np.ndarray, colors: np.ndarray, filename: str):
        """
        Save point cloud to file.

        Args:
            points: 3D points
            colors: Point colors
            filename: Output filename
        """
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Save to file
            o3d.io.write_point_cloud(filename, pcd)
            self.logger.info(f"Point cloud saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error saving point cloud: {e}")

    def load_point_cloud(self, filename: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load point cloud from file.

        Args:
            filename: Input filename

        Returns:
            Tuple of (points, colors) or (None, None) if failed
        """
        try:
            # Load point cloud
            pcd = o3d.io.read_point_cloud(filename)

            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            return points, colors

        except Exception as e:
            self.logger.error(f"Error loading point cloud: {e}")
            return None, None

    def get_generation_stats(self) -> Dict[str, float]:
        """Get point cloud generation statistics."""
        if not self.generation_times:
            return {'avg_generation_time': 0.0, 'num_generations': 0}

        return {
            'avg_generation_time': sum(self.generation_times) / len(self.generation_times),
            'num_generations': len(self.generation_times),
            'current_points': len(self.current_points) if self.current_points is not None else 0
        }

    def clear_history(self):
        """Clear point cloud history."""
        self.point_cloud_history.clear()
        self.generation_times.clear()
        self.logger.info("Point cloud history cleared")