"""
Edge Detection and Error Analysis Module
Implements edge detection, error-point marking, and distance calculations for depth maps.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, Any, List
from scipy import ndimage
from skimage import filters, feature
import yaml

class ErrorAnalyzer:
    """
    Analyzes depth maps for edges, errors, and provides distance calculations.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        Initialize error analyzer.

        Args:
            config_path: Path to system configuration file
        """
        self.config = self._load_config(config_path)
        self.error_config = self.config['error_analysis']

        # Edge detection parameters
        self.edge_threshold = self.error_config['edge_detection_threshold']
        self.enable_edge_detection = self.error_config['enable_edge_detection']

        # Error analysis parameters
        self.enable_error_marking = self.error_config['enable_error_marking']
        self.error_threshold = self.error_config['error_threshold']
        self.max_error_points = self.error_config['max_error_points']

        # Analysis results storage
        self.current_edges: Optional[np.ndarray] = None
        self.current_error_points: Optional[np.ndarray] = None
        self.current_distances: Optional[Dict[str, float]] = None

        # Performance tracking
        self.analysis_times = []
        self.max_history = 100

        # Setup logging
        self.logger = logging.getLogger('ErrorAnalyzer')

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {
                'error_analysis': {
                    'enable_edge_detection': True,
                    'edge_detection_threshold': 0.1,
                    'enable_error_marking': True,
                    'error_threshold': 0.05,
                    'max_error_points': 1000
                }
            }

    def detect_edges(self, depth_map: np.ndarray, rgb_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect edges in depth map using multiple methods.

        Args:
            depth_map: Input depth map
            rgb_image: Optional RGB image for combined analysis

        Returns:
            Edge map
        """
        if not self.enable_edge_detection or depth_map is None:
            return np.zeros_like(depth_map)

        start_time = time.time()

        try:
            # Method 1: Gradient-based edge detection
            edges_gradient = self._gradient_edge_detection(depth_map)

            # Method 2: Canny edge detection
            edges_canny = self._canny_edge_detection(depth_map)

            # Method 3: Morphological edge detection
            edges_morph = self._morphological_edge_detection(depth_map)

            # Combine edge detection methods
            combined_edges = self._combine_edge_methods([
                edges_gradient,
                edges_canny,
                edges_morph
            ])

            # Apply threshold
            combined_edges = (combined_edges > self.edge_threshold).astype(np.uint8) * 255

            # Store results
            self.current_edges = combined_edges

            analysis_time = time.time() - start_time
            self.analysis_times.append(analysis_time)

            if len(self.analysis_times) > self.max_history:
                self.analysis_times.pop(0)

            self.logger.debug(f"Edge detection completed in {analysis_time".3f"}s")

            return combined_edges

        except Exception as e:
            self.logger.error(f"Error in edge detection: {e}")
            return np.zeros_like(depth_map)

    def _gradient_edge_detection(self, depth_map: np.ndarray) -> np.ndarray:
        """Gradient-based edge detection using Sobel operators."""
        try:
            # Apply Gaussian blur to reduce noise
            smoothed = cv2.GaussianBlur(depth_map, (3, 3), 0)

            # Calculate gradients
            grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate magnitude
            magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Normalize
            if magnitude.max() > 0:
                magnitude = magnitude / magnitude.max()

            return magnitude

        except Exception as e:
            self.logger.error(f"Error in gradient edge detection: {e}")
            return np.zeros_like(depth_map)

    def _canny_edge_detection(self, depth_map: np.ndarray) -> np.ndarray:
        """Canny edge detection optimized for depth maps."""
        try:
            # Normalize depth map to [0, 255]
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_8u = depth_norm.astype(np.uint8)

            # Apply Canny edge detection
            edges = cv2.Canny(depth_8u, 50, 150)

            # Convert back to float
            return edges.astype(np.float32) / 255.0

        except Exception as e:
            self.logger.error(f"Error in Canny edge detection: {e}")
            return np.zeros_like(depth_map)

    def _morphological_edge_detection(self, depth_map: np.ndarray) -> np.ndarray:
        """Morphological edge detection using erosion and dilation."""
        try:
            # Create binary mask from depth map
            threshold = np.mean(depth_map) + np.std(depth_map)
            binary = (depth_map > threshold).astype(np.uint8)

            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=1)
            dilated = cv2.dilate(binary, kernel, iterations=1)

            # Edge map is difference between dilation and erosion
            edges = dilated - eroded

            return edges.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Error in morphological edge detection: {e}")
            return np.zeros_like(depth_map)

    def _combine_edge_methods(self, edge_maps: List[np.ndarray]) -> np.ndarray:
        """Combine multiple edge detection methods."""
        try:
            # Weighted combination of edge maps
            weights = [0.4, 0.4, 0.2]  # Weights for gradient, canny, morphological
            combined = np.zeros_like(edge_maps[0])

            for edge_map, weight in zip(edge_maps, weights):
                combined += edge_map * weight

            return combined

        except Exception as e:
            self.logger.error(f"Error combining edge methods: {e}")
            return edge_maps[0] if edge_maps else np.array([])

    def mark_error_points(self,
                         depth_map: np.ndarray,
                         reference_depth: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Mark points with depth estimation errors.

        Args:
            depth_map: Current depth map
            reference_depth: Reference depth map for comparison

        Returns:
            Tuple of (error_map, error_points_list)
        """
        if not self.enable_error_marking or depth_map is None:
            return np.zeros_like(depth_map), []

        try:
            error_map = np.zeros_like(depth_map)
            error_points = []

            # Method 1: Depth discontinuity detection
            discontinuity_errors = self._detect_discontinuity_errors(depth_map)
            error_map += discontinuity_errors

            # Method 2: Reference-based error detection
            if reference_depth is not None:
                reference_errors = self._detect_reference_errors(depth_map, reference_depth)
                error_map += reference_errors

            # Method 3: Statistical outlier detection
            statistical_errors = self._detect_statistical_errors(depth_map)
            error_map += statistical_errors

            # Normalize error map
            if error_map.max() > 0:
                error_map = error_map / error_map.max()

            # Extract error points
            error_mask = error_map > self.error_threshold
            error_locations = np.where(error_mask)

            for y, x in zip(error_locations[0], error_locations[1]):
                if len(error_points) < self.max_error_points:
                    error_point = {
                        'x': int(x),
                        'y': int(y),
                        'depth': float(depth_map[y, x]),
                        'error_score': float(error_map[y, x]),
                        'type': 'depth_error'
                    }
                    error_points.append(error_point)

            self.current_error_points = error_points

            return error_map, error_points

        except Exception as e:
            self.logger.error(f"Error marking error points: {e}")
            return np.zeros_like(depth_map), []

    def _detect_discontinuity_errors(self, depth_map: np.ndarray) -> np.ndarray:
        """Detect errors at depth discontinuities."""
        try:
            # Calculate Laplacian for edge detection
            laplacian = cv2.Laplacian(depth_map, cv2.CV_64F)

            # Find areas with high Laplacian values (potential errors)
            error_map = np.abs(laplacian)
            error_map = cv2.GaussianBlur(error_map, (5, 5), 0)

            # Normalize
            if error_map.max() > 0:
                error_map = error_map / error_map.max()

            return error_map

        except Exception as e:
            self.logger.error(f"Error detecting discontinuity errors: {e}")
            return np.zeros_like(depth_map)

    def _detect_reference_errors(self, depth_map: np.ndarray, reference_depth: np.ndarray) -> np.ndarray:
        """Detect errors by comparing to reference depth."""
        try:
            # Calculate absolute difference
            diff = np.abs(depth_map - reference_depth)

            # Apply threshold
            error_map = (diff > self.error_threshold).astype(np.float32)

            # Apply Gaussian blur to smooth
            error_map = cv2.GaussianBlur(error_map, (3, 3), 0)

            return error_map

        except Exception as e:
            self.logger.error(f"Error detecting reference errors: {e}")
            return np.zeros_like(depth_map)

    def _detect_statistical_errors(self, depth_map: np.ndarray) -> np.ndarray:
        """Detect statistical outliers in depth values."""
        try:
            # Calculate local statistics
            kernel_size = 5
            mean_depth = cv2.blur(depth_map, (kernel_size, kernel_size))
            sq_diff = (depth_map - mean_depth) ** 2
            var_depth = cv2.blur(sq_diff, (kernel_size, kernel_size))

            # Detect outliers using z-score
            std_depth = np.sqrt(var_depth)
            z_score = np.abs(depth_map - mean_depth) / (std_depth + 1e-6)

            # Mark high z-score areas as potential errors
            error_map = (z_score > 2.0).astype(np.float32)

            return error_map

        except Exception as e:
            self.logger.error(f"Error detecting statistical errors: {e}")
            return np.zeros_like(depth_map)

    def calculate_distances(self,
                          depth_map: np.ndarray,
                          points: Optional[List[Tuple[int, int]]] = None) -> Dict[str, float]:
        """
        Calculate various distance measurements.

        Args:
            depth_map: Input depth map
            points: Optional list of points to measure distances between

        Returns:
            Dictionary with distance measurements
        """
        try:
            distances = {}

            # Calculate depth statistics
            valid_depths = depth_map[depth_map > 0]
            if len(valid_depths) > 0:
                distances.update({
                    'min_depth': float(np.min(valid_depths)),
                    'max_depth': float(np.max(valid_depths)),
                    'mean_depth': float(np.mean(valid_depths)),
                    'median_depth': float(np.median(valid_depths)),
                    'depth_range': float(np.max(valid_depths) - np.min(valid_depths))
                })

            # Calculate distances between specified points
            if points and len(points) >= 2:
                point_distances = []
                for i in range(len(points) - 1):
                    for j in range(i + 1, len(points)):
                        p1, p2 = points[i], points[j]
                        if 0 <= p1[0] < depth_map.shape[1] and 0 <= p1[1] < depth_map.shape[0] and \
                           0 <= p2[0] < depth_map.shape[1] and 0 <= p2[1] < depth_map.shape[0]:

                            depth1 = depth_map[p1[1], p1[0]]
                            depth2 = depth_map[p2[1], p2[0]]

                            if depth1 > 0 and depth2 > 0:
                                # 3D distance calculation
                                fx, fy, cx, cy = 500.0, 500.0, depth_map.shape[1]/2, depth_map.shape[0]/2
                                x1 = (p1[0] - cx) * depth1 / fx
                                y1 = (p1[1] - cy) * depth1 / fy
                                x2 = (p2[0] - cx) * depth2 / fx
                                y2 = (p2[1] - cy) * depth2 / fy

                                distance_3d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (depth2 - depth1)**2)
                                point_distances.append(distance_3d)

                if point_distances:
                    distances.update({
                        'point_distances': point_distances,
                        'avg_point_distance': float(np.mean(point_distances)),
                        'min_point_distance': float(np.min(point_distances)),
                        'max_point_distance': float(np.max(point_distances))
                    })

            self.current_distances = distances
            return distances

        except Exception as e:
            self.logger.error(f"Error calculating distances: {e}")
            return {}

    def create_analytical_overlay(self,
                                rgb_image: np.ndarray,
                                depth_map: np.ndarray,
                                edges: Optional[np.ndarray] = None,
                                error_points: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """
        Create analytical overlay showing edges, errors, and measurements.

        Args:
            rgb_image: Input RGB image
            depth_map: Depth map
            edges: Edge map
            error_points: List of error points

        Returns:
            Image with analytical overlay
        """
        try:
            # Create overlay image
            overlay = rgb_image.copy()

            # Draw edges if available
            if edges is not None and self.current_edges is not None:
                edge_overlay = np.zeros_like(overlay)
                edge_mask = self.current_edges > 0.5

                # Color edges in red
                edge_overlay[edge_mask] = [0, 0, 255]
                overlay = cv2.addWeighted(overlay, 0.7, edge_overlay, 0.3, 0)

            # Draw error points if available
            if error_points:
                for error_point in error_points[:self.max_error_points]:
                    x, y = error_point['x'], error_point['y']
                    if 0 <= y < overlay.shape[0] and 0 <= x < overlay.shape[1]:
                        # Draw error point as red circle
                        cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
                        cv2.putText(overlay, f"E:{error_point['error_score']".2f"}",
                                  (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Draw distance measurements if available
            if self.current_distances:
                self._draw_distance_annotations(overlay, depth_map)

            return overlay

        except Exception as e:
            self.logger.error(f"Error creating analytical overlay: {e}")
            return rgb_image

    def _draw_distance_annotations(self, image: np.ndarray, depth_map: np.ndarray):
        """Draw distance measurement annotations on image."""
        try:
            # Draw depth range information
            h, w = image.shape[:2]

            # Background rectangle for text
            cv2.rectangle(image, (10, 10), (200, 100), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (200, 100), (255, 255, 255), 2)

            # Draw depth statistics
            y_offset = 25
            for key, value in self.current_distances.items():
                if isinstance(value, float):
                    text = f"{key.replace('_', ' ').title()}: {value".3f"}m"
                    cv2.putText(image, text, (15, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 15

        except Exception as e:
            self.logger.error(f"Error drawing distance annotations: {e}")

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return {
            'avg_analysis_time': sum(self.analysis_times) / len(self.analysis_times) if self.analysis_times else 0.0,
            'num_analyses': len(self.analysis_times),
            'num_error_points': len(self.current_error_points) if self.current_error_points else 0,
            'edge_detection_enabled': self.enable_edge_detection,
            'error_marking_enabled': self.enable_error_marking
        }

    def clear_analysis_data(self):
        """Clear current analysis data."""
        self.current_edges = None
        self.current_error_points = None
        self.current_distances = None
        self.analysis_times.clear()
        self.logger.info("Analysis data cleared")

    def save_analysis_results(self, output_dir: str = "output"):
        """Save analysis results to files."""
        try:
            import os

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Save edge map
            if self.current_edges is not None:
                edge_filename = f"{output_dir}/edges_{timestamp}.png"
                cv2.imwrite(edge_filename, (self.current_edges * 255).astype(np.uint8))

            # Save error points
            if self.current_error_points:
                error_filename = f"{output_dir}/error_points_{timestamp}.txt"
                with open(error_filename, 'w') as f:
                    for point in self.current_error_points:
                        f.write(f"{point['x']},{point['y']},{point['depth']".3f"},{point['error_score']".3f"}\n")

            # Save distances
            if self.current_distances:
                distances_filename = f"{output_dir}/distances_{timestamp}.txt"
                with open(distances_filename, 'w') as f:
                    for key, value in self.current_distances.items():
                        if isinstance(value, list):
                            f.write(f"{key}: {','.join([f'{v".3f"}' for v in value])}\n")
                        else:
                            f.write(f"{key}: {value".3f"}\n")

            self.logger.info(f"Analysis results saved to {output_dir}")

        except Exception as e:
            self.logger.error(f"Error saving analysis results: {e}")