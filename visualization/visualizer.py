"""
Dual Synchronized Visualization System
Provides 3D point cloud viewer and analytical 2D view with real-time updates.
"""

import cv2
import numpy as np
import open3d as o3d
import time
import threading
import logging
from typing import Optional, Tuple, Dict, Any, List
import yaml

class DualVisualizer:
    """
    Dual synchronized visualization system with 3D and 2D views.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        Initialize dual visualizer.

        Args:
            config_path: Path to system configuration file
        """
        self.config = self._load_config(config_path)
        self.vis_config = self.config['visualization']

        # Visualization windows
        self.point_cloud_window = self.vis_config['point_cloud_window']
        self.analytical_window = self.vis_config['analytical_window']

        # Window dimensions
        self.pc_width = self.vis_config['point_cloud_width']
        self.pc_height = self.vis_config['point_cloud_height']
        self.analytical_width = self.vis_config['analytical_width']
        self.analytical_height = self.vis_config['analytical_height']

        # 3D visualization components
        self.vis3d: Optional[o3d.visualization.Visualizer] = None
        self.point_cloud: Optional[o3d.geometry.PointCloud] = None
        self.coordinate_frame: Optional[o3d.geometry.TriangleMesh] = None

        # 2D visualization components
        self.analytical_image: Optional[np.ndarray] = None

        # Synchronization
        self.running = False
        self.lock = threading.Lock()

        # Performance tracking
        self.frame_times = []
        self.render_times = []

        # Setup logging
        self.logger = logging.getLogger('DualVisualizer')

        # Initialize visualizations
        self._initialize_3d_visualizer()
        self._initialize_2d_visualizer()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {
                'visualization': {
                    'point_cloud_window': '3D Point Cloud Viewer',
                    'point_cloud_width': 800,
                    'point_cloud_height': 600,
                    'analytical_window': 'Analytical View',
                    'analytical_width': 1200,
                    'analytical_height': 800,
                    'point_size': 2.0,
                    'background_color': [0.1, 0.1, 0.1],
                    'show_axes': True,
                    'show_grid': True
                }
            }

    def _initialize_3d_visualizer(self):
        """Initialize 3D point cloud visualizer."""
        try:
            # Create visualizer
            self.vis3d = o3d.visualization.Visualizer()
            self.vis3d.create_window(
                window_name=self.point_cloud_window,
                width=self.pc_width,
                height=self.pc_height
            )

            # Set background color
            opt = self.vis3d.get_render_option()
            bg_color = self.vis_config['background_color']
            opt.background_color = np.asarray(bg_color)

            # Set point size
            opt.point_size = self.vis_config['point_size']

            # Create coordinate frame
            if self.vis_config['show_axes']:
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                self.vis3d.add_geometry(self.coordinate_frame)

            # Create initial point cloud
            self.point_cloud = o3d.geometry.PointCloud()
            self.vis3d.add_geometry(self.point_cloud)

            # Set camera view
            ctr = self.vis3d.get_view_control()
            ctr.set_zoom(0.8)

            self.logger.info("3D visualizer initialized")

        except Exception as e:
            self.logger.error(f"Error initializing 3D visualizer: {e}")
            self.vis3d = None

    def _initialize_2d_visualizer(self):
        """Initialize 2D analytical visualizer."""
        try:
            # Create initial analytical image
            self.analytical_image = np.zeros(
                (self.analytical_height, self.analytical_width, 3),
                dtype=np.uint8
            )

            # Create window
            cv2.namedWindow(self.analytical_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.analytical_window, self.analytical_width, self.analytical_height)

            self.logger.info("2D visualizer initialized")

        except Exception as e:
            self.logger.error(f"Error initializing 2D visualizer: {e}")

    def start_visualization(self):
        """Start the visualization system."""
        self.running = True
        self.logger.info("Visualization system started")

    def stop_visualization(self):
        """Stop the visualization system."""
        self.running = False

        if self.vis3d:
            self.vis3d.destroy_window()

        cv2.destroyAllWindows()

        self.logger.info("Visualization system stopped")

    def update_3d_point_cloud(self,
                            points: np.ndarray,
                            colors: Optional[np.ndarray] = None,
                            object_detections: Optional[List[Dict[str, Any]]] = None):
        """
        Update 3D point cloud visualization.

        Args:
            points: 3D points
            colors: Point colors
            object_detections: Object detection results
        """
        if not self.running or self.vis3d is None or points is None:
            return

        try:
            with self.lock:
                # Update point cloud geometry
                self.point_cloud.points = o3d.utility.Vector3dVector(points)

                if colors is not None:
                    self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
                else:
                    # Default color if none provided
                    default_colors = np.ones((len(points), 3)) * 0.7
                    self.point_cloud.colors = o3d.utility.Vector3dVector(default_colors)

                # Add object bounding boxes if available
                if object_detections:
                    self._add_object_visualizations(object_detections)

                # Update visualizer
                self.vis3d.update_geometry(self.point_cloud)
                self.vis3d.poll_events()
                self.vis3d.update_renderer()

        except Exception as e:
            self.logger.error(f"Error updating 3D point cloud: {e}")

    def _add_object_visualizations(self, object_detections: List[Dict[str, Any]]):
        """Add object detection visualizations to 3D scene."""
        try:
            for detection in object_detections:
                if 'position_3d' in detection and 'point_cloud' in detection:
                    obj_points = detection['point_cloud']
                    if len(obj_points) > 0:
                        # Create object point cloud
                        obj_pcd = o3d.geometry.PointCloud()
                        obj_pcd.points = o3d.utility.Vector3dVector(obj_points)

                        # Color based on object class
                        class_colors = {
                            'person': [1.0, 0.0, 0.0],  # Red
                            'car': [0.0, 1.0, 0.0],     # Green
                            'chair': [0.0, 0.0, 1.0],   # Blue
                            'bottle': [1.0, 1.0, 0.0],  # Yellow
                        }

                        class_name = detection.get('class_name', 'unknown')
                        color = class_colors.get(class_name, [0.5, 0.5, 0.5])
                        obj_colors = np.tile(color, (len(obj_points), 1))
                        obj_pcd.colors = o3d.utility.Vector3dVector(obj_colors)

                        # Add to visualizer (temporary - in practice, you'd manage these)
                        # self.vis3d.add_geometry(obj_pcd)

        except Exception as e:
            self.logger.error(f"Error adding object visualizations: {e}")

    def update_analytical_view(self,
                             rgb_image: np.ndarray,
                             depth_map: Optional[np.ndarray] = None,
                             edges: Optional[np.ndarray] = None,
                             error_points: Optional[List[Dict[str, Any]]] = None,
                             detections: Optional[List[Dict[str, Any]]] = None,
                             distances: Optional[Dict[str, float]] = None):
        """
        Update 2D analytical view with overlays.

        Args:
            rgb_image: RGB image
            depth_map: Depth map for colorization
            edges: Edge detection results
            error_points: Error point locations
            detections: Object detections
            distances: Distance measurements
        """
        if not self.running:
            return

        try:
            with self.lock:
                # Start with RGB image
                display_image = rgb_image.copy()

                # Add depth map overlay if available
                if depth_map is not None:
                    depth_colormap = cv2.applyColorMap(
                        ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8),
                        cv2.COLORMAP_JET
                    )
                    display_image = cv2.addWeighted(display_image, 0.6, depth_colormap, 0.4, 0)

                # Add edge overlay if available
                if edges is not None:
                    edge_mask = (edges > 0.5).astype(np.uint8) * 255
                    edge_colormap = cv2.cvtColor(edge_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    edge_colormap[:, :, 0] = 0    # Remove blue channel
                    edge_colormap[:, :, 2] = 255  # Add red channel
                    display_image = cv2.addWeighted(display_image, 0.8, edge_colormap, 0.2, 0)

                # Draw error points if available
                if error_points:
                    for error_point in error_points:
                        x, y = error_point['x'], error_point['y']
                        if 0 <= y < display_image.shape[0] and 0 <= x < display_image.shape[1]:
                            cv2.circle(display_image, (x, y), 4, (0, 0, 255), -1)
                            cv2.putText(display_image, f"E:{error_point['error_score']:.2f}",
                                      (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # Draw object detections if available
                if detections:
                    display_image = self._draw_2d_detections(display_image, detections)

                # Add distance measurements if available
                if distances:
                    display_image = self._add_distance_overlay(display_image, distances)

                # Add performance information
                display_image = self._add_performance_overlay(display_image)

                # Update display
                self.analytical_image = display_image
                cv2.imshow(self.analytical_window, display_image)

        except Exception as e:
            self.logger.error(f"Error updating analytical view: {e}")

    def _draw_2d_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw 2D object detections on image."""
        try:
            result = image.copy()

            for detection in detections:
                # Draw bounding box
                bbox = detection['bbox']
                track_id = detection.get('track_id', -1)

                # Color based on track ID
                if track_id >= 0:
                    color = self._get_track_color(track_id)
                else:
                    color = (0, 255, 0)  # Green for untracked

                cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                # Draw label
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                if track_id >= 0:
                    label += f" ID:{track_id}"

                cv2.putText(result, label, (bbox[0], bbox[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw 3D position if available
                if 'position_3d' in detection:
                    pos_3d = detection['position_3d']
                    pos_text = f"X:{pos_3d[0]:.2f} Y:{pos_3d[1]:.2f} Z:{pos_3d[2]:.2f}"
                    cv2.putText(result, pos_text, (bbox[0], bbox[3] + 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            return result

        except Exception as e:
            self.logger.error(f"Error drawing 2D detections: {e}")
            return image

    def _add_distance_overlay(self, image: np.ndarray, distances: Dict[str, float]) -> np.ndarray:
        """Add distance measurement overlay to image."""
        try:
            result = image.copy()

            # Draw distance information in corner
            info_x, info_y = 10, 30
            info_width, info_height = 250, 150

            # Background
            cv2.rectangle(result, (info_x, info_y), (info_x + info_width, info_y + info_height),
                        (0, 0, 0), -1)
            cv2.rectangle(result, (info_x, info_y), (info_x + info_width, info_y + info_height),
                        (255, 255, 255), 2)

            # Distance text
            y_offset = info_y + 20
            for key, value in distances.items():
                if isinstance(value, float):
                    text = f"{key.replace('_', ' ').title()}: {value:.3f}m"
                    cv2.putText(result, text, (info_x + 10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20

            return result

        except Exception as e:
            self.logger.error(f"Error adding distance overlay: {e}")
            return image

    def _add_performance_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add performance information overlay."""
        try:
            result = image.copy()

            # Calculate FPS
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            else:
                fps = 0.0

            # Performance text
            perf_text = [
                f"FPS: {fps:.1f}",
                f"Frame Time: {avg_frame_time*1000:.1f}ms" if self.frame_times else "Frame Time: N/A"
            ]

            # Background
            text_x, text_y = image.shape[1] - 200, 30
            cv2.rectangle(result, (text_x, text_y), (text_x + 180, text_y + 60),
                        (0, 0, 0), -1)
            cv2.rectangle(result, (text_x, text_y), (text_x + 180, text_y + 60),
                        (255, 255, 255), 1)

            # Performance text
            for i, text in enumerate(perf_text):
                cv2.putText(result, text, (text_x + 10, text_y + 15 + i * 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            return result

        except Exception as e:
            self.logger.error(f"Error adding performance overlay: {e}")
            return image

    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID."""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        return colors[track_id % len(colors)]

    def capture_3d_screenshot(self, filename: str):
        """Capture screenshot of 3D view."""
        if self.vis3d:
            self.vis3d.capture_screen_image(filename)

    def capture_2d_screenshot(self, filename: str):
        """Capture screenshot of 2D analytical view."""
        if self.analytical_image is not None:
            cv2.imwrite(filename, self.analytical_image)

    def is_running(self) -> bool:
        """Check if visualization is running."""
        return self.running

    def get_performance_stats(self) -> Dict[str, float]:
        """Get visualization performance statistics."""
        return {
            'avg_frame_time': sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0,
            'avg_render_time': sum(self.render_times) / len(self.render_times) if self.render_times else 0.0,
            'num_frames': len(self.frame_times)
        }

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.frame_times.clear()
        self.render_times.clear()

    def handle_key_events(self):
        """Handle keyboard events for both windows."""
        # Check for key presses in analytical window
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.stop_visualization()
            return False
        elif key == ord('s'):
            # Save screenshots
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.capture_3d_screenshot(f"output/pc_view_{timestamp}.png")
            self.capture_2d_screenshot(f"output/analytical_view_{timestamp}.png")
            self.logger.info("Screenshots saved")
        elif key == ord('r'):
            # Reset view
            if self.vis3d:
                ctr = self.vis3d.get_view_control()
                ctr.reset()

        return True