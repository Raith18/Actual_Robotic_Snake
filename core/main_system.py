"""
Main System Orchestrator
Coordinates all components for the real-time monocular depth SLAM system.
"""

import cv2
import numpy as np
import time
import threading
import logging
import yaml
from typing import Optional, Dict, Any, List
from queue import Queue
import signal
import sys

class DepthSLAMSystem:
    """
    Main orchestrator for the real-time monocular depth SLAM system.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        Initialize the main system.

        Args:
            config_path: Path to system configuration file
        """
        self.config = self._load_config(config_path)
        self.running = False

        # Initialize core components
        self.camera = None
        self.depth_estimator = None
        self.point_cloud_generator = None
        self.point_cloud_processor = None
        self.object_detector = None
        self.error_analyzer = None
        self.visualizer = None
        self.slam_system = None

        # Data queues for thread communication
        self.frame_queue = Queue(maxsize=10)
        self.processed_data_queue = Queue(maxsize=10)

        # Performance monitoring
        self.frame_times = []
        self.processing_times = []
        self.start_time = None

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize components
        self._initialize_components()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/system.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('DepthSLAMSystem')

    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Import here to avoid circular imports
            from .camera_capture import WebcamCapture
            from ..models.depth_estimator import MiDaSDepthEstimator
            from ..models.point_cloud_generator import PointCloudGenerator
            from ..processing.point_cloud_processor import PointCloudProcessor
            from ..models.object_detector import ObjectDetector
            from ..processing.error_analysis import ErrorAnalyzer
            from ..visualization.simple_visualizer import SimpleVisualizer
            from ..slam.slam_system import SLAMSystem

            # Initialize components
            self.camera = WebcamCapture()
            self.depth_estimator = MiDaSDepthEstimator()
            self.point_cloud_generator = PointCloudGenerator()
            self.point_cloud_processor = PointCloudProcessor()
            self.object_detector = ObjectDetector()
            self.error_analyzer = ErrorAnalyzer()
            self.visualizer = DualVisualizer()
            self.slam_system = SLAMSystem()

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise

    def start(self) -> bool:
        """
        Start the complete system.

        Returns:
            True if started successfully
        """
        try:
            # Initialize camera
            if not self.camera.initialize_camera():
                self.logger.error("Failed to initialize camera")
                return False

            # Start camera capture
            if not self.camera.start_capture():
                self.logger.error("Failed to start camera capture")
                return False

            # Start visualizer
            self.visualizer.start_visualization()

            # Set running flag
            self.running = True
            self.start_time = time.time()

            # Start processing threads
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)

            self.capture_thread.start()
            self.processing_thread.start()

            self.logger.info("Depth SLAM system started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            return False

    def stop(self):
        """Stop the complete system."""
        self.running = False

        # Stop camera
        if self.camera:
            self.camera.stop_capture()

        # Stop visualizer
        if self.visualizer:
            self.visualizer.stop_visualization()

        # Wait for threads to finish
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)

        # Calculate final statistics
        self._calculate_final_stats()

        self.logger.info("Depth SLAM system stopped")

    def _capture_loop(self):
        """Main camera capture loop."""
        frame_count = 0
        last_fps_time = time.time()

        while self.running:
            try:
                # Get frame from camera
                frame = self.camera.get_frame(timeout=0.1)

                if frame is not None:
                    frame_count += 1

                    # Put frame in queue for processing
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)

                    # Calculate FPS
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        fps = frame_count
                        frame_count = 0
                        last_fps_time = current_time
                        self.logger.debug(f"Capture FPS: {fps}")

                time.sleep(0.01)  # Small delay to prevent busy waiting

            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                break

    def _processing_loop(self):
        """Main processing loop."""
        frame_count = 0
        processing_start_time = time.time()

        # Initialize SLAM with first frame
        slam_initialized = False

        while self.running:
            try:
                # Get frame from queue
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue

                frame = self.frame_queue.get()

                # Measure processing time
                loop_start_time = time.time()

                # Step 1: Depth estimation
                depth_map, depth_time = self.depth_estimator.estimate_depth(frame)

                # Step 2: Generate point cloud
                if depth_map is not None:
                    points_3d, colors = self.point_cloud_generator.generate_point_cloud(
                        depth_map, frame
                    )

                    # Step 3: Process point cloud
                    if points_3d is not None:
                        processed_points, processed_colors = self.point_cloud_processor.process_point_cloud(
                            points_3d, colors
                        )
                    else:
                        processed_points, processed_colors = None, None
                else:
                    processed_points, processed_colors = None, None

                # Step 4: Object detection and tracking
                detections = self.object_detector.detect_objects(frame)
                tracked_detections = self.object_detector.track_objects(detections, frame)

                # Step 5: Integrate with depth information
                if tracked_detections:
                    tracked_detections = self.object_detector.integrate_with_depth(
                        tracked_detections, depth_map, processed_points
                    )

                # Step 6: Error analysis
                if depth_map is not None:
                    edges = self.error_analyzer.detect_edges(depth_map, frame)
                    error_map, error_points = self.error_analyzer.mark_error_points(depth_map)
                    distances = self.error_analyzer.calculate_distances(depth_map)
                else:
                    edges, error_map, error_points, distances = None, None, None, None

                # Step 7: SLAM tracking
                if not slam_initialized and depth_map is not None:
                    if self.slam_system.initialize(frame, depth_map):
                        slam_initialized = True
                        self.logger.info("SLAM initialized")
                elif slam_initialized:
                    self.slam_system.track_frame(frame, depth_map)

                # Step 8: Update visualizations
                if processed_points is not None:
                    self.visualizer.update_3d_point_cloud(
                        processed_points, processed_colors, tracked_detections
                    )

                self.visualizer.update_analytical_view(
                    frame, depth_map, edges, error_points, tracked_detections, distances
                )

                # Step 9: Handle key events
                if not self.visualizer.handle_key_events():
                    break

                # Performance tracking
                frame_time = time.time() - loop_start_time
                self.frame_times.append(frame_time)
                frame_count += 1

                # Periodic statistics
                if frame_count % 100 == 0:
                    avg_frame_time = sum(self.frame_times[-100:]) / min(100, len(self.frame_times))
                    self.logger.info(f"Average frame time: {avg_frame_time:.3f}s ({1.0/avg_frame_time:.1f} FPS)")

                time.sleep(0.001)  # Small delay

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                break

    def _calculate_final_stats(self):
        """Calculate and log final system statistics."""
        try:
            if self.frame_times:
                total_time = time.time() - self.start_time if self.start_time else 0
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                total_frames = len(self.frame_times)

                self.logger.info("=== Final System Statistics ===")
                self.logger.info(f"Total runtime: {total_time:.2f}s")
                self.logger.info(f"Total frames processed: {total_frames}")
                self.logger.info(f"Average frame time: {avg_frame_time:.3f}s")
                self.logger.info(f"Average FPS: {1.0/avg_frame_time:.1f}")
                self.logger.info(f"Camera FPS: {self.camera.get_fps() if self.camera else 0}")

                # Component statistics
                if self.depth_estimator:
                    self.logger.info(f"Depth estimation avg time: {self.depth_estimator.get_average_inference_time():.3f}s")
                if self.object_detector:
                    stats = self.object_detector.get_detection_stats()
                    self.logger.info(f"Active object tracks: {stats['num_active_tracks']}")
                if self.slam_system:
                    slam_stats = self.slam_system.get_tracking_stats()
                    self.logger.info(f"SLAM keyframes: {slam_stats['num_keyframes']}")
                    self.logger.info(f"Map points: {slam_stats['num_map_points']}")

                self.logger.info("==============================")

        except Exception as e:
            self.logger.error(f"Error calculating final stats: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'running': self.running,
            'camera_fps': self.camera.get_fps() if self.camera else 0,
            'frame_queue_size': self.frame_queue.qsize(),
            'processed_queue_size': self.processed_data_queue.qsize(),
            'components_ready': {
                'camera': self.camera.is_capturing() if self.camera else False,
                'depth_estimator': self.depth_estimator.is_model_loaded() if self.depth_estimator else False,
                'object_detector': self.object_detector.model is not None if self.object_detector else False,
                'slam': self.slam_system.tracking_state if self.slam_system else "NOT_INITIALIZED",
                'visualizer': self.visualizer.is_running() if self.visualizer else False
            }
        }

    def save_system_state(self, filename: str = "system_state.json"):
        """Save current system state."""
        try:
            state = {
                'timestamp': time.time(),
                'config': self.config,
                'status': self.get_system_status(),
                'performance': {
                    'total_frames': len(self.frame_times),
                    'avg_frame_time': sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
                }
            }

            import json
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            self.logger.info(f"System state saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error saving system state: {e}")

    def load_system_state(self, filename: str) -> bool:
        """Load system state from file."""
        try:
            import json
            with open(filename, 'r') as f:
                state = json.load(f)

            self.logger.info(f"System state loaded from {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading system state: {e}")
            return False

    def signal_handler(self, signum, frame):
        """Handle system signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

def main():
    """Main function to run the system."""
    # Create system
    system = DepthSLAMSystem()

    # Setup signal handlers
    signal.signal(signal.SIGINT, system.signal_handler)
    signal.signal(signal.SIGTERM, system.signal_handler)

    try:
        # Start system
        if system.start():
            print("Depth SLAM system started. Press 'q' in any window to quit.")
            print("Press 's' to save screenshots.")

            # Keep main thread alive
            while system.running:
                time.sleep(1)

                # Print status every 30 seconds
                if int(time.time()) % 30 == 0:
                    status = system.get_system_status()
                    print(f"System running - Camera FPS: {status['camera_fps']}")

        else:
            print("Failed to start system")
            return 1

    except KeyboardInterrupt:
        print("Received keyboard interrupt")
    except Exception as e:
        print(f"Error running system: {e}")
        return 1
    finally:
        system.stop()

    return 0

if __name__ == "__main__":
    exit(main())