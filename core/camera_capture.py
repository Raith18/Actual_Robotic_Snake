"""
Real-time webcam capture module for monocular depth SLAM system.
Handles camera initialization, frame capture, and preprocessing.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, Any
from threading import Lock, Thread
import queue
import yaml

class WebcamCapture:
    """
    Real-time webcam capture with frame buffering and synchronization.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        Initialize webcam capture.

        Args:
            config_path: Path to system configuration file
        """
        self.config = self._load_config(config_path)
        self.camera_config = self.config['camera']

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

        # Threading components
        self.frame_queue = queue.Queue(maxsize=self.config['performance']['queue_size'])
        self.lock = Lock()
        self.capture_thread: Optional[Thread] = None

        # Frame dimensions
        self.width = self.camera_config['width']
        self.height = self.camera_config['height']
        self.target_fps = self.camera_config['fps']

        # Setup logging
        self.logger = self._setup_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config: {e}")
            # Return default config
            return {
                'camera': {
                    'width': 640,
                    'height': 480,
                    'fps': 30,
                    'device_id': 0,
                    'api_preference': 'CAP_DSHOW'
                },
                'performance': {
                    'queue_size': 10
                }
            }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('WebcamCapture')

    def initialize_camera(self) -> bool:
        """
        Initialize the webcam capture.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Set camera API preference for Windows
            api_preference = getattr(cv2, self.camera_config['api_preference'])

            self.cap = cv2.VideoCapture(self.camera_config['device_id'], api_preference)

            if not self.cap.isOpened():
                self.logger.error("Failed to open webcam")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")

            return True

        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            return False

    def start_capture(self) -> bool:
        """
        Start the capture thread.

        Returns:
            bool: True if successful, False otherwise
        """
        if self.cap is None:
            self.logger.error("Camera not initialized")
            return False

        self.running = True
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        self.logger.info("Capture thread started")
        return True

    def stop_capture(self):
        """Stop the capture thread and release resources."""
        self.running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()

        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.logger.info("Capture stopped")

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        frame_interval = 1.0 / self.target_fps

        while self.running:
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("Failed to capture frame")
                time.sleep(0.1)
                continue

            # Put frame in queue if space available
            try:
                self.frame_queue.put_nowait(frame.copy())
                self.frame_count += 1
            except queue.Full:
                # Remove oldest frame if queue is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Empty:
                    pass

            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                with self.lock:
                    self.fps = self.frame_count
                    self.frame_count = 0
                self.last_time = current_time

            # Maintain target frame rate
            elapsed_time = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed_time)
            time.sleep(sleep_time)

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the latest frame from the queue.

        Args:
            timeout: Maximum time to wait for frame

        Returns:
            Latest frame or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_fps(self) -> int:
        """Get current FPS."""
        with self.lock:
            return self.fps

    def is_capturing(self) -> bool:
        """Check if capture is running."""
        return self.running and self.capture_thread.is_alive()

    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information."""
        if self.cap is None:
            return {}

        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'format': str(self.cap.get(cv2.CAP_PROP_FORMAT)),
            'backend': str(self.cap.get(cv2.CAP_PROP_BACKEND))
        }

    def set_exposure(self, exposure: float) -> bool:
        """
        Set camera exposure.

        Args:
            exposure: Exposure value (-10 to -1 for manual control)

        Returns:
            bool: True if successful
        """
        if self.cap is None:
            return False

        return self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def set_focus(self, focus: float) -> bool:
        """
        Set camera focus.

        Args:
            focus: Focus value (0-255)

        Returns:
            bool: True if successful
        """
        if self.cap is None:
            return False

        return self.cap.set(cv2.CAP_PROP_FOCUS, focus)

    def auto_focus(self) -> bool:
        """Enable auto focus."""
        if self.cap is None:
            return False

        return self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    def get_supported_resolutions(self) -> list:
        """Get list of supported camera resolutions."""
        if self.cap is None:
            return []

        resolutions = []
        common_resolutions = [
            (640, 480), (800, 600), (1024, 768),
            (1280, 720), (1920, 1080), (2560, 1440)
        ]

        for width, height in common_resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_width == width and actual_height == height:
                resolutions.append((width, height))

        return resolutions