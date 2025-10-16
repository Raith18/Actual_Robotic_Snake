"""
Common utility functions for the monocular depth SLAM system.
"""

import numpy as np
import cv2
import yaml
import json
import time
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading config {config_path}: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
    except Exception as e:
        logging.error(f"Error saving config {config_path}: {e}")

def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/{name.lower()}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def create_directories(dirs: list):
    """Create directories if they don't exist."""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def normalize_depth(depth_map: np.ndarray, min_depth: float = 0.1, max_depth: float = 10.0) -> np.ndarray:
    """
    Normalize depth map to [0, 1] range.

    Args:
        depth_map: Input depth map
        min_depth: Minimum depth value
        max_depth: Maximum depth value

    Returns:
        Normalized depth map
    """
    depth_map = np.clip(depth_map, min_depth, max_depth)
    return (depth_map - min_depth) / (max_depth - min_depth)

def apply_colormap(depth_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Apply colormap to depth map for visualization.

    Args:
        depth_map: Normalized depth map [0, 1]
        colormap: OpenCV colormap

    Returns:
        Colorized depth map
    """
    depth_255 = (depth_map * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_255, colormap)

def resize_image(image: np.ndarray, target_size: Tuple[int, int], keep_aspect: bool = False) -> np.ndarray:
    """
    Resize image to target size.

    Args:
        image: Input image
        target_size: Target (width, height)
        keep_aspect: Maintain aspect ratio

    Returns:
        Resized image
    """
    if keep_aspect:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(image, target_size)

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Preprocess image for depth estimation.

    Args:
        image: Input RGB image
        target_size: Target size for resizing

    Returns:
        Preprocessed image
    """
    if target_size:
        image = resize_image(image, target_size)

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image

def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate frames per second.

    Args:
        start_time: Start time
        frame_count: Number of frames processed

    Returns:
        FPS value
    """
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0.0

def smooth_depth_map(depth_map: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to depth map.

    Args:
        depth_map: Input depth map
        sigma: Gaussian sigma

    Returns:
        Smoothed depth map
    """
    return cv2.GaussianBlur(depth_map, (0, 0), sigma)

def compute_depth_statistics(depth_map: np.ndarray) -> Dict[str, float]:
    """
    Compute depth map statistics.

    Args:
        depth_map: Input depth map

    Returns:
        Dictionary with statistics
    """
    valid_depth = depth_map[depth_map > 0]

    if len(valid_depth) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }

    return {
        'mean': float(np.mean(valid_depth)),
        'std': float(np.std(valid_depth)),
        'min': float(np.min(valid_depth)),
        'max': float(np.max(valid_depth)),
        'median': float(np.median(valid_depth))
    }

def create_point_cloud_from_depth(depth_map: np.ndarray,
                                 rgb_image: np.ndarray,
                                 fx: float = 500.0,
                                 fy: float = 500.0,
                                 cx: float = None,
                                 cy: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D point cloud from depth map and RGB image.

    Args:
        depth_map: Depth map
        rgb_image: RGB image
        fx, fy: Focal lengths
        cx, cy: Principal point coordinates

    Returns:
        Tuple of (points, colors) arrays
    """
    height, width = depth_map.shape

    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    # Create meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate 3D coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    # Stack to create point cloud
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Get colors
    colors = rgb_image.reshape(-1, 3) / 255.0

    # Filter out invalid points
    valid_mask = (z.reshape(-1) > 0) & (z.reshape(-1) < 10.0)  # Filter reasonable depth range
    points = points[valid_mask]
    colors = colors[valid_mask]

    return points, colors

def transform_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """
    Transform 3D points using 4x4 transformation matrix.

    Args:
        points: 3D points (N, 3)
        transformation: 4x4 transformation matrix

    Returns:
        Transformed points
    """
    # Add homogeneous coordinates
    points_homo = np.ones((len(points), 4))
    points_homo[:, :3] = points

    # Apply transformation
    transformed_homo = (transformation @ points_homo.T).T

    return transformed_homo[:, :3]

def compute_camera_matrix(width: int, height: int, fov: float = 60.0) -> Tuple[float, float, float, float]:
    """
    Compute camera intrinsic matrix parameters.

    Args:
        width: Image width
        height: Image height
        fov: Field of view in degrees

    Returns:
        Tuple of (fx, fy, cx, cy)
    """
    cx = width / 2.0
    cy = height / 2.0

    # Convert FOV to focal length
    fov_rad = np.radians(fov)
    fx = cx / np.tan(fov_rad / 2)
    fy = cy / np.tan(fov_rad / 2)

    return fx, fy, cx, cy

def save_point_cloud(points: np.ndarray, colors: np.ndarray, filename: str):
    """
    Save point cloud to file.

    Args:
        points: 3D points
        colors: Point colors
        filename: Output filename
    """
    try:
        import open3d as o3d

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save to file
        o3d.io.write_point_cloud(filename, pcd)

    except ImportError:
        print("Open3D not available for saving point cloud")
    except Exception as e:
        print(f"Error saving point cloud: {e}")

def load_point_cloud(filename: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load point cloud from file.

    Args:
        filename: Input filename

    Returns:
        Tuple of (points, colors) or (None, None) if failed
    """
    try:
        import open3d as o3d

        # Load point cloud
        pcd = o3d.io.read_point_cloud(filename)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        return points, colors

    except ImportError:
        print("Open3D not available for loading point cloud")
        return None, None
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return None, None

def timestamp_filename(prefix: str = "", extension: str = ".txt") -> str:
    """
    Generate timestamped filename.

    Args:
        prefix: Filename prefix
        extension: File extension

    Returns:
        Timestamped filename
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{extension}" if prefix else f"{timestamp}{extension}"

def print_system_info():
    """Print system information for debugging."""
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not available")

    try:
        import open3d
        print(f"Open3D version: {open3d.__version__}")
    except ImportError:
        print("Open3D not available")

    try:
        import ultralytics
        print(f"YOLO version: {ultralytics.__version__}")
    except ImportError:
        print("YOLO not available")

    print("=========================")

# Import sys for print_system_info
import sys