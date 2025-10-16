"""
MiDaS Depth Estimation Module
Handles monocular depth estimation using MiDaS models with GPU acceleration.
"""

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import numpy as np
import cv2
import time
import logging
from typing import Optional, Tuple, Dict, Any
import yaml
from pathlib import Path

class MiDaSDepthEstimator:
    """
    MiDaS-based depth estimation with GPU acceleration and optimization.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        Initialize MiDaS depth estimator.

        Args:
            config_path: Path to system configuration file
        """
        self.config = self._load_config(config_path)
        self.midas_config = self.config['midas']

        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config['performance']['enable_cuda'] else 'cpu')
        self.model = None
        self.transform = None

        # Model parameters
        self.model_type = self.midas_config['model_type']
        self.optimize = self.midas_config['optimize']

        # Performance tracking
        self.inference_times = []
        self.max_inference_history = 100

        # Setup logging
        self.logger = logging.getLogger('MiDaSDepthEstimator')

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
                'midas': {
                    'model_type': 'DPT_Hybrid',
                    'optimize': True,
                    'height': 384,
                    'square': False,
                    'grayscale': False
                },
                'performance': {
                    'enable_cuda': True
                }
            }

    def _load_model(self):
        """Load MiDaS model with optimizations."""
        try:
            self.logger.info(f"Loading MiDaS model: {self.model_type}")

            # Import MiDaS
            if self.model_type == "DPT_Large":
                from midas.dpt_depth import DPTDepthModel
                model_path = "models/dpt_large_384.pt"
                model = DPTDepthModel(
                    path=model_path,
                    backbone="vitl16_384",
                    non_negative=True,
                )
            elif self.model_type == "DPT_Hybrid":
                from midas.dpt_depth import DPTDepthModel
                model_path = "models/dpt_hybrid_384.pt"
                model = DPTDepthModel(
                    path=model_path,
                    backbone="vitb_rn50_384",
                    non_negative=True,
                )
            elif self.model_type == "MiDaS_small":
                from midas.midas_net import MidasNet
                model_path = "models/midas_v21_small_256.pt"
                model = MidasNet(model_path, features=64, backbone="efficientnet_lite3",
                               exportable=True, non_negative=True, blocks={'expand': True})
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Move to device
            model.to(self.device)
            model.eval()

            # Apply optimizations
            if self.optimize and self.device.type == 'cuda':
                model = torch.jit.script(model)
                # Use mixed precision
                try:
                    from torch.cuda.amp import autocast
                    self.autocast = autocast
                except ImportError:
                    self.autocast = None
            else:
                self.autocast = None

            self.model = model

            # Create transform pipeline
            self._create_transform()

            self.logger.info(f"MiDaS model loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Error loading MiDaS model: {e}")
            self._fallback_to_opencv()

    def _create_transform(self):
        """Create image transformation pipeline."""
        target_height = self.midas_config['height']

        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            # DPT models expect 384x384 input
            self.transform = Compose([
                Resize((target_height, target_height)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # MiDaS v2.1 small expects 256x256 input
            self.transform = Compose([
                Resize((target_height, target_height)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _fallback_to_opencv(self):
        """Fallback to OpenCV-based depth estimation if MiDaS fails."""
        self.logger.warning("Using OpenCV fallback for depth estimation")
        self.model = None
        self.transform = None

    def estimate_depth(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Estimate depth from RGB image.

        Args:
            image: Input RGB image (H, W, 3)

        Returns:
            Tuple of (depth_map, inference_time)
        """
        if image is None:
            return None, 0.0

        start_time = time.time()

        try:
            if self.model is None:
                # Fallback to simple depth estimation
                depth_map = self._estimate_depth_opencv(image)
                inference_time = time.time() - start_time
                return depth_map, inference_time

            # Preprocess image
            original_size = image.shape[:2]

            # Convert to RGB if needed
            if len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply transforms
            input_tensor = self.transform(rgb_image).unsqueeze(0)

            # Move to device
            input_tensor = input_tensor.to(self.device)

            # Inference with optional mixed precision
            with torch.no_grad():
                if self.autocast and self.device.type == 'cuda':
                    with self.autocast():
                        prediction = self.model.forward(input_tensor)
                else:
                    prediction = self.model.forward(input_tensor)

                # Resize to original size
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=original_size,
                    mode='bicubic',
                    align_corners=False
                ).squeeze()

                depth_map = prediction.cpu().numpy()

            inference_time = time.time() - start_time

            # Track performance
            self.inference_times.append(inference_time)
            if len(self.inference_times) > self.max_inference_history:
                self.inference_times.pop(0)

            return depth_map, inference_time

        except Exception as e:
            self.logger.error(f"Error in depth estimation: {e}")
            inference_time = time.time() - start_time
            return None, inference_time

    def _estimate_depth_opencv(self, image: np.ndarray) -> np.ndarray:
        """
        Simple depth estimation using OpenCV (fallback method).

        Args:
            image: Input image

        Returns:
            Estimated depth map
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Laplacian for edge detection
        edges = cv2.Laplacian(gray, cv2.CV_64F)

        # Convert to depth (simple heuristic)
        depth_map = cv2.convertScaleAbs(edges)

        # Normalize to [0, 1]
        depth_map = depth_map.astype(np.float32) / 255.0

        # Apply Gaussian blur for smoothing
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)

        return depth_map

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for MiDaS input.

        Args:
            image: Input RGB image

        Returns:
            Preprocessed tensor
        """
        if self.transform is None:
            return torch.zeros(1, 3, 384, 384)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        return self.transform(rgb_image).unsqueeze(0)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'optimize': self.optimize,
            'input_size': self.midas_config['height'],
            'cuda_available': torch.cuda.is_available(),
            'memory_allocated': self._get_gpu_memory() if torch.cuda.is_available() else 0
        }

    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def get_average_inference_time(self) -> float:
        """Get average inference time."""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)

    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def get_supported_models(self) -> list:
        """Get list of supported MiDaS models."""
        return [
            "DPT_Large",
            "DPT_Hybrid",
            "MiDaS_small"
        ]

    def download_model(self, model_type: str) -> bool:
        """
        Download MiDaS model weights.

        Args:
            model_type: Type of model to download

        Returns:
            True if successful
        """
        try:
            import urllib.request
            from midas.model_loader import load_model

            self.logger.info(f"Downloading {model_type} model...")

            # This would typically download from the official repository
            # For now, we'll use the model loader
            model = load_model(model_type.lower())

            self.logger.info(f"Successfully downloaded {model_type}")
            return True

        except Exception as e:
            self.logger.error(f"Error downloading model: {e}")
            return False

    def save_depth_map(self, depth_map: np.ndarray, filename: str):
        """
        Save depth map to file.

        Args:
            depth_map: Depth map to save
            filename: Output filename
        """
        try:
            # Normalize for visualization
            normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            colored_depth = (normalized_depth * 255).astype(np.uint8)

            # Apply colormap
            colored_depth = cv2.applyColorMap(colored_depth, cv2.COLORMAP_JET)

            # Save image
            cv2.imwrite(filename, colored_depth)

        except Exception as e:
            self.logger.error(f"Error saving depth map: {e}")

    def load_depth_map(self, filename: str) -> Optional[np.ndarray]:
        """
        Load depth map from file.

        Args:
            filename: Input filename

        Returns:
            Loaded depth map or None
        """
        try:
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None

            return image.astype(np.float32) / 255.0

        except Exception as e:
            self.logger.error(f"Error loading depth map: {e}")
            return None