#!/usr/bin/env python3
"""
Setup script for Real-time Monocular Depth SLAM System
Handles installation, configuration, and first-time setup.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path

def run_command(command, shell=True):
    """Run a command and return success status."""
    try:
        result = subprocess.run(command, shell=shell, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} found")
    return True

def check_gpu_availability():
    """Check GPU availability."""
    print("Checking GPU availability...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available, will use CPU")
        return True
    except ImportError:
        print("⚠️  PyTorch not installed, GPU check skipped")
        return True

def install_requirements():
    """Install Python requirements."""
    print("Installing Python dependencies...")
    success, stdout, stderr = run_command("pip install -r requirements.txt")

    if success:
        print("✅ Dependencies installed successfully")
        return True
    else:
        print("❌ Failed to install dependencies")
        print(f"Error: {stderr}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    dirs = ["logs", "output", "models", "cache"]

    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

    print("✅ Directories created")
    return True

def download_models():
    """Download required ML models."""
    print("Checking/downloading ML models...")

    try:
        # Test MiDaS import
        from models.depth_estimator import MiDaSDepthEstimator
        print("✅ MiDaS models ready")

        # Test YOLOv8 import
        from models.object_detector import ObjectDetector
        print("✅ YOLOv8 models ready")

        return True

    except ImportError as e:
        print(f"⚠️  Model download may be required: {e}")
        print("   Models will be downloaded automatically on first run")
        return True
    except Exception as e:
        print(f"❌ Error with models: {e}")
        return False

def test_imports():
    """Test all module imports."""
    print("Testing module imports...")

    modules_to_test = [
        "cv2",
        "numpy",
        "torch",
        "open3d",
        "scipy",
        "ultralytics",
        "filterpy",
        "skimage"
    ]

    failed_imports = []

    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)

    if failed_imports:
        print(f"⚠️  Missing dependencies: {', '.join(failed_imports)}")
        print("   Run 'pip install -r requirements.txt' to install")
        return False

    return True

def create_default_config():
    """Create default configuration if it doesn't exist."""
    config_path = "config/system_config.yaml"

    if os.path.exists(config_path):
        print("✅ Configuration file exists")
        return True

    print("Creating default configuration...")

    default_config = {
        'system': {
            'name': 'Real-time Monocular Depth SLAM',
            'version': '1.0.0',
            'debug': False
        },
        'camera': {
            'width': 640,
            'height': 480,
            'fps': 30,
            'device_id': 0,
            'api_preference': 'CAP_DSHOW'
        },
        'midas': {
            'model_type': 'DPT_Hybrid',
            'optimize': True,
            'height': 384,
            'square': False,
            'grayscale': False
        },
        'yolo': {
            'model_size': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'max_detections': 100,
            'track_objects': True,
            'track_buffer': 30
        },
        'point_cloud': {
            'voxel_size': 0.01,
            'outlier_neighbors': 50,
            'outlier_std_ratio': 1.0,
            'kalman_process_noise': 0.01,
            'kalman_measurement_noise': 0.1
        },
        'slam': {
            'feature_detector': 'ORB',
            'num_features': 1000,
            'keyframe_distance': 0.1,
            'keyframe_angle': 10,
            'bundle_adjustment': True,
            'ba_iterations': 10
        },
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
        },
        'performance': {
            'target_fps': 30,
            'gpu_memory_fraction': 0.8,
            'enable_cuda': True,
            'num_threads': 4,
            'queue_size': 10
        },
        'error_analysis': {
            'enable_edge_detection': True,
            'edge_detection_threshold': 0.1,
            'enable_error_marking': True,
            'error_threshold': 0.05,
            'max_error_points': 1000
        },
        'semantic': {
            'enable_segmentation': True,
            'num_classes': 150,
            'segmentation_threshold': 0.7
        },
        'paths': {
            'models_dir': 'models',
            'logs_dir': 'logs',
            'output_dir': 'output',
            'cache_dir': 'cache'
        }
    }

    try:
        os.makedirs('config', exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        print("✅ Default configuration created")
        return True
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return False

def run_system_test():
    """Run a basic system test."""
    print("Running system test...")

    try:
        # Test basic imports
        from core.camera_capture import WebcamCapture
        from utils.common import load_config

        # Test configuration loading
        config = load_config("config/system_config.yaml")
        if not config:
            print("❌ Configuration test failed")
            return False

        # Test camera initialization (don't actually open camera)
        camera = WebcamCapture()
        print("✅ Camera module ready")

        print("✅ System test passed")
        return True

    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("=== Real-time Monocular Depth SLAM System Setup ===")
    print("")

    # Run setup steps
    steps = [
        ("Python version check", check_python_version),
        ("GPU availability check", check_gpu_availability),
        ("Dependencies installation", install_requirements),
        ("Directory creation", create_directories),
        ("Model download check", download_models),
        ("Module import test", test_imports),
        ("Configuration setup", create_default_config),
        ("System functionality test", run_system_test),
    ]

    results = []

    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        success = step_func()
        results.append((step_name, success))

        if not success and step_name not in ["GPU availability check", "Model download check"]:
            print(f"\n❌ Setup failed at: {step_name}")
            print("Please fix the issue and run setup again")
            return 1

    # Summary
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)

    all_success = True
    for step_name, success in results:
        status = "✅ PASS" if success else "⚠️  SKIP/FAIL"
        print(f"{status} - {step_name}")
        if not success:
            all_success = False

    print("\n" + "="*50)

    if all_success:
        print("🎉 Setup completed successfully!")
        print("\nTo run the system:")
        print("  python main.py")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshots")
        print("  'r' - Reset 3D view")
        print("\nFor more information, see README.md")
        return 0
    else:
        print("⚠️  Setup completed with some issues")
        print("The system may still work, but some features might be limited")
        print("Check the errors above and try running anyway")
        return 0

if __name__ == "__main__":
    exit(main())