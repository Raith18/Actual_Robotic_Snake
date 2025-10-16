# Real-time Monocular Depth SLAM System

A comprehensive Python application for real-time 3D scene reconstruction using monocular depth estimation, object detection, and SLAM functionality.

## Features

### Core Components
- **Real-time Webcam Capture**: High-performance video capture with frame buffering
- **MiDaS Depth Estimation**: State-of-the-art monocular depth estimation with GPU acceleration
- **3D Point Cloud Generation**: Convert depth maps to colored 3D point clouds
- **Advanced Point Cloud Processing**: Voxel grid downsampling, statistical outlier removal, Kalman filtering
- **YOLOv8 Object Detection**: Real-time object detection and tracking with 3D integration
- **Edge Detection & Error Analysis**: Advanced error detection and distance calculations
- **SLAM System**: Camera pose estimation and 3D map building
- **Dual Visualization**: Synchronized 3D point cloud viewer and analytical 2D view

### Advanced Features
- **GPU Acceleration**: PyTorch + CUDA optimization for real-time performance
- **Modular Architecture**: Clean separation of concerns with extensible design
- **Real-time Performance**: Optimized for 30+ FPS processing
- **Error Visualization**: Visual feedback for depth estimation quality
- **Object Tracking**: Persistent object tracking across frames
- **Keyframe Management**: Efficient SLAM keyframe selection and management

## Installation

### Prerequisites
- Windows 11
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- Webcam camera

### Dependencies Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install MiDaS models (first run will download automatically)
python -c "from models.depth_estimator import MiDaSDepthEstimator; de = MiDaSDepthEstimator()"

# Download YOLOv8 model
python -c "from models.object_detector import ObjectDetector; od = ObjectDetector()"
```

## Usage

### Basic Usage
```bash
# Run the complete system
python main.py
```

### Configuration
Edit `config/system_config.yaml` to customize:
- Camera settings (resolution, FPS)
- Model parameters (depth estimation, object detection)
- Processing options (filters, thresholds)
- Visualization settings

### Controls
- **'q'**: Quit application
- **'s'**: Save screenshots of both views
- **'r'**: Reset 3D camera view

## Architecture

### Module Structure
```
mono_depth_slam_python_win/
├── core/                 # Core system components
│   ├── camera_capture.py    # Webcam capture
│   └── main_system.py       # Main orchestrator
├── models/               # ML models
│   ├── depth_estimator.py   # MiDaS integration
│   ├── point_cloud_generator.py  # 3D point cloud creation
│   └── object_detector.py   # YOLOv8 detection
├── processing/           # Data processing
│   ├── point_cloud_processor.py  # Advanced filtering
│   └── error_analysis.py    # Edge detection & errors
├── visualization/        # Visualization system
│   └── visualizer.py        # Dual view system
├── slam/                 # SLAM functionality
│   └── slam_system.py       # Pose estimation & mapping
├── utils/                # Utilities
│   └── common.py           # Common functions
├── config/               # Configuration
│   └── system_config.yaml   # System settings
└── main.py              # Application entry point
```

### Data Flow
1. **Capture**: Webcam frames captured in separate thread
2. **Depth Estimation**: MiDaS processes frames for depth maps
3. **Point Cloud**: Depth maps converted to 3D point clouds
4. **Processing**: Advanced filtering and optimization applied
5. **Object Detection**: YOLOv8 detects and tracks objects
6. **SLAM**: Camera pose estimation and map updates
7. **Error Analysis**: Edge detection and error marking
8. **Visualization**: Dual synchronized views updated

## Performance Optimization

### GPU Acceleration
- MiDaS depth estimation with CUDA
- PyTorch tensor operations on GPU
- Optimized memory management

### Real-time Processing
- Multi-threaded architecture
- Frame buffering and queue management
- Efficient data structures
- Memory pooling where applicable

### Configuration Tuning
```yaml
performance:
  target_fps: 30
  gpu_memory_fraction: 0.8
  enable_cuda: true
  num_threads: 4
  queue_size: 10
```

## Output and Visualization

### 3D Point Cloud View
- Real-time 3D point cloud rendering
- Object detection overlays
- Camera trajectory visualization
- Coordinate frame display

### Analytical View
- RGB image with depth overlay
- Edge detection visualization
- Error point marking
- Distance measurements
- Performance metrics

### Data Export
- Point cloud saving (PLY/PLY format)
- Screenshot capture
- System state serialization
- Performance logging

## Troubleshooting

### Common Issues

**Camera not found:**
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

**CUDA out of memory:**
```yaml
performance:
  gpu_memory_fraction: 0.6  # Reduce memory usage
```

**Low frame rate:**
```yaml
camera:
  width: 640    # Reduce resolution
  height: 480
  fps: 30
```

**Model download issues:**
- Ensure internet connection
- Check firewall settings
- Models download to system cache automatically

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extension Points

### Adding New Models
1. Implement model interface in `models/`
2. Add configuration parameters
3. Integrate with main processing loop

### Custom Processing
1. Extend processors in `processing/`
2. Add configuration options
3. Update main system integration

### Additional Visualizations
1. Implement in `visualization/`
2. Add to dual view system
3. Update configuration

## License

This project is designed for research and educational purposes. Please ensure compliance with all library licenses when using in production.

## Citation

If you use this system in your research, please cite:

```
@software{depth_slam_system,
  title={Real-time Monocular Depth SLAM System},
  author={AI Assistant},
  year={2024}
}