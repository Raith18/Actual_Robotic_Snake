#!/usr/bin/env python3
"""
Main launcher for the Real-time Monocular Depth SLAM System
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the depth SLAM system."""
    try:
        # Add current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))

        # Import and run main system
        from core.main_system import main as system_main

        print("=== Real-time Monocular Depth SLAM System ===")
        print("Starting system...")
        print("Features:")
        print("- Real-time webcam capture")
        print("- MiDaS depth estimation")
        print("- 3D point cloud generation")
        print("- Advanced point cloud processing")
        print("- YOLOv8 object detection & tracking")
        print("- Edge detection & error analysis")
        print("- SLAM functionality")
        print("- Dual synchronized visualization")
        print("")
        print("Controls:")
        print("- 'q': Quit application")
        print("- 's': Save screenshots")
        print("- 'r': Reset 3D view")
        print("")

        # Run system
        exit_code = system_main()

        print("System shutdown complete")
        return exit_code

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"Error starting system: {e}")
        return 1

if __name__ == "__main__":
    exit(main())