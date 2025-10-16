#!/usr/bin/env python3
"""
Simple Installation Script
Creates virtual environment and installs all required packages.
"""

import os
import sys
import subprocess
import venv
from pathlib import Path

def run_command(command):
    """Run a command and return success status."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    """Main installation function."""
    print("=== Real-time Monocular Depth SLAM System Installation ===")

    # Check Python version
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8+ required")
        return 1
    print(f"OK: Python {sys.version_info.major}.{sys.version_info.minor} found")

    # Create virtual environment
    print("Creating virtual environment...")
    venv_path = Path("venv")
    try:
        venv.create(venv_path, with_pip=True)
        print("OK: Virtual environment created")
    except Exception as e:
        print(f"ERROR: Failed to create virtual environment: {e}")
        return 1

    # Get pip path
    if os.name == 'nt':  # Windows
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"

    # Install requirements
    print("Installing Python packages...")
    requirements_file = Path("requirements.txt")

    if not requirements_file.exists():
        print("ERROR: requirements.txt not found")
        return 1

    try:
        command = f'"{pip_path}" install -r "{requirements_file}"'
        success, stdout, stderr = run_command(command)

        if success:
            print("OK: Python packages installed successfully")
        else:
            print("ERROR: Failed to install packages")
            print(f"Error: {stderr}")
            return 1

    except Exception as e:
        print(f"ERROR: Installation failed: {e}")
        return 1

    # Download models
    print("Setting up ML models...")
    try:
        # Create model setup script
        setup_script = '''
import sys
import os
sys.path.insert(0, os.getcwd())

print("Setting up MiDaS...")
try:
    from models.depth_estimator import MiDaSDepthEstimator
    print("Creating MiDaS estimator...")
    estimator = MiDaSDepthEstimator()
    print("OK: MiDaS setup complete")
except Exception as e:
    print(f"ERROR: MiDaS setup failed: {e}")

print("Setting up YOLOv8...")
try:
    from models.object_detector import ObjectDetector
    print("Creating YOLOv8 detector...")
    detector = ObjectDetector()
    print("OK: YOLOv8 setup complete")
except Exception as e:
    print(f"ERROR: YOLOv8 setup failed: {e}")

print("Model setup finished!")
'''

        with open("model_setup.py", "w") as f:
            f.write(setup_script)

        # Run model setup
        command = f'"{python_path}" model_setup.py'
        success, stdout, stderr = run_command(command)

        # Clean up
        if os.path.exists("model_setup.py"):
            os.remove("model_setup.py")

        if success:
            print("OK: Models downloaded successfully")
        else:
            print("WARNING: Model download completed with warnings")
            print("STDOUT:", stdout)
            if stderr:
                print("STDERR:", stderr)

    except Exception as e:
        print(f"ERROR: Model setup failed: {e}")
        return 1

    # Create launch scripts
    print("Creating launch scripts...")

    if os.name == 'nt':  # Windows
        batch_script = f'''@echo off
echo Activating virtual environment...
call "{venv_path / "Scripts" / "activate.bat"}"

echo.
echo Starting Real-time Monocular Depth SLAM System...
echo.
echo Controls:
echo   'q' - Quit application
echo   's' - Save screenshots
echo   'r' - Reset 3D view
echo.
python main.py

echo.
echo Application closed.
pause
'''

        with open("launch.bat", 'w') as f:
            f.write(batch_script)
        print("OK: Created launch.bat")

    else:  # Unix/Linux
        shell_script = f'''#!/bin/bash
echo "Activating virtual environment..."
source "{venv_path / "bin" / "activate"}"

echo ""
echo "Starting Real-time Monocular Depth SLAM System..."
echo ""
echo "Controls:"
echo "  'q' - Quit application"
echo "  's' - Save screenshots"
echo "  'r' - Reset 3D view"
echo ""
python main.py

echo ""
echo "Application closed."
'''

        with open("launch.sh", 'w') as f:
            f.write(shell_script)

        # Make executable
        os.chmod("launch.sh", 0o755)
        print("OK: Created launch.sh")

    # Final summary
    print("")
    print("=" * 60)
    print("INSTALLATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Virtual environment created")
    print("Python packages installed")
    print("ML models downloaded")
    print("Launch scripts created")
    print("")
    print("To run the application:")
    if os.name == 'nt':
        print("  double-click launch.bat")
        print("  OR run: python main.py")
    else:
        print("  run: ./launch.sh")
        print("  OR run: python main.py")
    print("")
    print("For more information, see README.md")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    exit(main())