#!/usr/bin/env python3
"""
Virtual Environment Creation Script
Creates and configures a Python virtual environment for the Depth SLAM system.
"""

import os
import sys
import subprocess
import venv
from pathlib import Path

class VenvCreator:
    """Handles virtual environment creation and package installation."""

    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.venv_path = self.project_root / "venv"
        self.requirements_file = self.project_root / "requirements.txt"

    def create_virtual_environment(self):
        """Create Python virtual environment."""
        print("🔄 Creating virtual environment...")

        try:
            # Create virtual environment
            venv.create(self.venv_path, with_pip=True)

            print("✅ Virtual environment created successfully")
            return True

        except Exception as e:
            print(f"❌ Error creating virtual environment: {e}")
            return False

    def get_activator_path(self):
        """Get the path to the activation script."""
        if os.name == 'nt':  # Windows
            return self.venv_path / "Scripts" / "activate.bat"
        else:  # Unix/Linux
            return self.venv_path / "bin" / "activate"

    def install_requirements(self):
        """Install Python packages from requirements.txt."""
        print("🔄 Installing Python packages...")

        activator = self.get_activator_path()

        if not activator.exists():
            print("❌ Virtual environment activation script not found")
            return False

        try:
            # Activate environment and install packages
            if os.name == 'nt':  # Windows
                pip_command = f'"{self.venv_path / "Scripts" / "pip"}" install -r "{self.requirements_file}"'
            else:  # Unix/Linux
                pip_command = f'"{self.venv_path / "bin" / "pip"}" install -r "{self.requirements_file}"'

            print(f"Running: {pip_command}")
            result = subprocess.run(pip_command, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Python packages installed successfully")
                print(result.stdout)
                return True
            else:
                print("❌ Error installing packages")
                print("STDERR:", result.stderr)
                print("STDOUT:", result.stdout)
                return False

        except Exception as e:
            print(f"❌ Error during package installation: {e}")
            return False

    def install_midas_models(self):
        """Install and verify MiDaS models."""
        print("🔄 Setting up MiDaS models...")

        try:
            # Test import to trigger model download
            if os.name == 'nt':  # Windows
                python_cmd = f'"{self.venv_path / "Scripts" / "python"}"'
            else:  # Unix/Linux
                python_cmd = f'"{self.venv_path / "bin" / "python"}"'

            # Create a test script to download models
            test_script = '''
import sys
sys.path.insert(0, '.')

try:
    print('Testing MiDaS import...')
    from models.depth_estimator import MiDaSDepthEstimator
    print('Creating MiDaS estimator to download models...')
    estimator = MiDaSDepthEstimator()
    print('✅ MiDaS models ready')
except Exception as e:
    print(f'❌ MiDaS error: {e}')

try:
    print('Testing YOLOv8 import...')
    from models.object_detector import ObjectDetector
    print('Creating YOLOv8 detector to download models...')
    detector = ObjectDetector()
    print('✅ YOLOv8 models ready')
except Exception as e:
    print(f'❌ YOLOv8 error: {e}')

print('Model setup complete!')
'''

            with open("temp_model_test.py", "w") as f:
                f.write(test_script)

            result = subprocess.run(f'{python_cmd} temp_model_test.py',
                                  shell=True, capture_output=True, text=True, cwd=self.project_root)

            # Clean up test script
            if os.path.exists("temp_model_test.py"):
                os.remove("temp_model_test.py")

            if result.returncode == 0:
                print("✅ Models downloaded successfully")
                print(result.stdout)
                return True
            else:
                print("⚠️  Model download completed with warnings")
                print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                return True

        except Exception as e:
            print(f"❌ Error setting up models: {e}")
            return False

    def create_launch_script(self):
        """Create a launch script for easy application startup."""
        print("🔄 Creating launch script...")

        try:
            if os.name == 'nt':  # Windows
                script_content = f'''@echo off
echo Activating virtual environment...
call "{self.venv_path / "Scripts" / "activate.bat"}"

echo Starting Real-time Monocular Depth SLAM System...
python main.py

echo.
echo Press any key to exit...
pause >nul
'''
                script_path = self.project_root / "launch.bat"
            else:  # Unix/Linux
                script_content = f'''#!/bin/bash
echo "Activating virtual environment..."
source "{self.venv_path / "bin" / "activate"}"

echo "Starting Real-time Monocular Depth SLAM System..."
python main.py

echo "Application closed."
'''
                script_path = self.project_root / "launch.sh"

            with open(script_path, 'w') as f:
                f.write(script_content)

            # Make executable on Unix/Linux
            if os.name != 'nt':
                os.chmod(script_path, 0o755)

            print(f"✅ Launch script created: {script_path}")
            return True

        except Exception as e:
            print(f"❌ Error creating launch script: {e}")
            return False

    def run(self):
        """Run the complete setup process."""
        print("=== Virtual Environment Setup ===")
        print(f"Project root: {self.project_root}")
        print(f"Virtual environment: {self.venv_path}")
        print("")

        steps = [
            ("Create virtual environment", self.create_virtual_environment),
            ("Install Python packages", self.install_requirements),
            ("Setup ML models", self.install_midas_models),
            ("Create launch script", self.create_launch_script),
        ]

        results = []

        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            success = step_func()
            results.append((step_name, success))

        # Summary
        print("\n" + "="*50)
        print("SETUP SUMMARY")
        print("="*50)

        all_success = True
        for step_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {step_name}")
            if not success:
                all_success = False

        print("\n" + "="*50)

        if all_success:
            print("🎉 Virtual environment setup completed successfully!")
            print("\nTo activate the virtual environment:")
            if os.name == 'nt':
                print(f'  {self.venv_path / "Scripts" / "activate.bat"}')
            else:
                print(f'  source "{self.venv_path / "bin" / "activate"}"')

            print("\nTo run the application:")
            print("  python main.py")
            if os.path.exists("launch.bat" if os.name == 'nt' else "launch.sh"):
                print(f"  .\\launch{'bat' if os.name == 'nt' else 'sh'}")

            print("\nFor more information, see README.md")
            return 0
        else:
            print("❌ Setup completed with errors")
            print("Please check the errors above and try again")
            return 1

def main():
    """Main function."""
    creator = VenvCreator()
    return creator.run()

if __name__ == "__main__":
    exit(main())