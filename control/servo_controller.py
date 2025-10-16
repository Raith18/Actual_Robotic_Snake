"""
Servo Controller for Robotic Snake
Controls MG996R continuous servos for snake locomotion and movement.
"""

import time
import threading
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
import serial
import json

class ServoType(Enum):
    """Types of servos supported."""
    MG996R = "MG996R"
    MG995 = "MG995"
    MG90S = "MG90S"
    CUSTOM = "CUSTOM"

class ControlMode(Enum):
    """Servo control modes."""
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    TRAJECTORY = "trajectory"

class MG996RController:
    """
    Controller for MG996R continuous rotation servos.
    """

    def __init__(self, servo_id: int, serial_port: str = "/dev/ttyUSB0"):
        """
        Initialize MG996R servo controller.

        Args:
            servo_id: Unique identifier for this servo
            serial_port: Serial port for communication
        """
        self.servo_id = servo_id
        self.serial_port = serial_port

        # Servo specifications
        self.pulse_width_min = 500   # microseconds
        self.pulse_width_max = 2500  # microseconds
        self.pulse_width_neutral = 1500  # microseconds

        # Operating range
        self.angle_min = -90.0  # degrees
        self.angle_max = 90.0   # degrees
        self.velocity_max = 360.0  # degrees/second

        # Current state
        self.current_angle = 0.0
        self.current_velocity = 0.0
        self.target_angle = 0.0
        self.target_velocity = 0.0

        # Control parameters
        self.kp_position = 1.0  # Proportional gain for position control
        self.ki_position = 0.1  # Integral gain for position control
        self.kd_position = 0.05 # Derivative gain for position control

        self.kp_velocity = 2.0  # Proportional gain for velocity control
        self.ki_velocity = 0.2  # Integral gain for velocity control

        # Error tracking for PID
        self.position_error_integral = 0.0
        self.velocity_error_integral = 0.0
        self.last_position_error = 0.0
        self.last_velocity_error = 0.0

        # Communication
        self.serial_connection = None
        self.is_connected = False

        # Setup logging
        self.logger = logging.getLogger(f'Servo_{servo_id}')

    def connect(self) -> bool:
        """Establish serial connection to servo controller."""
        try:
            self.serial_connection = serial.Serial(
                port=self.serial_port,
                baudrate=115200,
                timeout=0.1,
                write_timeout=0.1
            )
            self.is_connected = True
            self.logger.info(f"Servo {self.servo_id} connected successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error connecting servo {self.servo_id}: {e}")
            return False

    def disconnect(self):
        """Close serial connection."""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
            self.is_connected = False
            self.logger.info(f"Servo {self.servo_id} disconnected")
        except Exception as e:
            self.logger.error(f"Error disconnecting servo {self.servo_id}: {e}")

    def angle_to_pulse_width(self, angle: float) -> int:
        """
        Convert angle to pulse width for servo control.

        Args:
            angle: Target angle in degrees

        Returns:
            Pulse width in microseconds
        """
        # Clamp angle to valid range
        angle = max(self.angle_min, min(self.angle_max, angle))

        # Convert angle to pulse width
        angle_range = self.angle_max - self.angle_min
        pulse_range = self.pulse_width_max - self.pulse_width_min

        pulse_width = self.pulse_width_min + (
            (angle - self.angle_min) * pulse_range / angle_range
        )

        return int(pulse_width)

    def velocity_to_pulse_width(self, velocity: float) -> int:
        """
        Convert velocity to pulse width for continuous rotation servos.

        Args:
            velocity: Target velocity in degrees/second

        Returns:
            Pulse width in microseconds
        """
        # For continuous rotation servos, pulse width controls speed and direction
        # 1500 μs = stop, <1500 = one direction, >1500 = other direction

        velocity = max(-self.velocity_max, min(self.velocity_max, velocity))

        # Map velocity to pulse width deviation from neutral
        velocity_range = 2 * self.velocity_max  # Total range
        pulse_deviation_max = 600  # Maximum deviation from neutral (1500 ± 600)

        pulse_deviation = (velocity * pulse_deviation_max) / self.velocity_max

        # Add direction consideration
        if velocity >= 0:
            pulse_width = self.pulse_width_neutral + abs(pulse_deviation)
        else:
            pulse_width = self.pulse_width_neutral - abs(pulse_deviation)

        return int(pulse_width)

    def send_command(self, pulse_width: int) -> bool:
        """
        Send pulse width command to servo.

        Args:
            pulse_width: Pulse width in microseconds

        Returns:
            True if command sent successfully
        """
        if not self.is_connected or not self.serial_connection:
            return False

        try:
            # Create command packet
            command = {
                'servo_id': self.servo_id,
                'pulse_width': pulse_width,
                'timestamp': time.time()
            }

            # Send command as JSON
            command_str = json.dumps(command) + '\n'
            self.serial_connection.write(command_str.encode())

            return True

        except Exception as e:
            self.logger.error(f"Error sending command to servo {self.servo_id}: {e}")
            return False

    def set_angle(self, angle: float) -> bool:
        """
        Set servo to specific angle.

        Args:
            angle: Target angle in degrees

        Returns:
            True if command sent successfully
        """
        self.target_angle = angle
        pulse_width = self.angle_to_pulse_width(angle)

        success = self.send_command(pulse_width)

        if success:
            self.current_angle = angle

        return success

    def set_velocity(self, velocity: float) -> bool:
        """
        Set servo velocity for continuous rotation.

        Args:
            velocity: Target velocity in degrees/second

        Returns:
            True if command sent successfully
        """
        self.target_velocity = velocity
        pulse_width = self.velocity_to_pulse_width(velocity)

        success = self.send_command(pulse_width)

        if success:
            self.current_velocity = velocity

        return success

    def get_state(self) -> Dict[str, Any]:
        """Get current servo state."""
        return {
            'servo_id': self.servo_id,
            'current_angle': self.current_angle,
            'current_velocity': self.current_velocity,
            'target_angle': self.target_angle,
            'target_velocity': self.target_velocity,
            'is_connected': self.is_connected,
            'timestamp': time.time()
        }

class SnakeServoController:
    """
    Controller for complete robotic snake with multiple servos.
    """

    def __init__(self, num_joints: int = 10, serial_port: str = "/dev/ttyUSB0"):
        """
        Initialize snake servo controller.

        Args:
            num_joints: Number of snake joints/servos
            serial_port: Serial port for servo communication
        """
        self.num_joints = num_joints
        self.serial_port = serial_port

        # Servo controllers for each joint
        self.servos = [
            MG996RController(servo_id=i, serial_port=serial_port)
            for i in range(num_joints)
        ]

        # Control mode
        self.control_mode = ControlMode.TRAJECTORY

        # Trajectory tracking
        self.joint_angles = np.zeros(num_joints)
        self.joint_velocities = np.zeros(num_joints)

        # Synchronization
        self.control_thread = None
        self.is_controlling = False
        self.control_lock = threading.Lock()

        # Performance monitoring
        self.last_update_time = time.time()
        self.update_rate_history = []

        # Setup logging
        self.logger = logging.getLogger('SnakeServoController')

    def connect_all_servos(self) -> bool:
        """Connect to all servos."""
        success_count = 0

        for servo in self.servos:
            if servo.connect():
                success_count += 1
            time.sleep(0.1)  # Small delay between connections

        self.logger.info(f"Connected {success_count}/{self.num_joints} servos")
        return success_count == self.num_joints

    def disconnect_all_servos(self):
        """Disconnect all servos."""
        for servo in self.servos:
            servo.disconnect()

    def set_joint_angles(self, angles: Union[List[float], np.ndarray]) -> bool:
        """
        Set angles for all joints.

        Args:
            angles: Array of joint angles in degrees

        Returns:
            True if all commands sent successfully
        """
        if len(angles) != self.num_joints:
            self.logger.error(f"Expected {self.num_joints} angles, got {len(angles)}")
            return False

        success = True
        for i, angle in enumerate(angles):
            if not self.servos[i].set_angle(angle):
                success = False

        if success:
            self.joint_angles = np.array(angles)

        return success

    def set_joint_velocities(self, velocities: Union[List[float], np.ndarray]) -> bool:
        """
        Set velocities for all joints.

        Args:
            velocities: Array of joint velocities in degrees/second

        Returns:
            True if all commands sent successfully
        """
        if len(velocities) != self.num_joints:
            self.logger.error(f"Expected {self.num_joints} velocities, got {len(velocities)}")
            return False

        success = True
        for i, velocity in enumerate(velocities):
            if not self.servos[i].set_velocity(velocity):
                success = False

        if success:
            self.joint_velocities = np.array(velocities)

        return success

    def execute_trajectory(self, trajectory_data: Dict[str, Any],
                          time_scaling: float = 1.0) -> bool:
        """
        Execute trajectory using quintic polynomial or CPG data.

        Args:
            trajectory_data: Trajectory data from quintic polynomial or CPG
            time_scaling: Scaling factor for trajectory timing

        Returns:
            True if trajectory execution started successfully
        """
        try:
            if 'joints' not in trajectory_data:
                self.logger.error("Invalid trajectory data format")
                return False

            # Start trajectory execution thread
            self.is_controlling = True
            self.control_thread = threading.Thread(
                target=self._trajectory_execution_loop,
                args=(trajectory_data, time_scaling),
                daemon=True
            )
            self.control_thread.start()

            self.logger.info("Trajectory execution started")
            return True

        except Exception as e:
            self.logger.error(f"Error starting trajectory execution: {e}")
            return False

    def _trajectory_execution_loop(self, trajectory_data: Dict[str, Any],
                                 time_scaling: float):
        """Main loop for trajectory execution."""
        try:
            start_time = time.time()
            joint_keys = list(trajectory_data['joints'].keys())

            while self.is_controlling:
                current_time = time.time() - start_time
                scaled_time = current_time * time_scaling

                # Check if trajectory is complete
                if scaled_time >= trajectory_data.get('duration', 2.0):
                    break

                # Calculate joint angles for current time
                joint_angles = []
                joint_velocities = []

                for joint_key in joint_keys:
                    joint_traj = trajectory_data['joints'][joint_key]

                    if 'position' in joint_traj and 'velocity' in joint_traj:
                        # Interpolate position and velocity for current time
                        time_array = joint_traj['time']
                        position_array = joint_traj['position']
                        velocity_array = joint_traj['velocity']

                        # Simple linear interpolation
                        if scaled_time <= time_array[0]:
                            angle = position_array[0]
                            velocity = velocity_array[0]
                        elif scaled_time >= time_array[-1]:
                            angle = position_array[-1]
                            velocity = velocity_array[-1]
                        else:
                            # Find interpolation indices
                            idx = np.searchsorted(time_array, scaled_time) - 1
                            idx = max(0, min(idx, len(time_array) - 2))

                            t1, t2 = time_array[idx], time_array[idx + 1]
                            p1, p2 = position_array[idx], position_array[idx + 1]
                            v1, v2 = velocity_array[idx], velocity_array[idx + 1]

                            # Linear interpolation
                            t_ratio = (scaled_time - t1) / (t2 - t1)
                            angle = p1 + t_ratio * (p2 - p1)
                            velocity = v1 + t_ratio * (v2 - v1)

                        joint_angles.append(angle)
                        joint_velocities.append(velocity)

                # Send commands to servos
                if joint_angles and len(joint_angles) == self.num_joints:
                    self.set_joint_angles(joint_angles)
                    self.joint_velocities = np.array(joint_velocities)

                # Control update rate
                time.sleep(0.02)  # 50 Hz update rate

        except Exception as e:
            self.logger.error(f"Error in trajectory execution loop: {e}")
        finally:
            self.is_controlling = False

    def stop_motion(self):
        """Stop all servo motion."""
        try:
            self.is_controlling = False

            if self.control_thread and self.control_thread.is_alive:
                self.control_thread.join(timeout=1.0)

            # Set all servos to neutral position
            neutral_angles = [0.0] * self.num_joints
            self.set_joint_angles(neutral_angles)

            self.logger.info("Motion stopped")

        except Exception as e:
            self.logger.error(f"Error stopping motion: {e}")

    def get_snake_state(self) -> Dict[str, Any]:
        """Get complete snake state."""
        try:
            servo_states = []
            for servo in self.servos:
                servo_states.append(servo.get_state())

            return {
                'num_joints': self.num_joints,
                'control_mode': self.control_mode.value,
                'is_controlling': self.is_controlling,
                'joint_angles': self.joint_angles.tolist(),
                'joint_velocities': self.joint_velocities.tolist(),
                'servo_states': servo_states,
                'timestamp': time.time()
            }

        except Exception as e:
            self.logger.error(f"Error getting snake state: {e}")
            return {}

    def calibrate_servos(self) -> bool:
        """Calibrate all servos to find neutral positions."""
        try:
            self.logger.info("Starting servo calibration...")

            # Move each servo through its range to find center
            for i, servo in enumerate(self.servos):
                self.logger.info(f"Calibrating servo {i}...")

                # Move to minimum angle
                servo.set_angle(servo.angle_min)
                time.sleep(0.5)

                # Move to maximum angle
                servo.set_angle(servo.angle_max)
                time.sleep(0.5)

                # Move to center
                servo.set_angle(0.0)
                time.sleep(0.5)

            self.logger.info("Servo calibration completed")
            return True

        except Exception as e:
            self.logger.error(f"Error during servo calibration: {e}")
            return False

    def test_servo_range(self) -> Dict[str, List[float]]:
        """Test servo range and responsiveness."""
        try:
            self.logger.info("Testing servo range...")

            test_angles = [-90, -45, 0, 45, 90]
            response_times = []

            for i, servo in enumerate(self.servos):
                servo_times = []

                for angle in test_angles:
                    start_time = time.time()
                    servo.set_angle(angle)
                    end_time = time.time()

                    servo_times.append(end_time - start_time)

                response_times.append(servo_times)
                time.sleep(0.1)

            return {
                'test_angles': test_angles,
                'response_times': response_times
            }

        except Exception as e:
            self.logger.error(f"Error testing servo range: {e}")
            return {}

    def save_servo_configuration(self, filename: str):
        """Save servo configuration to file."""
        try:
            config = {
                'num_joints': self.num_joints,
                'serial_port': self.serial_port,
                'control_mode': self.control_mode.value,
                'servo_configs': []
            }

            for servo in self.servos:
                servo_config = {
                    'servo_id': servo.servo_id,
                    'angle_min': servo.angle_min,
                    'angle_max': servo.angle_max,
                    'velocity_max': servo.velocity_max,
                    'kp_position': servo.kp_position,
                    'ki_position': servo.ki_position,
                    'kd_position': servo.kd_position,
                    'kp_velocity': servo.kp_velocity,
                    'ki_velocity': servo.ki_velocity
                }
                config['servo_configs'].append(servo_config)

            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)

            self.logger.info(f"Servo configuration saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error saving servo configuration: {e}")

    def load_servo_configuration(self, filename: str) -> bool:
        """Load servo configuration from file."""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)

            # Restore configuration
            self.num_joints = config['num_joints']
            self.serial_port = config['serial_port']
            self.control_mode = ControlMode(config['control_mode'])

            # Restore servo configurations
            for i, servo_config in enumerate(config['servo_configs']):
                if i < len(self.servos):
                    servo = self.servos[i]
                    servo.angle_min = servo_config['angle_min']
                    servo.angle_max = servo_config['angle_max']
                    servo.velocity_max = servo_config['velocity_max']
                    servo.kp_position = servo_config['kp_position']
                    servo.ki_position = servo_config['ki_position']
                    servo.kd_position = servo_config['kd_position']
                    servo.kp_velocity = servo_config['kp_velocity']
                    servo.ki_velocity = servo_config['ki_velocity']

            self.logger.info(f"Servo configuration loaded from {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading servo configuration: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Create snake servo controller for 10 joints
    snake_controller = SnakeServoController(num_joints=10)

    try:
        # Connect to servos
        if snake_controller.connect_all_servos():
            print("All servos connected successfully")

            # Calibrate servos
            snake_controller.calibrate_servos()

            # Test servo range
            range_test = snake_controller.test_servo_range()
            print(f"Servo range test completed: {range_test}")

            # Example: Set all joints to wave pattern
            wave_angles = [30 * np.sin(2 * np.pi * i / 10) for i in range(10)]
            snake_controller.set_joint_angles(wave_angles)
            print("Wave pattern sent to servos")

            # Wait a moment
            time.sleep(2.0)

            # Stop motion
            snake_controller.stop_motion()
            print("Motion stopped")

        else:
            print("Failed to connect to all servos")

    finally:
        # Clean up
        snake_controller.disconnect_all_servos()
        print("Servos disconnected")