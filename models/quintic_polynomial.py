"""
Quintic Polynomial Trajectory Generator for Robotic Snake
Generates smooth trajectories for snake-like movement using quintic polynomials.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum

class TrajectoryType(Enum):
    """Types of trajectories for snake movement."""
    STRAIGHT = "straight"
    CIRCULAR = "circular"
    SERPENTINE = "serpentine"
    SPIRAL = "spiral"
    CUSTOM = "custom"

@dataclass
class TrajectoryConstraints:
    """Constraints for trajectory generation."""
    start_position: np.ndarray
    end_position: np.ndarray
    start_velocity: np.ndarray
    end_velocity: np.ndarray
    start_acceleration: np.ndarray
    end_acceleration: np.ndarray
    time_duration: float

class QuinticPolynomialTrajectory:
    """
    Quintic polynomial trajectory generator for smooth snake movement.
    Uses 6th order polynomials to ensure smooth position, velocity, and acceleration.
    """

    def __init__(self, num_joints: int = 10):
        """
        Initialize the trajectory generator.

        Args:
            num_joints: Number of snake joints/servos
        """
        self.num_joints = num_joints
        self.coefficients = {}
        self.trajectory_cache = {}

        # Setup logging
        self.logger = logging.getLogger('QuinticTrajectory')

        # Physical constraints
        self.max_joint_angle = np.pi  # radians
        self.max_angular_velocity = 2.0 * np.pi  # rad/s
        self.max_angular_acceleration = 10.0 * np.pi  # rad/s²

    def generate_polynomial_coefficients(self, constraints: TrajectoryConstraints) -> np.ndarray:
        """
        Generate coefficients for quintic polynomial trajectory.

        Args:
            constraints: Trajectory constraints

        Returns:
            Array of polynomial coefficients [a5, a4, a3, a2, a1, a0]
        """
        try:
            # Quintic polynomial: x(t) = a5*t^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0
            t = constraints.time_duration

            # Boundary conditions
            x0, x_dot0, x_ddot0 = constraints.start_position, constraints.start_velocity, constraints.start_acceleration
            xf, x_dotf, x_ddotf = constraints.end_position, constraints.end_velocity, constraints.end_acceleration

            # Solve for coefficients using boundary conditions
            A = np.array([
                [1, 0, 0, 0, 0, 0],                    # x(0) = x0
                [0, 1, 0, 0, 0, 0],                    # x'(0) = x_dot0
                [0, 0, 2, 0, 0, 0],                    # x''(0) = x_ddot0
                [t**5, t**4, t**3, t**2, t, 1],        # x(t) = xf
                [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0],   # x'(t) = x_dotf
                [20*t**3, 12*t**2, 6*t, 2, 0, 0]      # x''(t) = x_ddotf
            ])

            b = np.array([x0, x_dot0, x_ddot0, xf, x_dotf, x_ddotf])

            # Solve linear system
            coefficients = np.linalg.solve(A, b)

            return coefficients

        except Exception as e:
            self.logger.error(f"Error generating polynomial coefficients: {e}")
            return np.zeros(6)

    def generate_joint_trajectory(self, joint_index: int, trajectory_type: TrajectoryType,
                                duration: float, amplitude: float = 1.0,
                                frequency: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate trajectory for a specific joint.

        Args:
            joint_index: Index of the joint (0-9 for 10-joint snake)
            trajectory_type: Type of trajectory to generate
            duration: Duration of trajectory in seconds
            amplitude: Amplitude of motion
            frequency: Frequency of oscillation

        Returns:
            Dictionary containing position, velocity, and acceleration trajectories
        """
        try:
            # Time array
            t = np.linspace(0, duration, int(duration * 100))  # 100 Hz sampling

            # Generate base trajectory based on type
            if trajectory_type == TrajectoryType.SERPENTINE:
                # Serpentine wave pattern
                phase_offset = (2 * np.pi * joint_index) / self.num_joints
                base_position = amplitude * np.sin(2 * np.pi * frequency * t + phase_offset)

            elif trajectory_type == TrajectoryType.STRAIGHT:
                # Straight line movement
                base_position = np.linspace(0, amplitude, len(t))

            elif trajectory_type == TrajectoryType.CIRCULAR:
                # Circular motion pattern
                phase_offset = (2 * np.pi * joint_index) / self.num_joints
                base_position = amplitude * np.sin(2 * np.pi * frequency * t + phase_offset)

            elif trajectory_type == TrajectoryType.SPIRAL:
                # Spiral pattern
                radius = amplitude * (1 - t/duration)  # Decreasing radius
                angle = 2 * np.pi * frequency * t + (2 * np.pi * joint_index) / self.num_joints
                base_position = radius * np.sin(angle)

            else:
                # Default to sinusoidal
                base_position = amplitude * np.sin(2 * np.pi * frequency * t)

            # Apply physical constraints
            base_position = np.clip(base_position, -self.max_joint_angle, self.max_joint_angle)

            # Calculate velocity and acceleration using finite differences
            dt = t[1] - t[0]
            velocity = np.gradient(base_position, dt)
            acceleration = np.gradient(velocity, dt)

            # Apply velocity and acceleration constraints
            velocity = np.clip(velocity, -self.max_angular_velocity, self.max_angular_velocity)
            acceleration = np.clip(acceleration, -self.max_angular_acceleration, self.max_angular_acceleration)

            return {
                'time': t,
                'position': base_position,
                'velocity': velocity,
                'acceleration': acceleration,
                'joint_index': joint_index,
                'trajectory_type': trajectory_type.value
            }

        except Exception as e:
            self.logger.error(f"Error generating joint trajectory: {e}")
            return {}

    def generate_snake_trajectory(self, trajectory_type: TrajectoryType,
                                duration: float, amplitude: float = 1.0,
                                frequency: float = 1.0) -> Dict[str, Any]:
        """
        Generate complete trajectory for entire snake.

        Args:
            trajectory_type: Type of trajectory for the snake
            duration: Duration in seconds
            amplitude: Motion amplitude
            frequency: Motion frequency

        Returns:
            Complete snake trajectory data
        """
        try:
            snake_trajectory = {
                'trajectory_type': trajectory_type.value,
                'duration': duration,
                'amplitude': amplitude,
                'frequency': frequency,
                'joints': {}
            }

            # Generate trajectory for each joint
            for joint_idx in range(self.num_joints):
                joint_traj = self.generate_joint_trajectory(
                    joint_idx, trajectory_type, duration, amplitude, frequency
                )
                snake_trajectory['joints'][f'joint_{joint_idx}'] = joint_traj

            # Calculate snake-level properties
            snake_trajectory.update(self._calculate_snake_properties(snake_trajectory))

            return snake_trajectory

        except Exception as e:
            self.logger.error(f"Error generating snake trajectory: {e}")
            return {}

    def _calculate_snake_properties(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall snake movement properties."""
        try:
            # Extract positions from all joints
            positions = []
            for joint_key, joint_traj in trajectory['joints'].items():
                if 'position' in joint_traj:
                    positions.append(joint_traj['position'])

            if not positions:
                return {}

            positions = np.array(positions)

            # Calculate snake-level metrics
            center_of_mass = np.mean(positions, axis=0)
            total_displacement = np.linalg.norm(positions[-1] - positions[0])

            # Calculate average velocity and acceleration
            avg_velocity = np.mean([joint_traj['velocity'] for joint_traj in trajectory['joints'].values()], axis=0)
            avg_acceleration = np.mean([joint_traj['acceleration'] for joint_traj in trajectory['joints'].values()], axis=0)

            return {
                'center_of_mass': center_of_mass,
                'total_displacement': total_displacement,
                'average_velocity': avg_velocity,
                'average_acceleration': avg_acceleration,
                'efficiency_metric': self._calculate_efficiency_metric(trajectory)
            }

        except Exception as e:
            self.logger.error(f"Error calculating snake properties: {e}")
            return {}

    def _calculate_efficiency_metric(self, trajectory: Dict[str, Any]) -> float:
        """Calculate efficiency metric for the trajectory."""
        try:
            # Simple efficiency metric based on smoothness vs displacement
            total_displacement = 0
            total_acceleration = 0

            for joint_traj in trajectory['joints'].values():
                if 'position' in joint_traj and 'acceleration' in joint_traj:
                    displacement = np.linalg.norm(joint_traj['position'][-1] - joint_traj['position'][0])
                    avg_acceleration = np.mean(np.abs(joint_traj['acceleration']))

                    total_displacement += displacement
                    total_acceleration += avg_acceleration

            if total_acceleration == 0:
                return 0.0

            # Efficiency is displacement per unit acceleration (higher is better)
            return total_displacement / total_acceleration

        except Exception as e:
            self.logger.error(f"Error calculating efficiency metric: {e}")
            return 0.0

    def optimize_trajectory(self, trajectory: Dict[str, Any],
                          optimization_target: str = "efficiency") -> Dict[str, Any]:
        """
        Optimize trajectory parameters for better performance.

        Args:
            trajectory: Input trajectory to optimize
            optimization_target: Optimization target ("efficiency", "speed", "smoothness")

        Returns:
            Optimized trajectory
        """
        try:
            # Simple optimization through parameter search
            best_trajectory = trajectory.copy()
            best_score = 0

            # Test different amplitudes and frequencies
            for amp_factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
                for freq_factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    test_trajectory = self.generate_snake_trajectory(
                        TrajectoryType(trajectory['trajectory_type']),
                        trajectory['duration'],
                        trajectory['amplitude'] * amp_factor,
                        trajectory['frequency'] * freq_factor
                    )

                    if optimization_target == "efficiency":
                        score = test_trajectory.get('efficiency_metric', 0)
                    elif optimization_target == "speed":
                        score = np.mean([np.mean(np.abs(joint['velocity']))
                                       for joint in test_trajectory['joints'].values()])
                    else:  # smoothness
                        score = 1.0 / (1.0 + np.mean([np.mean(np.abs(joint['acceleration']))
                                                     for joint in test_trajectory['joints'].values()]))

                    if score > best_score:
                        best_score = score
                        best_trajectory = test_trajectory

            return best_trajectory

        except Exception as e:
            self.logger.error(f"Error optimizing trajectory: {e}")
            return trajectory

    def save_trajectory(self, trajectory: Dict[str, Any], filename: str):
        """Save trajectory to file."""
        try:
            np.savez(filename, **trajectory)
            self.logger.info(f"Trajectory saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving trajectory: {e}")

    def load_trajectory(self, filename: str) -> Dict[str, Any]:
        """Load trajectory from file."""
        try:
            data = np.load(filename, allow_pickle=True)
            trajectory = {key: data[key].item() if hasattr(data[key], 'item') else data[key]
                         for key in data.keys()}
            self.logger.info(f"Trajectory loaded from {filename}")
            return trajectory
        except Exception as e:
            self.logger.error(f"Error loading trajectory: {e}")
            return {}

    def visualize_trajectory(self, trajectory: Dict[str, Any], save_plot: bool = False):
        """Visualize the generated trajectory."""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            # Plot positions
            axes[0].set_title('Joint Positions')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Position (rad)')

            # Plot velocities
            axes[1].set_title('Joint Velocities')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Velocity (rad/s)')

            # Plot accelerations
            axes[2].set_title('Joint Accelerations')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Acceleration (rad/s²)')

            # Plot data for each joint
            for joint_key, joint_traj in trajectory['joints'].items():
                if 'position' in joint_traj and 'velocity' in joint_traj and 'acceleration' in joint_traj:
                    t = joint_traj['time']
                    axes[0].plot(t, joint_traj['position'], label=joint_key, alpha=0.7)
                    axes[1].plot(t, joint_traj['velocity'], label=joint_key, alpha=0.7)
                    axes[2].plot(t, joint_traj['acceleration'], label=joint_key, alpha=0.7)

            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True)
            axes[1].grid(True)
            axes[2].grid(True)

            plt.tight_layout()

            if save_plot:
                plt.savefig('trajectory_visualization.png', dpi=300, bbox_inches='tight')
                self.logger.info("Trajectory visualization saved")

            plt.show()

        except Exception as e:
            self.logger.error(f"Error visualizing trajectory: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Create trajectory generator for 10-joint snake
    generator = QuinticPolynomialTrajectory(num_joints=10)

    # Generate serpentine trajectory
    trajectory = generator.generate_snake_trajectory(
        TrajectoryType.SERPENTINE,
        duration=2.0,
        amplitude=1.0,
        frequency=1.0
    )

    # Visualize trajectory
    generator.visualize_trajectory(trajectory, save_plot=True)

    # Optimize trajectory
    optimized_trajectory = generator.optimize_trajectory(trajectory, "efficiency")

    print(f"Original efficiency: {trajectory.get('efficiency_metric', 0):.3f}")
    print(f"Optimized efficiency: {optimized_trajectory.get('efficiency_metric', 0):.3f}")