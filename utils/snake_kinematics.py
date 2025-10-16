"""
Snake Kinematics and Dynamics Utilities
Mathematical utilities for robotic snake kinematics, dynamics, and motion analysis.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math

@dataclass
class JointState:
    """State of a single snake joint."""
    angle: float          # Current angle in radians
    velocity: float       # Current angular velocity in rad/s
    acceleration: float   # Current angular acceleration in rad/s²
    torque: float         # Current torque in Nm

@dataclass
class SnakeConfiguration:
    """Configuration of the entire snake."""
    joint_angles: List[float]      # Angles of all joints in radians
    joint_positions: List[Tuple[float, float, float]]  # 3D positions of each joint
    link_length: float            # Length of each link in meters

class SnakeKinematics:
    """
    Kinematic calculations for robotic snake.
    """

    def __init__(self, num_joints: int = 10, link_length: float = 0.08):
        """
        Initialize snake kinematics.

        Args:
            num_joints: Number of snake joints
            link_length: Length of each link in meters
        """
        self.num_joints = num_joints
        self.link_length = link_length

        # Setup logging
        self.logger = logging.getLogger('SnakeKinematics')

    def forward_kinematics(self, joint_angles: List[float]) -> List[Tuple[float, float, float]]:
        """
        Calculate forward kinematics for the snake.

        Args:
            joint_angles: List of joint angles in radians

        Returns:
            List of 3D positions for each joint
        """
        try:
            if len(joint_angles) != self.num_joints:
                raise ValueError(f"Expected {self.num_joints} angles, got {len(joint_angles)}")

            positions = [(0.0, 0.0, 0.0)]  # Base position

            for i in range(self.num_joints):
                # Get current position and orientation
                current_pos = positions[-1]
                current_angle = joint_angles[i]

                # Calculate next joint position
                # Assuming planar movement for simplicity (x-y plane)
                dx = self.link_length * np.cos(current_angle)
                dy = self.link_length * np.sin(current_angle)
                dz = 0.0  # No vertical movement for now

                next_pos = (
                    current_pos[0] + dx,
                    current_pos[1] + dy,
                    current_pos[2] + dz
                )

                positions.append(next_pos)

            return positions

        except Exception as e:
            self.logger.error(f"Error in forward kinematics: {e}")
            return [(0.0, 0.0, 0.0)] * (self.num_joints + 1)

    def inverse_kinematics(self, target_positions: List[Tuple[float, float, float]],
                          max_iterations: int = 100) -> List[float]:
        """
        Calculate inverse kinematics using Jacobian pseudoinverse method.

        Args:
            target_positions: Target positions for each joint
            max_iterations: Maximum iterations for convergence

        Returns:
            Joint angles that achieve target positions
        """
        try:
            # Initialize with current angles (or zeros)
            joint_angles = [0.0] * self.num_joints

            for iteration in range(max_iterations):
                # Forward kinematics
                current_positions = self.forward_kinematics(joint_angles)

                # Calculate position errors
                position_errors = []
                for i, (current, target) in enumerate(zip(current_positions[1:], target_positions)):
                    error = np.array(target) - np.array(current)
                    position_errors.extend(error)

                # Check convergence
                max_error = max(abs(error) for error in position_errors)
                if max_error < 1e-6:
                    break

                # Calculate Jacobian
                jacobian = self._calculate_jacobian(joint_angles)

                # Pseudoinverse solution
                if len(position_errors) == jacobian.shape[0]:
                    delta_angles = np.linalg.pinv(jacobian) @ position_errors
                    joint_angles = [a + da for a, da in zip(joint_angles, delta_angles)]

            return joint_angles

        except Exception as e:
            self.logger.error(f"Error in inverse kinematics: {e}")
            return [0.0] * self.num_joints

    def _calculate_jacobian(self, joint_angles: List[float]) -> np.ndarray:
        """Calculate Jacobian matrix for current joint configuration."""
        try:
            # Numerical Jacobian calculation
            jacobian = []
            epsilon = 1e-6

            # Base positions
            base_positions = self.forward_kinematics(joint_angles)

            for j in range(self.num_joints):
                # Perturb joint j
                angles_plus = joint_angles.copy()
                angles_minus = joint_angles.copy()
                angles_plus[j] += epsilon
                angles_minus[j] -= epsilon

                # Calculate positions
                positions_plus = self.forward_kinematics(angles_plus)
                positions_minus = self.forward_kinematics(angles_minus)

                # Calculate numerical derivatives
                derivatives = []
                for i in range(self.num_joints):
                    dx = (positions_plus[i+1][0] - positions_minus[i+1][0]) / (2 * epsilon)
                    dy = (positions_plus[i+1][1] - positions_minus[i+1][1]) / (2 * epsilon)
                    dz = (positions_plus[i+1][2] - positions_minus[i+1][2]) / (2 * epsilon)
                    derivatives.extend([dx, dy, dz])

                jacobian.append(derivatives)

            return np.array(jacobian)

        except Exception as e:
            self.logger.error(f"Error calculating Jacobian: {e}")
            return np.zeros((self.num_joints, 3 * self.num_joints))

    def calculate_center_of_mass(self, joint_angles: List[float],
                               joint_masses: Optional[List[float]] = None) -> Tuple[float, float, float]:
        """
        Calculate center of mass of the snake.

        Args:
            joint_angles: Current joint angles
            joint_masses: Mass of each joint (defaults to uniform)

        Returns:
            Center of mass position (x, y, z)
        """
        try:
            if joint_masses is None:
                joint_masses = [1.0] * self.num_joints

            positions = self.forward_kinematics(joint_angles)

            # Calculate weighted average
            total_mass = sum(joint_masses)
            com_x = sum(pos[0] * mass for pos, mass in zip(positions, joint_masses)) / total_mass
            com_y = sum(pos[1] * mass for pos, mass in zip(positions, joint_masses)) / total_mass
            com_z = sum(pos[2] * mass for pos, mass in zip(positions, joint_masses)) / total_mass

            return (com_x, com_y, com_z)

        except Exception as e:
            self.logger.error(f"Error calculating center of mass: {e}")
            return (0.0, 0.0, 0.0)

    def calculate_workspace(self, joint_angles: List[float]) -> Dict[str, Any]:
        """
        Calculate snake workspace and reachability.

        Args:
            joint_angles: Current joint configuration

        Returns:
            Dictionary containing workspace information
        """
        try:
            positions = self.forward_kinematics(joint_angles)

            # Extract coordinates
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            z_coords = [pos[2] for pos in positions]

            workspace = {
                'min_bounds': (min(x_coords), min(y_coords), min(z_coords)),
                'max_bounds': (max(x_coords), max(y_coords), max(z_coords)),
                'center': (
                    (min(x_coords) + max(x_coords)) / 2,
                    (min(y_coords) + max(y_coords)) / 2,
                    (min(z_coords) + max(z_coords)) / 2
                ),
                'volume': (
                    (max(x_coords) - min(x_coords)) *
                    (max(y_coords) - min(y_coords)) *
                    (max(z_coords) - min(z_coords))
                ),
                'joint_positions': positions
            }

            return workspace

        except Exception as e:
            self.logger.error(f"Error calculating workspace: {e}")
            return {}

class SnakeDynamics:
    """
    Dynamic calculations for robotic snake.
    """

    def __init__(self, num_joints: int = 10, link_length: float = 0.08, joint_mass: float = 0.05):
        """
        Initialize snake dynamics.

        Args:
            num_joints: Number of snake joints
            link_length: Length of each link in meters
            joint_mass: Mass of each joint in kg
        """
        self.num_joints = num_joints
        self.link_length = link_length
        self.joint_mass = joint_mass

        # Physical constants
        self.gravity = 9.81  # m/s²

        # Dynamic parameters (can be estimated or measured)
        self.joint_inertia = 0.001  # kg*m²
        self.friction_coefficient = 0.3

        # Setup logging
        self.logger = logging.getLogger('SnakeDynamics')

    def calculate_joint_torques(self, joint_angles: List[float],
                              joint_velocities: List[float],
                              joint_accelerations: List[float]) -> List[float]:
        """
        Calculate required joint torques using inverse dynamics.

        Args:
            joint_angles: Joint angles in radians
            joint_velocities: Joint angular velocities in rad/s
            joint_accelerations: Joint angular accelerations in rad/s²

        Returns:
            Required torques for each joint in Nm
        """
        try:
            torques = []

            for i in range(self.num_joints):
                # Simplified torque calculation
                # In practice, this would use full Newton-Euler or Lagrange formulation

                # Inertia torque
                inertia_torque = self.joint_inertia * joint_accelerations[i]

                # Coriolis and centrifugal torques (simplified)
                coriolis_torque = 0.0
                for j in range(self.num_joints):
                    if j != i:
                        coriolis_torque += joint_velocities[j] * joint_velocities[i] * 0.01

                # Gravity torque (simplified)
                gravity_torque = self.joint_mass * self.gravity * self.link_length * np.sin(joint_angles[i])

                # Friction torque
                friction_torque = self.friction_coefficient * joint_velocities[i]

                total_torque = inertia_torque + coriolis_torque + gravity_torque + friction_torque
                torques.append(total_torque)

            return torques

        except Exception as e:
            self.logger.error(f"Error calculating joint torques: {e}")
            return [0.0] * self.num_joints

    def calculate_energy_consumption(self, joint_angles: List[float],
                                   joint_velocities: List[float],
                                   joint_torques: List[float],
                                   time_duration: float) -> Dict[str, float]:
        """
        Calculate energy consumption of the snake.

        Args:
            joint_angles: Joint angles over time
            joint_velocities: Joint velocities over time
            joint_torques: Joint torques over time
            time_duration: Duration of movement in seconds

        Returns:
            Dictionary containing energy metrics
        """
        try:
            # Kinetic energy
            kinetic_energy = 0.0
            for i in range(self.num_joints):
                kinetic_energy += 0.5 * self.joint_inertia * joint_velocities[i]**2

            # Potential energy (simplified)
            potential_energy = 0.0
            for i in range(self.num_joints):
                height = self.link_length * (1 - np.cos(joint_angles[i]))  # Approximate height
                potential_energy += self.joint_mass * self.gravity * height

            # Mechanical power
            mechanical_power = 0.0
            for i in range(self.num_joints):
                mechanical_power += abs(joint_torques[i] * joint_velocities[i])

            # Average power consumption
            avg_power = mechanical_power / self.num_joints if self.num_joints > 0 else 0.0

            # Efficiency (simplified)
            efficiency = kinetic_energy / (kinetic_energy + potential_energy + 0.001)

            return {
                'kinetic_energy': kinetic_energy,
                'potential_energy': potential_energy,
                'mechanical_power': mechanical_power,
                'average_power': avg_power,
                'efficiency': efficiency,
                'total_energy': kinetic_energy + potential_energy
            }

        except Exception as e:
            self.logger.error(f"Error calculating energy consumption: {e}")
            return {}

    def calculate_stability_margin(self, joint_angles: List[float],
                                 contact_points: Optional[List[Tuple[float, float, float]]] = None) -> float:
        """
        Calculate stability margin of the snake configuration.

        Args:
            joint_angles: Current joint angles
            contact_points: Points of contact with ground (optional)

        Returns:
            Stability margin (higher is more stable)
        """
        try:
            if contact_points is None:
                # Assume all joints are potential contact points
                positions = self.forward_kinematics(joint_angles)
                contact_points = positions

            # Calculate center of mass
            com = self.calculate_center_of_mass(joint_angles)

            # Calculate support polygon (simplified as convex hull of contact points)
            if len(contact_points) < 3:
                return 0.0  # Unstable if fewer than 3 contact points

            # Simple stability metric: distance from COM to nearest edge of support polygon
            min_distance = float('inf')
            for i in range(len(contact_points)):
                p1 = np.array(contact_points[i][:2])  # Use x,y coordinates
                p2 = np.array(contact_points[(i + 1) % len(contact_points)][:2])

                # Distance from point to line
                distance = self._point_to_line_distance(com[:2], p1, p2)
                min_distance = min(min_distance, distance)

            return min_distance

        except Exception as e:
            self.logger.error(f"Error calculating stability margin: {e}")
            return 0.0

    def _point_to_line_distance(self, point: Tuple[float, float],
                              line_p1: np.ndarray, line_p2: np.ndarray) -> float:
        """Calculate distance from point to line segment."""
        try:
            # Vector calculations
            line_vec = line_p2 - line_p1
            point_vec = np.array(point) - line_p1

            line_length = np.linalg.norm(line_vec)

            if line_length == 0:
                return np.linalg.norm(point_vec)

            # Projection of point onto line
            projection = np.dot(point_vec, line_vec) / line_length
            projection = max(0, min(line_length, projection))

            closest_point = line_p1 + (projection / line_length) * line_vec

            return np.linalg.norm(np.array(point) - closest_point)

        except Exception as e:
            self.logger.error(f"Error calculating point to line distance: {e}")
            return float('inf')

    def calculate_center_of_mass(self, joint_angles: List[float]) -> Tuple[float, float, float]:
        """Calculate center of mass using kinematics."""
        try:
            kinematics = SnakeKinematics(self.num_joints, self.link_length)
            return kinematics.calculate_center_of_mass(joint_angles)
        except Exception as e:
            self.logger.error(f"Error calculating center of mass: {e}")
            return (0.0, 0.0, 0.0)

class MotionAnalysis:
    """
    Analysis tools for snake motion patterns.
    """

    def __init__(self):
        """Initialize motion analysis."""
        self.logger = logging.getLogger('MotionAnalysis')

    def analyze_gait_efficiency(self, trajectory_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze efficiency of a gait pattern.

        Args:
            trajectory_data: Trajectory data from CPG or quintic generator

        Returns:
            Dictionary containing efficiency metrics
        """
        try:
            if 'joints' not in trajectory_data:
                return {}

            efficiency_metrics = {
                'smoothness': 0.0,
                'energy_efficiency': 0.0,
                'stability': 0.0,
                'overall_score': 0.0
            }

            # Calculate smoothness (inverse of jerk)
            total_jerk = 0.0
            total_displacement = 0.0

            for joint_key, joint_traj in trajectory_data['joints'].items():
                if 'acceleration' in joint_traj and 'position' in joint_traj:
                    # Calculate jerk (derivative of acceleration)
                    time_array = joint_traj['time']
                    acc_array = joint_traj['acceleration']

                    if len(acc_array) > 2:
                        jerk = np.gradient(acc_array, time_array)
                        total_jerk += np.sum(np.abs(jerk))

                    # Calculate displacement
                    displacement = np.linalg.norm(
                        joint_traj['position'][-1] - joint_traj['position'][0]
                    )
                    total_displacement += displacement

            # Smoothness score (higher is better)
            if total_jerk > 0:
                efficiency_metrics['smoothness'] = total_displacement / total_jerk

            # Energy efficiency (simplified)
            efficiency_metrics['energy_efficiency'] = efficiency_metrics['smoothness'] / 100.0

            # Stability (based on acceleration variance)
            accelerations = []
            for joint_traj in trajectory_data['joints'].values():
                if 'acceleration' in joint_traj:
                    accelerations.extend(joint_traj['acceleration'])

            if accelerations:
                acc_variance = np.var(accelerations)
                efficiency_metrics['stability'] = 1.0 / (1.0 + acc_variance)

            # Overall score
            weights = {'smoothness': 0.4, 'energy_efficiency': 0.3, 'stability': 0.3}
            efficiency_metrics['overall_score'] = sum(
                efficiency_metrics[key] * weights[key]
                for key in weights
            )

            return efficiency_metrics

        except Exception as e:
            self.logger.error(f"Error analyzing gait efficiency: {e}")
            return {}

    def detect_motion_patterns(self, joint_angles_history: List[List[float]]) -> Dict[str, Any]:
        """
        Detect motion patterns in joint angle data.

        Args:
            joint_angles_history: History of joint angle configurations

        Returns:
            Dictionary containing detected patterns
        """
        try:
            if not joint_angles_history:
                return {}

            # Convert to numpy array for analysis
            angles_array = np.array(joint_angles_history)

            # Calculate frequency content using FFT
            fft_result = np.fft.fft(angles_array, axis=0)
            frequencies = np.fft.fftfreq(angles_array.shape[0])

            # Find dominant frequencies for each joint
            dominant_frequencies = []
            for joint_idx in range(angles_array.shape[1]):
                joint_fft = fft_result[:, joint_idx]
                dominant_idx = np.argmax(np.abs(joint_fft[1:])) + 1  # Skip DC component
                dominant_freq = abs(frequencies[dominant_idx])
                dominant_frequencies.append(dominant_freq)

            # Detect wave patterns
            wave_analysis = self._analyze_wave_patterns(angles_array)

            return {
                'dominant_frequencies': dominant_frequencies,
                'wave_analysis': wave_analysis,
                'motion_type': self._classify_motion_type(dominant_frequencies, wave_analysis),
                'synchronization': self._analyze_synchronization(angles_array)
            }

        except Exception as e:
            self.logger.error(f"Error detecting motion patterns: {e}")
            return {}

    def _analyze_wave_patterns(self, angles_array: np.ndarray) -> Dict[str, Any]:
        """Analyze wave patterns in joint angles."""
        try:
            # Calculate phase relationships between joints
            phases = []
            amplitudes = []

            for joint_idx in range(angles_array.shape[1]):
                joint_angles = angles_array[:, joint_idx]

                # Simple phase estimation using Hilbert transform
                from scipy.signal import hilbert
                analytic_signal = hilbert(joint_angles)
                phase = np.angle(analytic_signal)
                phases.append(phase)

                # Amplitude estimation
                amplitude = np.std(joint_angles)
                amplitudes.append(amplitude)

            # Calculate phase differences between adjacent joints
            phase_differences = []
            for i in range(len(phases) - 1):
                phase_diff = phases[i+1] - phases[i]
                phase_differences.append(np.mean(phase_diff))

            return {
                'phases': phases,
                'amplitudes': amplitudes,
                'phase_differences': phase_differences,
                'wave_number': self._estimate_wave_number(phase_differences)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing wave patterns: {e}")
            return {}

    def _estimate_wave_number(self, phase_differences: List[float]) -> float:
        """Estimate number of waves along snake body."""
        try:
            if not phase_differences:
                return 1.0

            # Average phase difference
            avg_phase_diff = np.mean(phase_differences)

            # Wave number is 2π divided by average phase difference
            wave_number = (2 * np.pi) / abs(avg_phase_diff) if avg_phase_diff != 0 else 1.0

            return wave_number

        except Exception as e:
            self.logger.error(f"Error estimating wave number: {e}")
            return 1.0

    def _classify_motion_type(self, frequencies: List[float], wave_analysis: Dict[str, Any]) -> str:
        """Classify the type of motion pattern."""
        try:
            # Simple classification based on frequency and wave patterns
            avg_frequency = np.mean(frequencies)

            if avg_frequency < 0.5:
                return "slow_creeping"
            elif avg_frequency > 2.0:
                return "fast_vibration"
            else:
                wave_number = wave_analysis.get('wave_number', 1.0)
                if wave_number > 1.5:
                    return "serpentine"
                else:
                    return "undulation"

        except Exception as e:
            self.logger.error(f"Error classifying motion type: {e}")
            return "unknown"

    def _analyze_synchronization(self, angles_array: np.ndarray) -> float:
        """Analyze synchronization between joints."""
        try:
            # Calculate correlation between adjacent joints
            correlations = []

            for i in range(angles_array.shape[1] - 1):
                corr = np.corrcoef(angles_array[:, i], angles_array[:, i+1])[0, 1]
                correlations.append(abs(corr))

            # Average correlation as synchronization measure
            avg_correlation = np.mean(correlations) if correlations else 0.0

            return avg_correlation

        except Exception as e:
            self.logger.error(f"Error analyzing synchronization: {e}")
            return 0.0

# Utility functions
def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return np.deg2rad(degrees)

def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return np.rad2deg(radians)

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π] range."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_distance_3d(point1: Tuple[float, float, float],
                         point2: Tuple[float, float, float]) -> float:
    """Calculate 3D distance between two points."""
    return np.sqrt(sum((a - b)**2 for a, b in zip(point1, point2)))

# Example usage
if __name__ == "__main__":
    # Example kinematic calculations
    kinematics = SnakeKinematics(num_joints=10, link_length=0.08)

    # Example joint angles (serpentine pattern)
    joint_angles = [30 * np.sin(2 * np.pi * i / 10) for i in range(10)]
    joint_angles = [degrees_to_radians(angle) for angle in joint_angles]

    # Forward kinematics
    positions = kinematics.forward_kinematics(joint_angles)
    print(f"Snake positions: {len(positions)} joints")
    print(f"Head position: {positions[-1]}")

    # Workspace calculation
    workspace = kinematics.calculate_workspace(joint_angles)
    print(f"Workspace bounds: {workspace.get('min_bounds', 'N/A')} to {workspace.get('max_bounds', 'N/A')}")

    # Dynamics example
    dynamics = SnakeDynamics(num_joints=10, link_length=0.08, joint_mass=0.05)

    # Example velocities and accelerations
    joint_velocities = [1.0] * 10  # rad/s
    joint_accelerations = [0.5] * 10  # rad/s²

    # Calculate torques
    torques = dynamics.calculate_joint_torques(joint_angles, joint_velocities, joint_accelerations)
    print(f"Joint torques: {torques[:3]}... (showing first 3)")

    # Motion analysis
    analysis = MotionAnalysis()

    # Create sample trajectory data for analysis
    trajectory_data = {
        'joints': {
            f'joint_{i}': {
                'time': np.linspace(0, 2, 100),
                'position': np.sin(2 * np.pi * np.linspace(0, 2, 100) + i * 0.5),
                'velocity': 2 * np.pi * np.cos(2 * np.pi * np.linspace(0, 2, 100) + i * 0.5),
                'acceleration': - (2 * np.pi)**2 * np.sin(2 * np.pi * np.linspace(0, 2, 100) + i * 0.5)
            }
            for i in range(10)
        }
    }

    efficiency = analysis.analyze_gait_efficiency(trajectory_data)
    print(f"Gait efficiency: {efficiency}")

    print("Snake kinematics and dynamics utilities initialized successfully")