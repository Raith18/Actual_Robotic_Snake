"""
Test cases for Quintic Polynomial Trajectory Generator
"""

import unittest
import numpy as np
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.quintic_polynomial import (
    QuinticPolynomialTrajectory,
    TrajectoryType,
    TrajectoryConstraints
)

class TestQuinticPolynomialTrajectory(unittest.TestCase):
    """Test cases for quintic polynomial trajectory generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = QuinticPolynomialTrajectory(num_joints=10)
        self.duration = 2.0
        self.amplitude = 1.0
        self.frequency = 1.0

    def test_initialization(self):
        """Test proper initialization of trajectory generator."""
        self.assertEqual(self.generator.num_joints, 10)
        self.assertIsNotNone(self.generator.coefficients)
        self.assertIsNotNone(self.generator.trajectory_cache)

    def test_polynomial_coefficients_generation(self):
        """Test generation of quintic polynomial coefficients."""
        constraints = TrajectoryConstraints(
            start_position=np.array([0.0]),
            end_position=np.array([1.0]),
            start_velocity=np.array([0.0]),
            end_velocity=np.array([0.0]),
            start_acceleration=np.array([0.0]),
            end_acceleration=np.array([0.0]),
            time_duration=1.0
        )

        coeffs = self.generator.generate_polynomial_coefficients(constraints)

        self.assertEqual(len(coeffs), 6)
        self.assertIsInstance(coeffs, np.ndarray)

        # Test that boundary conditions are satisfied
        t = np.array([0, 1.0])
        for i, t_val in enumerate([0, constraints.time_duration]):
            position = np.polyval(coeffs[::-1], t_val)  # Reverse coeffs for np.polyval
            expected = constraints.start_position if i == 0 else constraints.end_position
            np.testing.assert_almost_equal(position, expected[0], decimal=5)

    def test_joint_trajectory_generation(self):
        """Test generation of individual joint trajectories."""
        trajectory = self.generator.generate_joint_trajectory(
            joint_index=0,
            trajectory_type=TrajectoryType.SERPENTINE,
            duration=self.duration,
            amplitude=self.amplitude,
            frequency=self.frequency
        )

        # Check required keys
        required_keys = ['time', 'position', 'velocity', 'acceleration', 'joint_index', 'trajectory_type']
        for key in required_keys:
            self.assertIn(key, trajectory)

        # Check data consistency
        self.assertEqual(len(trajectory['position']), len(trajectory['velocity']))
        self.assertEqual(len(trajectory['velocity']), len(trajectory['acceleration']))
        self.assertEqual(trajectory['joint_index'], 0)
        self.assertEqual(trajectory['trajectory_type'], TrajectoryType.SERPENTINE.value)

    def test_snake_trajectory_generation(self):
        """Test generation of complete snake trajectory."""
        trajectory = self.generator.generate_snake_trajectory(
            TrajectoryType.SERPENTINE,
            duration=self.duration,
            amplitude=self.amplitude,
            frequency=self.frequency
        )

        # Check structure
        self.assertIn('trajectory_type', trajectory)
        self.assertIn('duration', trajectory)
        self.assertIn('joints', trajectory)
        self.assertEqual(len(trajectory['joints']), self.generator.num_joints)

        # Check each joint has trajectory data
        for joint_key in trajectory['joints']:
            joint_traj = trajectory['joints'][joint_key]
            self.assertIn('position', joint_traj)
            self.assertIn('velocity', joint_traj)
            self.assertIn('acceleration', joint_traj)

    def test_trajectory_optimization(self):
        """Test trajectory optimization functionality."""
        original_trajectory = self.generator.generate_snake_trajectory(
            TrajectoryType.SERPENTINE,
            duration=self.duration,
            amplitude=self.amplitude,
            frequency=self.frequency
        )

        optimized_trajectory = self.generator.optimize_trajectory(
            original_trajectory,
            optimization_target="efficiency"
        )

        # Optimized trajectory should have different parameters
        self.assertIsNotNone(optimized_trajectory)
        self.assertIn('efficiency_metric', optimized_trajectory)

    def test_physical_constraints(self):
        """Test that generated trajectories respect physical constraints."""
        trajectory = self.generator.generate_joint_trajectory(
            joint_index=0,
            trajectory_type=TrajectoryType.SERPENTINE,
            duration=self.duration,
            amplitude=self.generator.max_joint_angle * 2,  # Exceed limit
            frequency=self.frequency
        )

        # Check that positions are within limits
        positions = trajectory['position']
        self.assertTrue(np.all(np.abs(positions) <= self.generator.max_joint_angle))

        # Check that velocities are within limits
        velocities = trajectory['velocity']
        self.assertTrue(np.all(np.abs(velocities) <= self.generator.max_angular_velocity))

        # Check that accelerations are within limits
        accelerations = trajectory['acceleration']
        self.assertTrue(np.all(np.abs(accelerations) <= self.generator.max_angular_acceleration))

    def test_trajectory_serialization(self):
        """Test saving and loading trajectories."""
        trajectory = self.generator.generate_snake_trajectory(
            TrajectoryType.SERPENTINE,
            duration=self.duration,
            amplitude=self.amplitude,
            frequency=self.frequency
        )

        # Test save
        test_file = "test_trajectory.npz"
        self.generator.save_trajectory(trajectory, test_file)

        # Test load
        loaded_trajectory = self.generator.load_trajectory(test_file)

        # Check that loaded trajectory has same structure
        self.assertEqual(trajectory['trajectory_type'], loaded_trajectory['trajectory_type'])
        self.assertEqual(trajectory['duration'], loaded_trajectory['duration'])
        self.assertEqual(len(trajectory['joints']), len(loaded_trajectory['joints']))

        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

    def test_efficiency_calculation(self):
        """Test efficiency metric calculation."""
        trajectory = self.generator.generate_snake_trajectory(
            TrajectoryType.SERPENTINE,
            duration=self.duration,
            amplitude=self.amplitude,
            frequency=self.frequency
        )

        # Check that efficiency metric is calculated and reasonable
        self.assertIn('efficiency_metric', trajectory)
        self.assertGreater(trajectory['efficiency_metric'], 0)

    def test_different_trajectory_types(self):
        """Test different types of trajectories."""
        trajectory_types = [
            TrajectoryType.STRAIGHT,
            TrajectoryType.CIRCULAR,
            TrajectoryType.SERPENTINE,
            TrajectoryType.SPIRAL
        ]

        for traj_type in trajectory_types:
            trajectory = self.generator.generate_joint_trajectory(
                joint_index=0,
                trajectory_type=traj_type,
                duration=self.duration,
                amplitude=self.amplitude,
                frequency=self.frequency
            )

            self.assertIn('position', trajectory)
            self.assertIn('velocity', trajectory)
            self.assertIn('acceleration', trajectory)

    def test_error_handling(self):
        """Test error handling in trajectory generation."""
        # Test with invalid constraints
        invalid_constraints = TrajectoryConstraints(
            start_position=np.array([np.nan]),
            end_position=np.array([1.0]),
            start_velocity=np.array([0.0]),
            end_velocity=np.array([0.0]),
            start_acceleration=np.array([0.0]),
            end_acceleration=np.array([0.0]),
            time_duration=0.0  # Invalid duration
        )

        coeffs = self.generator.generate_polynomial_coefficients(invalid_constraints)
        # Should return zeros array on error
        np.testing.assert_array_equal(coeffs, np.zeros(6))

if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)