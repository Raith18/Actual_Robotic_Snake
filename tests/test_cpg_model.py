"""
Test cases for Central Pattern Generator (CPG) Model
"""

import unittest
import numpy as np
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cpg_model import (
    CentralPatternGenerator,
    CPGType,
    GaitPattern,
    HopfOscillator,
    MatsuokaOscillator,
    CPGParameters,
    JointCoupling
)

class TestHopfOscillator(unittest.TestCase):
    """Test cases for Hopf oscillator."""

    def setUp(self):
        """Set up test fixtures."""
        self.oscillator = HopfOscillator(mu=1.0, omega=2*np.pi)

    def test_initialization(self):
        """Test proper initialization of Hopf oscillator."""
        self.assertEqual(self.oscillator.mu, 1.0)
        self.assertEqual(self.oscillator.omega, 2*np.pi)
        self.assertEqual(self.oscillator.x, 1.0)  # sqrt(mu)
        self.assertEqual(self.oscillator.y, 0.0)

    def test_step_integration(self):
        """Test integration step."""
        dt = 0.01
        x, y = self.oscillator.step(dt)

        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        # Should maintain roughly constant amplitude
        amplitude = np.sqrt(x**2 + y**2)
        self.assertAlmostEqual(amplitude, 1.0, places=2)

    def test_amplitude_calculation(self):
        """Test amplitude calculation."""
        amplitude = self.oscillator.get_amplitude()
        expected = np.sqrt(self.oscillator.x**2 + self.oscillator.y**2)
        self.assertEqual(amplitude, expected)

    def test_phase_calculation(self):
        """Test phase calculation."""
        phase = self.oscillator.get_phase()
        expected = np.arctan2(self.oscillator.y, self.oscillator.x)
        self.assertEqual(phase, expected)

class TestMatsuokaOscillator(unittest.TestCase):
    """Test cases for Matsuoka oscillator."""

    def setUp(self):
        """Set up test fixtures."""
        self.oscillator = MatsuokaOscillator(tau=0.1, beta=2.5, omega=2*np.pi)

    def test_initialization(self):
        """Test proper initialization of Matsuoka oscillator."""
        self.assertEqual(self.oscillator.tau, 0.1)
        self.assertEqual(self.oscillator.beta, 2.5)
        self.assertEqual(self.oscillator.omega, 2*np.pi)
        self.assertEqual(self.oscillator.x1, 0.0)
        self.assertEqual(self.oscillator.x2, 0.0)
        self.assertEqual(self.oscillator.v1, 0.0)
        self.assertEqual(self.oscillator.v2, 0.0)

    def test_step_integration(self):
        """Test integration step."""
        dt = 0.01
        x1, x2 = self.oscillator.step(dt)

        self.assertIsInstance(x1, float)
        self.assertIsInstance(x2, float)

class TestCentralPatternGenerator(unittest.TestCase):
    """Test cases for Central Pattern Generator."""

    def setUp(self):
        """Set up test fixtures."""
        self.cpg = CentralPatternGenerator(num_joints=10, cpg_type=CPGType.HOPF)

    def test_initialization(self):
        """Test proper initialization of CPG."""
        self.assertEqual(self.cpg.num_joints, 10)
        self.assertEqual(self.cpg.cpg_type, CPGType.HOPF)
        self.assertEqual(len(self.cpg.oscillators), 10)
        self.assertIsInstance(self.cpg.parameters, CPGParameters)
        self.assertIsInstance(self.cpg.coupling, JointCoupling)

    def test_oscillator_types(self):
        """Test different oscillator types."""
        # Test Hopf oscillators
        cpg_hopf = CentralPatternGenerator(num_joints=5, cpg_type=CPGType.HOPF)
        self.assertEqual(len(cpg_hopf.oscillators), 5)
        for osc in cpg_hopf.oscillators:
            self.assertIsInstance(osc, HopfOscillator)

        # Test Matsuoka oscillators
        cpg_matsuoka = CentralPatternGenerator(num_joints=5, cpg_type=CPGType.MATSUOKA)
        self.assertEqual(len(cpg_matsuoka.oscillators), 5)
        for osc in cpg_matsuoka.oscillators:
            self.assertIsInstance(osc, MatsuokaOscillator)

    def test_cpg_step(self):
        """Test CPG integration step."""
        dt = 0.01
        joint_angles = self.cpg.step(dt)

        self.assertEqual(len(joint_angles), self.cpg.num_joints)
        self.assertIsInstance(joint_angles, np.ndarray)

        # All angles should be finite
        self.assertTrue(np.all(np.isfinite(joint_angles)))

    def test_gait_patterns(self):
        """Test different gait pattern configurations."""
        gait_patterns = [
            GaitPattern.SERPENTINE,
            GaitPattern.SIDEWINDING,
            GaitPattern.CONCERTINA,
            GaitPattern.SLIDING,
            GaitPattern.ROLLING
        ]

        for gait in gait_patterns:
            cpg = CentralPatternGenerator(num_joints=8, cpg_type=CPGType.HOPF)
            cpg.set_gait_pattern(gait)

            # Should not raise any exceptions
            dt = 0.01
            angles = cpg.step(dt)
            self.assertEqual(len(angles), 8)

    def test_serpentine_gait_configuration(self):
        """Test serpentine gait specific configuration."""
        cpg = CentralPatternGenerator(num_joints=10, cpg_type=CPGType.HOPF)
        cpg.set_gait_pattern(GaitPattern.SERPENTINE, wave_number=2.0)

        # Check that oscillators have different phases
        phases = []
        for osc in cpg.oscillators:
            phases.append(osc.get_phase())

        phases = np.array(phases)
        # Phases should be distributed across the snake
        self.assertGreater(np.std(phases), 0.1)

    def test_environment_adaptation(self):
        """Test environment adaptation functionality."""
        original_frequency = self.cpg.parameters.frequency
        original_amplitude = self.cpg.parameters.amplitude
        original_coupling = self.cpg.coupling.neighbor_coupling

        environment_feedback = {
            'terrain_difficulty': 0.8,
            'obstacle_density': 0.6,
            'surface_type': 'rough'
        }

        self.cpg.adapt_to_environment(environment_feedback)

        # Parameters should have changed
        self.assertNotEqual(self.cpg.parameters.frequency, original_frequency)
        self.assertNotEqual(self.cpg.parameters.amplitude, original_amplitude)
        self.assertNotEqual(self.cpg.coupling.neighbor_coupling, original_coupling)

    def test_oscillator_states(self):
        """Test oscillator state retrieval."""
        dt = 0.01
        self.cpg.step(dt)

        states = self.cpg.get_oscillator_states()

        required_keys = ['joint_angles', 'joint_velocities', 'phases']
        for key in required_keys:
            self.assertIn(key, states)
            self.assertEqual(len(states[key]), self.cpg.num_joints)

    def test_cpg_reset(self):
        """Test CPG reset functionality."""
        # Run for a few steps
        dt = 0.01
        for _ in range(10):
            self.cpg.step(dt)

        # Verify state is not zero
        initial_angles = self.cpg.joint_angles.copy()

        # Reset
        self.cpg.reset()

        # State should be reset
        np.testing.assert_array_equal(self.cpg.joint_angles, np.zeros(self.cpg.num_joints))
        np.testing.assert_array_equal(self.cpg.joint_velocities, np.zeros(self.cpg.num_joints))
        np.testing.assert_array_equal(self.cpg.phases, np.zeros(self.cpg.num_joints))

    def test_state_serialization(self):
        """Test CPG state save/load functionality."""
        # Run for a few steps to create state
        dt = 0.01
        for _ in range(10):
            self.cpg.step(dt)

        # Save state
        test_file = "test_cpg_state.json"
        self.cpg.save_cpg_state(test_file)

        # Create new CPG and load state
        new_cpg = CentralPatternGenerator(num_joints=10, cpg_type=CPGType.HOPF)
        success = new_cpg.load_cpg_state(test_file)

        self.assertTrue(success)

        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

    def test_coupling_calculation(self):
        """Test coupling signal calculation."""
        # Test with different oscillator indices
        for i in range(min(3, self.cpg.num_joints)):
            coupling_signal = self.cpg._calculate_coupling_signal(i)
            self.assertIsInstance(coupling_signal, float)

class TestIntegration(unittest.TestCase):
    """Integration tests for CPG system."""

    def test_cpg_trajectory_generation(self):
        """Test complete trajectory generation."""
        cpg = CentralPatternGenerator(num_joints=8, cpg_type=CPGType.HOPF)
        cpg.set_gait_pattern(GaitPattern.SERPENTINE)

        # Generate trajectory for 1 second
        duration = 1.0
        dt = 0.01
        num_steps = int(duration / dt)

        trajectory = []
        for _ in range(num_steps):
            angles = cpg.step(dt)
            trajectory.append(angles.copy())

        trajectory = np.array(trajectory)

        # Check trajectory properties
        self.assertEqual(trajectory.shape, (num_steps, 8))

        # Check that trajectory varies over time
        self.assertGreater(np.std(trajectory), 0.1)

    def test_multiple_gait_switching(self):
        """Test switching between different gaits."""
        cpg = CentralPatternGenerator(num_joints=6, cpg_type=CPGType.HOPF)

        gaits = [GaitPattern.SERPENTINE, GaitPattern.SIDEWINDING, GaitPattern.CONCERTINA]

        for gait in gaits:
            cpg.set_gait_pattern(gait)
            dt = 0.01
            angles = cpg.step(dt)

            self.assertEqual(len(angles), 6)
            self.assertTrue(np.all(np.isfinite(angles)))

if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)

    # Create test suite
    test_classes = [
        TestHopfOscillator,
        TestMatsuokaOscillator,
        TestCentralPatternGenerator,
        TestIntegration
    ]

    # Run tests
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")