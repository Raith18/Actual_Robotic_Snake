"""
Central Pattern Generator (CPG) Model for Robotic Snake
Implements biologically-inspired locomotion patterns using coupled oscillators.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import json

class CPGType(Enum):
    """Types of CPG models."""
    HOPF = "hopf"
    MATSUOKA = "matsuoka"
    KURAMOTO = "kuramoto"
    VAN_DER_POL = "van_der_pol"

class GaitPattern(Enum):
    """Snake gait patterns."""
    SERPENTINE = "serpentine"
    SIDEWINDING = "sidewinding"
    CONCERTINA = "concertina"
    SLIDING = "sliding"
    ROLLING = "rolling"

@dataclass
class CPGParameters:
    """Parameters for CPG configuration."""
    frequency: float = 1.0  # Hz
    amplitude: float = 1.0
    coupling_strength: float = 0.5
    phase_bias: float = 0.0
    convergence_rate: float = 2.0

@dataclass
class JointCoupling:
    """Coupling configuration between joints."""
    neighbor_coupling: float = 0.8
    global_coupling: float = 0.1
    phase_lag: float = np.pi / 5  # Phase difference between adjacent joints

class HopfOscillator:
    """Hopf oscillator implementation for CPG."""

    def __init__(self, mu: float = 1.0, omega: float = 2*np.pi):
        """
        Initialize Hopf oscillator.

        Args:
            mu: Bifurcation parameter (radius of oscillation)
            omega: Natural frequency
        """
        self.mu = mu
        self.omega = omega
        self.x = np.sqrt(mu)  # Initial condition on limit cycle
        self.y = 0.0

    def step(self, dt: float, coupling_input: float = 0.0) -> Tuple[float, float]:
        """
        Perform one integration step.

        Args:
            dt: Time step
            coupling_input: External coupling input

        Returns:
            Tuple of (x, y) state variables
        """
        # Hopf oscillator equations with coupling
        dx = (self.mu - self.x**2 - self.y**2) * self.x - self.omega * self.y + coupling_input
        dy = (self.mu - self.x**2 - self.y**2) * self.y + self.omega * self.x

        # Integrate
        self.x += dx * dt
        self.y += dy * dt

        return self.x, self.y

    def get_amplitude(self) -> float:
        """Get current amplitude."""
        return np.sqrt(self.x**2 + self.y**2)

    def get_phase(self) -> float:
        """Get current phase."""
        return np.arctan2(self.y, self.x)

class MatsuokaOscillator:
    """Matsuoka oscillator implementation."""

    def __init__(self, tau: float = 0.1, beta: float = 2.5, omega: float = 2*np.pi):
        """
        Initialize Matsuoka oscillator.

        Args:
            tau: Time constant
            beta: Adaptation parameter
            omega: Natural frequency
        """
        self.tau = tau
        self.beta = beta
        self.omega = omega

        # State variables
        self.x1 = 0.0  # First extensor neuron
        self.x2 = 0.0  # Second flexor neuron
        self.v1 = 0.0  # Adaptation variable 1
        self.v2 = 0.0  # Adaptation variable 2

    def step(self, dt: float, coupling_input: float = 0.0) -> Tuple[float, float]:
        """
        Perform one integration step.

        Args:
            dt: Time step
            coupling_input: External coupling input

        Returns:
            Tuple of (x1, x2) output variables
        """
        # Matsuoka oscillator equations
        dx1 = (-self.x1 - self.beta * self.v1 - coupling_input + 2.0) / self.tau
        dx2 = (-self.x2 - self.beta * self.v2 + coupling_input + 2.0) / self.tau
        dv1 = (-self.v1 + max(self.x1, 0.0)) / self.tau
        dv2 = (-self.v2 + max(self.x2, 0.0)) / self.tau

        # Integrate
        self.x1 += dx1 * dt
        self.x2 += dx2 * dt
        self.v1 += dv1 * dt
        self.v2 += dv2 * dt

        return self.x1, self.x2

class CentralPatternGenerator:
    """
    Central Pattern Generator for robotic snake locomotion.
    """

    def __init__(self, num_joints: int = 10, cpg_type: CPGType = CPGType.HOPF):
        # Initialize oscillators first so they can be used in _initialize_oscillators
        self.oscillators = []
        """
        Initialize CPG system.

        Args:
            num_joints: Number of snake joints
            cpg_type: Type of oscillator to use
        """
        self.num_joints = num_joints
        self.cpg_type = cpg_type

        # CPG parameters
        self.parameters = CPGParameters()
        self.coupling = JointCoupling()

        # Initialize oscillators
        self.oscillators = []
        self._initialize_oscillators()

        # State tracking
        self.joint_angles = np.zeros(num_joints)
        self.joint_velocities = np.zeros(num_joints)
        self.phases = np.zeros(num_joints)

        # Setup logging
        self.logger = logging.getLogger('CPG_Model')

    def _initialize_oscillators(self):
        """Initialize oscillators based on type."""
        if self.cpg_type == CPGType.HOPF:
            for i in range(self.num_joints):
                # Vary frequency slightly for each joint to create traveling wave
                freq_variation = 1.0 + 0.1 * np.sin(2 * np.pi * i / self.num_joints)
                mu = self.parameters.amplitude
                omega = 2 * np.pi * self.parameters.frequency * freq_variation

                oscillator = HopfOscillator(mu=mu, omega=omega)
                self.oscillators.append(oscillator)

        elif self.cpg_type == CPGType.MATSUOKA:
            for i in range(self.num_joints):
                freq_variation = 1.0 + 0.1 * np.sin(2 * np.pi * i / self.num_joints)
                tau = 1.0 / (2 * np.pi * self.parameters.frequency * freq_variation)
                omega = 2 * np.pi * self.parameters.frequency * freq_variation

                oscillator = MatsuokaOscillator(tau=tau, omega=omega)
                self.oscillators.append(oscillator)

    def _calculate_coupling_signal(self, oscillator_idx: int) -> float:
        """Calculate coupling signal for a specific oscillator."""
        coupling_signal = 0.0

        # Neighbor coupling
        for i in range(self.num_joints):
            if i != oscillator_idx:
                distance = min(abs(i - oscillator_idx), self.num_joints - abs(i - oscillator_idx))
                coupling_strength = self.coupling.neighbor_coupling / (1 + distance)

                if self.cpg_type == CPGType.HOPF:
                    phase_diff = self.oscillators[i].get_phase() - self.oscillators[oscillator_idx].get_phase()
                    coupling_signal += coupling_strength * np.sin(phase_diff)
                else:
                    # For Matsuoka, use output coupling
                    x1_i, x2_i = self.oscillators[i].x1, self.oscillators[i].x2
                    x1_self, x2_self = self.oscillators[oscillator_idx].x1, self.oscillators[oscillator_idx].x2
                    coupling_signal += coupling_strength * (x1_i - x1_self)

        return coupling_signal

    def step(self, dt: float, external_input: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform one CPG integration step.

        Args:
            dt: Time step
            external_input: External input array (optional)

        Returns:
            Array of joint angles
        """
        try:
            # Apply coupling and external inputs
            for i, oscillator in enumerate(self.oscillators):
                coupling_signal = self._calculate_coupling_signal(i)

                if external_input is not None and i < len(external_input):
                    coupling_signal += external_input[i]

                # Step oscillator
                if self.cpg_type == CPGType.HOPF:
                    x, y = oscillator.step(dt, coupling_signal)
                    # Convert oscillator state to joint angle
                    self.joint_angles[i] = self.parameters.amplitude * x
                    self.phases[i] = oscillator.get_phase()

                elif self.cpg_type == CPGType.MATSUOKA:
                    x1, x2 = oscillator.step(dt, coupling_signal)
                    # Convert neuron outputs to joint angle
                    self.joint_angles[i] = self.parameters.amplitude * (x1 - x2)
                    self.phases[i] = np.arctan2(x2, x1)

            # Calculate velocities
            self.joint_velocities = np.gradient(self.joint_angles, dt)

            return self.joint_angles.copy()

        except Exception as e:
            self.logger.error(f"Error in CPG step: {e}")
            return self.joint_angles.copy()

    def set_gait_pattern(self, gait: GaitPattern, **kwargs):
        """
        Configure CPG for specific gait pattern.

        Args:
            gait: Target gait pattern
            **kwargs: Additional parameters for gait configuration
        """
        try:
            if gait == GaitPattern.SERPENTINE:
                self._configure_serpentine_gait(**kwargs)
            elif gait == GaitPattern.SIDEWINDING:
                self._configure_sidewinding_gait(**kwargs)
            elif gait == GaitPattern.CONCERTINA:
                self._configure_concertina_gait(**kwargs)
            elif gait == GaitPattern.SLIDING:
                self._configure_sliding_gait(**kwargs)
            elif gait == GaitPattern.ROLLING:
                self._configure_rolling_gait(**kwargs)

        except Exception as e:
            self.logger.error(f"Error setting gait pattern: {e}")

    def _configure_serpentine_gait(self, **kwargs):
        """Configure for serpentine locomotion."""
        wave_number = kwargs.get('wave_number', 2.0)
        amplitude_scale = kwargs.get('amplitude_scale', 1.0)

        for i, oscillator in enumerate(self.oscillators):
            # Create traveling wave
            phase_offset = (2 * np.pi * wave_number * i) / self.num_joints

            if self.cpg_type == CPGType.HOPF:
                oscillator.omega = 2 * np.pi * self.parameters.frequency
                # Reset to create phase offset
                oscillator.x = self.parameters.amplitude * np.cos(phase_offset)
                oscillator.y = self.parameters.amplitude * np.sin(phase_offset)

    def _configure_sidewinding_gait(self, **kwargs):
        """Configure for sidewinding locomotion."""
        amplitude_scale = kwargs.get('amplitude_scale', 0.8)
        frequency_scale = kwargs.get('frequency_scale', 1.2)

        for i, oscillator in enumerate(self.oscillators):
            # Alternating pattern for sidewinding
            if i % 2 == 0:
                phase_offset = 0
            else:
                phase_offset = np.pi

            if self.cpg_type == CPGType.HOPF:
                oscillator.omega = 2 * np.pi * self.parameters.frequency * frequency_scale
                oscillator.x = amplitude_scale * self.parameters.amplitude * np.cos(phase_offset)
                oscillator.y = amplitude_scale * self.parameters.amplitude * np.sin(phase_offset)

    def _configure_concertina_gait(self, **kwargs):
        """Configure for concertina locomotion."""
        # Concertina uses more discrete movements
        for i, oscillator in enumerate(self.oscillators):
            # Reduce frequency for more deliberate movements
            if self.cpg_type == CPGType.HOPF:
                oscillator.omega = 2 * np.pi * self.parameters.frequency * 0.5

    def _configure_sliding_gait(self, **kwargs):
        """Configure for sliding locomotion."""
        # Similar to serpentine but with different coupling
        self.coupling.neighbor_coupling = 0.9  # Stronger coupling for sliding

    def _configure_rolling_gait(self, **kwargs):
        """Configure for rolling locomotion."""
        # Rolling pattern with 90-degree phase shifts
        for i, oscillator in enumerate(self.oscillators):
            phase_offset = (np.pi / 2) * i

            if self.cpg_type == CPGType.HOPF:
                oscillator.x = self.parameters.amplitude * np.cos(phase_offset)
                oscillator.y = self.parameters.amplitude * np.sin(phase_offset)

    def adapt_to_environment(self, environment_feedback: Dict[str, float]):
        """
        Adapt CPG parameters based on environment feedback.

        Args:
            environment_feedback: Dictionary containing environment information
        """
        try:
            # Adjust frequency based on terrain difficulty
            if 'terrain_difficulty' in environment_feedback:
                difficulty = environment_feedback['terrain_difficulty']
                self.parameters.frequency *= (0.5 + 0.5 * (1 - difficulty))

            # Adjust amplitude based on obstacle density
            if 'obstacle_density' in environment_feedback:
                density = environment_feedback['obstacle_density']
                self.parameters.amplitude *= (0.3 + 0.7 * (1 - density))

            # Adjust coupling based on surface type
            if 'surface_type' in environment_feedback:
                surface = environment_feedback['surface_type']
                if surface == 'rough':
                    self.coupling.neighbor_coupling = 0.9
                elif surface == 'smooth':
                    self.coupling.neighbor_coupling = 0.6

        except Exception as e:
            self.logger.error(f"Error adapting to environment: {e}")

    def get_oscillator_states(self) -> Dict[str, np.ndarray]:
        """Get current states of all oscillators."""
        states = {
            'joint_angles': self.joint_angles.copy(),
            'joint_velocities': self.joint_velocities.copy(),
            'phases': self.phases.copy()
        }

        if self.cpg_type == CPGType.HOPF:
            amplitudes = [osc.get_amplitude() for osc in self.oscillators]
            states['amplitudes'] = np.array(amplitudes)

        return states

    def reset(self):
        """Reset all oscillators to initial state."""
        for oscillator in self.oscillators:
            if self.cpg_type == CPGType.HOPF:
                oscillator.x = np.sqrt(oscillator.mu)
                oscillator.y = 0.0
            elif self.cpg_type == CPGType.MATSUOKA:
                oscillator.x1 = 0.0
                oscillator.x2 = 0.0
                oscillator.v1 = 0.0
                oscillator.v2 = 0.0

        self.joint_angles.fill(0.0)
        self.joint_velocities.fill(0.0)
        self.phases.fill(0.0)

    def save_cpg_state(self, filename: str):
        """Save CPG state to file."""
        try:
            state = {
                'cpg_type': self.cpg_type.value,
                'num_joints': self.num_joints,
                'parameters': {
                    'frequency': self.parameters.frequency,
                    'amplitude': self.parameters.amplitude,
                    'coupling_strength': self.parameters.coupling_strength,
                    'phase_bias': self.parameters.phase_bias,
                    'convergence_rate': self.parameters.convergence_rate
                },
                'coupling': {
                    'neighbor_coupling': self.coupling.neighbor_coupling,
                    'global_coupling': self.coupling.global_coupling,
                    'phase_lag': self.coupling.phase_lag
                },
                'states': self.get_oscillator_states()
            }

            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)

            self.logger.info(f"CPG state saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error saving CPG state: {e}")

    def load_cpg_state(self, filename: str) -> bool:
        """Load CPG state from file."""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)

            # Restore parameters
            params = state['parameters']
            self.parameters.frequency = params['frequency']
            self.parameters.amplitude = params['amplitude']
            self.parameters.coupling_strength = params['coupling_strength']
            self.parameters.phase_bias = params['phase_bias']
            self.parameters.convergence_rate = params['convergence_rate']

            # Restore coupling
            coupling = state['coupling']
            self.coupling.neighbor_coupling = coupling['neighbor_coupling']
            self.coupling.global_coupling = coupling['global_coupling']
            self.coupling.phase_lag = coupling['phase_lag']

            # Restore states
            states = state['states']
            self.joint_angles = np.array(states['joint_angles'])
            self.joint_velocities = np.array(states['joint_velocities'])
            self.phases = np.array(states['phases'])

            self.logger.info(f"CPG state loaded from {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading CPG state: {e}")
            return False

    def visualize_cpg_activity(self, duration: float = 2.0, dt: float = 0.01,
                              save_plot: bool = False):
        """Visualize CPG activity over time."""
        try:
            num_steps = int(duration / dt)
            time_array = np.linspace(0, duration, num_steps)

            # Store joint angle history
            angle_history = np.zeros((self.num_joints, num_steps))

            # Run simulation
            for step in range(num_steps):
                angles = self.step(dt)
                angle_history[:, step] = angles

            # Create visualization
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # Plot joint angles over time
            for i in range(self.num_joints):
                axes[0].plot(time_array, angle_history[i, :],
                           label=f'Joint {i}', alpha=0.7)

            axes[0].set_title('Joint Angles Over Time')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Angle (rad)')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True)

            # Plot phase relationships
            phases = self.phases
            phase_circle = np.linspace(0, 2*np.pi, 100)

            for i, phase in enumerate(phases):
                x = np.cos(phase)
                y = np.sin(phase)
                axes[1].plot(x, y, 'o', markersize=8, label=f'Joint {i}')

            # Draw unit circle
            axes[1].plot(np.cos(phase_circle), np.sin(phase_circle), 'k--', alpha=0.3)
            axes[1].set_title('Phase Relationships')
            axes[1].set_xlabel('cos(φ)')
            axes[1].set_ylabel('sin(φ)')
            axes[1].set_aspect('equal')
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()

            if save_plot:
                plt.savefig('cpg_activity.png', dpi=300, bbox_inches='tight')
                self.logger.info("CPG visualization saved")

            plt.show()

        except Exception as e:
            self.logger.error(f"Error visualizing CPG activity: {e}")

# Example usage
if __name__ == "__main__":
    # Create CPG for 10-joint snake
    cpg = CentralPatternGenerator(num_joints=10, cpg_type=CPGType.HOPF)

    # Configure for serpentine gait
    cpg.set_gait_pattern(GaitPattern.SERPENTINE, wave_number=2.0)

    # Visualize CPG activity
    cpg.visualize_cpg_activity(duration=2.0, save_plot=True)

    # Test adaptation
    environment_feedback = {
        'terrain_difficulty': 0.7,
        'obstacle_density': 0.3,
        'surface_type': 'rough'
    }
    cpg.adapt_to_environment(environment_feedback)

    print("CPG system initialized and tested successfully")