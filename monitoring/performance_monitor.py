"""
Performance Monitoring and Logging System for Robotic Snake
Comprehensive monitoring of system performance, health, and efficiency metrics.
"""

import time
import threading
import logging
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import json
import os
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    frame_time: float = 0.0
    processing_rate: float = 0.0
    power_consumption: float = 0.0
    temperature: float = 0.0
    efficiency_score: float = 0.0

@dataclass
class SystemHealth:
    """System health indicators."""
    overall_status: str = "unknown"
    component_status: Dict[str, str] = field(default_factory=dict)
    error_count: int = 0
    warning_count: int = 0
    last_error: Optional[str] = None
    uptime: float = 0.0

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for the robotic snake.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        """
        Initialize performance monitor.

        Args:
            config_path: Path to system configuration file
        """
        self.config = self._load_config(config_path)

        # Monitoring parameters
        self.monitoring_interval = 1.0  # seconds
        self.history_length = 1000      # Number of samples to keep

        # Performance tracking
        self.metrics_history = deque(maxlen=self.history_length)
        self.system_health = SystemHealth()
        self.start_time = time.time()

        # Component monitors
        self.component_monitors = {}
        self.alert_thresholds = self._load_alert_thresholds()

        # Threading
        self.monitoring_thread = None
        self.is_monitoring = False
        self.monitor_lock = threading.Lock()

        # Logging setup
        self.logger = self._setup_logging()
        self.performance_logger = logging.getLogger('PerformanceMonitor')

        # File logging
        self.log_file = Path("logs/performance.log")
        self.log_file.parent.mkdir(exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            import yaml
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}

    def _load_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load alert thresholds for different metrics."""
        return {
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'frame_time': {'warning': 0.05, 'critical': 0.1},
            'temperature': {'warning': 50.0, 'critical': 70.0},
            'error_rate': {'warning': 0.1, 'critical': 0.3}
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )

        # Setup main logger
        logger = logging.getLogger('SnakePerformance')
        logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def start_monitoring(self) -> bool:
        """
        Start performance monitoring.

        Returns:
            True if monitoring started successfully
        """
        try:
            self.is_monitoring = True
            self.start_time = time.time()

            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()

            self.logger.info("Performance monitoring started")
            return True

        except Exception as e:
            self.logger.error(f"Error starting performance monitoring: {e}")
            return False

    def stop_monitoring(self):
        """Stop performance monitoring."""
        try:
            self.is_monitoring = False

            if self.monitoring_thread and self.monitoring_thread.is_alive:
                self.monitoring_thread.join(timeout=2.0)

            # Generate final report
            self._generate_final_report()

            self.logger.info("Performance monitoring stopped")

        except Exception as e:
            self.logger.error(f"Error stopping performance monitoring: {e}")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.is_monitoring:
                # Collect current metrics
                metrics = self._collect_system_metrics()

                # Store metrics
                with self.monitor_lock:
                    self.metrics_history.append(metrics)

                # Update system health
                self._update_system_health(metrics)

                # Check for alerts
                self._check_alerts(metrics)

                # Log periodic summary
                if len(self.metrics_history) % 60 == 0:  # Every minute
                    self._log_performance_summary()

                # Wait for next interval
                time.sleep(self.monitoring_interval)

        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.is_monitoring = False

    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        try:
            timestamp = time.time()

            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # GPU usage (if available)
            gpu_usage = self._get_gpu_usage()

            # Process-specific metrics
            current_process = psutil.Process()
            process_memory = current_process.memory_info().rss / 1024 / 1024  # MB

            # Calculate frame time and processing rate from recent history
            frame_time = self._calculate_average_frame_time()
            processing_rate = 1.0 / frame_time if frame_time > 0 else 0.0

            # Power consumption (estimated)
            power_consumption = self._estimate_power_consumption(cpu_usage, memory_usage)

            # Temperature (if available)
            temperature = self._get_system_temperature()

            # Efficiency score
            efficiency_score = self._calculate_efficiency_score(
                cpu_usage, memory_usage, frame_time, processing_rate
            )

            return PerformanceMetrics(
                timestamp=timestamp,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                frame_time=frame_time,
                processing_rate=processing_rate,
                power_consumption=power_consumption,
                temperature=temperature,
                efficiency_score=efficiency_score
            )

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return PerformanceMetrics(timestamp=time.time())

    def _get_gpu_usage(self) -> float:
        """Get GPU usage if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100  # Convert to percentage
        except ImportError:
            pass  # GPUtil not available
        except Exception as e:
            self.logger.debug(f"Error getting GPU usage: {e}")

        return 0.0

    def _calculate_average_frame_time(self) -> float:
        """Calculate average frame processing time."""
        try:
            if len(self.metrics_history) < 2:
                return 0.033  # Default 30 FPS

            # Get recent frame times
            recent_metrics = list(self.metrics_history)[-10:]
            frame_times = [m.frame_time for m in recent_metrics if m.frame_time > 0]

            if frame_times:
                return np.mean(frame_times)
            else:
                return 0.033

        except Exception as e:
            self.logger.error(f"Error calculating frame time: {e}")
            return 0.033

    def _estimate_power_consumption(self, cpu_usage: float, memory_usage: float) -> float:
        """Estimate power consumption based on resource usage."""
        try:
            # Simple power estimation model
            # In practice, this would use actual power measurements

            # Base power consumption (Watts)
            base_power = 5.0

            # CPU power (scale with usage)
            cpu_power = 10.0 * (cpu_usage / 100.0)

            # Memory power (smaller contribution)
            memory_power = 2.0 * (memory_usage / 100.0)

            total_power = base_power + cpu_power + memory_power

            return total_power

        except Exception as e:
            self.logger.error(f"Error estimating power consumption: {e}")
            return 0.0

    def _get_system_temperature(self) -> float:
        """Get system temperature if available."""
        try:
            # Try to get CPU temperature
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if 'cpu' in name.lower() or 'core' in name.lower():
                        return entries[0].current

        except Exception as e:
            self.logger.debug(f"Error getting system temperature: {e}")

        return 0.0

    def _calculate_efficiency_score(self, cpu_usage: float, memory_usage: float,
                                  frame_time: float, processing_rate: float) -> float:
        """Calculate overall system efficiency score."""
        try:
            # Normalize metrics (0-1 scale, where 1 is best)
            cpu_efficiency = max(0, 1.0 - (cpu_usage / 100.0))
            memory_efficiency = max(0, 1.0 - (memory_usage / 100.0))

            # Frame time efficiency (target: 30 FPS = 0.033s per frame)
            target_frame_time = 0.033
            frame_efficiency = max(0, 1.0 - min(1.0, frame_time / target_frame_time))

            # Processing rate efficiency
            target_rate = 30.0
            rate_efficiency = min(1.0, processing_rate / target_rate)

            # Weighted average
            weights = {
                'cpu': 0.3,
                'memory': 0.2,
                'frame': 0.3,
                'rate': 0.2
            }

            efficiency = (
                cpu_efficiency * weights['cpu'] +
                memory_efficiency * weights['memory'] +
                frame_efficiency * weights['frame'] +
                rate_efficiency * weights['rate']
            )

            return efficiency

        except Exception as e:
            self.logger.error(f"Error calculating efficiency score: {e}")
            return 0.0

    def _update_system_health(self, metrics: PerformanceMetrics):
        """Update system health based on current metrics."""
        try:
            # Overall status determination
            if (metrics.cpu_usage > self.alert_thresholds['cpu_usage']['critical'] or
                metrics.memory_usage > self.alert_thresholds['memory_usage']['critical'] or
                metrics.frame_time > self.alert_thresholds['frame_time']['critical']):
                self.system_health.overall_status = "critical"
            elif (metrics.cpu_usage > self.alert_thresholds['cpu_usage']['warning'] or
                  metrics.memory_usage > self.alert_thresholds['memory_usage']['warning']):
                self.system_health.overall_status = "warning"
            else:
                self.system_health.overall_status = "healthy"

            # Update uptime
            self.system_health.uptime = time.time() - self.start_time

        except Exception as e:
            self.logger.error(f"Error updating system health: {e}")

    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for alert conditions and log them."""
        try:
            # CPU alerts
            if metrics.cpu_usage > self.alert_thresholds['cpu_usage']['critical']:
                self.logger.critical(f"Critical CPU usage: {metrics.cpu_usage".1f"}%")
                self.system_health.error_count += 1
                self.system_health.last_error = f"High CPU usage: {metrics.cpu_usage".1f"}%"
            elif metrics.cpu_usage > self.alert_thresholds['cpu_usage']['warning']:
                self.logger.warning(f"High CPU usage: {metrics.cpu_usage".1f"}%")
                self.system_health.warning_count += 1

            # Memory alerts
            if metrics.memory_usage > self.alert_thresholds['memory_usage']['critical']:
                self.logger.critical(f"Critical memory usage: {metrics.memory_usage".1f"}%")
                self.system_health.error_count += 1
                self.system_health.last_error = f"High memory usage: {metrics.memory_usage".1f"}%"
            elif metrics.memory_usage > self.alert_thresholds['memory_usage']['warning']:
                self.logger.warning(f"High memory usage: {metrics.memory_usage".1f"}%")
                self.system_health.warning_count += 1

            # Frame time alerts
            if metrics.frame_time > self.alert_thresholds['frame_time']['critical']:
                self.logger.critical(f"Critical frame time: {metrics.frame_time".3f"}s")
                self.system_health.error_count += 1
                self.system_health.last_error = f"Slow frame time: {metrics.frame_time".3f"}s"
            elif metrics.frame_time > self.alert_thresholds['frame_time']['warning']:
                self.logger.warning(f"Slow frame time: {metrics.frame_time".3f"}s")
                self.system_health.warning_count += 1

        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")

    def _log_performance_summary(self):
        """Log periodic performance summary."""
        try:
            if len(self.metrics_history) < 10:
                return

            # Get recent metrics for averaging
            recent_metrics = list(self.metrics_history)[-10:]

            avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
            avg_memory = np.mean([m.memory_usage for m in recent_metrics])
            avg_frame_time = np.mean([m.frame_time for m in recent_metrics if m.frame_time > 0])
            avg_efficiency = np.mean([m.efficiency_score for m in recent_metrics])

            summary = (
                f"Performance Summary - "
                f"CPU: {avg_cpu".1f"}%, "
                f"Memory: {avg_memory".1f"}%, "
                f"Frame Time: {avg_frame_time".3f"}s, "
                f"Efficiency: {avg_efficiency".2f"}"
            )

            self.logger.info(summary)

        except Exception as e:
            self.logger.error(f"Error logging performance summary: {e}")

    def _generate_final_report(self):
        """Generate final performance report."""
        try:
            if not self.metrics_history:
                return

            # Calculate statistics
            metrics_array = np.array([
                [m.cpu_usage, m.memory_usage, m.frame_time, m.efficiency_score]
                for m in self.metrics_history
            ])

            report = {
                'monitoring_duration': time.time() - self.start_time,
                'total_samples': len(self.metrics_history),
                'statistics': {
                    'cpu_usage': {
                        'mean': float(np.mean(metrics_array[:, 0])),
                        'std': float(np.std(metrics_array[:, 0])),
                        'min': float(np.min(metrics_array[:, 0])),
                        'max': float(np.max(metrics_array[:, 0]))
                    },
                    'memory_usage': {
                        'mean': float(np.mean(metrics_array[:, 1])),
                        'std': float(np.std(metrics_array[:, 1])),
                        'min': float(np.min(metrics_array[:, 1])),
                        'max': float(np.max(metrics_array[:, 1]))
                    },
                    'frame_time': {
                        'mean': float(np.mean(metrics_array[:, 2])),
                        'std': float(np.std(metrics_array[:, 2])),
                        'min': float(np.min(metrics_array[:, 2])),
                        'max': float(np.max(metrics_array[:, 2]))
                    },
                    'efficiency_score': {
                        'mean': float(np.mean(metrics_array[:, 3])),
                        'std': float(np.std(metrics_array[:, 3])),
                        'min': float(np.min(metrics_array[:, 3])),
                        'max': float(np.max(metrics_array[:, 3]))
                    }
                },
                'system_health': {
                    'final_status': self.system_health.overall_status,
                    'total_errors': self.system_health.error_count,
                    'total_warnings': self.system_health.warning_count,
                    'uptime': self.system_health.uptime
                }
            }

            # Save report
            report_file = Path("logs/performance_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Final performance report saved to {report_file}")

        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")

    def register_component_monitor(self, component_name: str, monitor_function: callable):
        """
        Register a component-specific monitor function.

        Args:
            component_name: Name of the component
            monitor_function: Function that returns component metrics
        """
        try:
            self.component_monitors[component_name] = monitor_function
            self.logger.info(f"Registered monitor for component: {component_name}")

        except Exception as e:
            self.logger.error(f"Error registering component monitor: {e}")

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent performance metrics."""
        try:
            with self.monitor_lock:
                if self.metrics_history:
                    return self.metrics_history[-1]
                else:
                    return None

        except Exception as e:
            self.logger.error(f"Error getting current metrics: {e}")
            return None

    def get_metrics_history(self, last_n: Optional[int] = None) -> List[PerformanceMetrics]:
        """Get performance metrics history."""
        try:
            with self.monitor_lock:
                if last_n is None:
                    return list(self.metrics_history)
                else:
                    return list(self.metrics_history)[-last_n:]

        except Exception as e:
            self.logger.error(f"Error getting metrics history: {e}")
            return []

    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        return self.system_health

    def export_metrics_csv(self, filename: str):
        """Export metrics history to CSV file."""
        try:
            import csv

            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'cpu_usage', 'memory_usage', 'gpu_usage',
                            'frame_time', 'processing_rate', 'power_consumption',
                            'temperature', 'efficiency_score']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()

                for metrics in self.metrics_history:
                    writer.writerow({
                        'timestamp': metrics.timestamp,
                        'cpu_usage': metrics.cpu_usage,
                        'memory_usage': metrics.memory_usage,
                        'gpu_usage': metrics.gpu_usage,
                        'frame_time': metrics.frame_time,
                        'processing_rate': metrics.processing_rate,
                        'power_consumption': metrics.power_consumption,
                        'temperature': metrics.temperature,
                        'efficiency_score': metrics.efficiency_score
                    })

            self.logger.info(f"Metrics exported to {filename}")

        except Exception as e:
            self.logger.error(f"Error exporting metrics to CSV: {e}")

    def log_component_status(self, component_name: str, status: str, details: str = ""):
        """
        Log status of a specific component.

        Args:
            component_name: Name of the component
            status: Status level (healthy, warning, error, critical)
            details: Additional status details
        """
        try:
            self.system_health.component_status[component_name] = status

            log_message = f"Component {component_name}: {status}"
            if details:
                log_message += f" - {details}"

            if status == "critical":
                self.logger.critical(log_message)
                self.system_health.error_count += 1
                self.system_health.last_error = log_message
            elif status == "error":
                self.logger.error(log_message)
                self.system_health.error_count += 1
                self.system_health.last_error = log_message
            elif status == "warning":
                self.logger.warning(log_message)
                self.system_health.warning_count += 1
            else:
                self.logger.info(log_message)

        except Exception as e:
            self.logger.error(f"Error logging component status: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Create performance monitor
    monitor = PerformanceMonitor()

    # Start monitoring
    if monitor.start_monitoring():
        print("Performance monitoring started")

        # Simulate some operation time
        time.sleep(5.0)

        # Get current metrics
        current_metrics = monitor.get_current_metrics()
        if current_metrics:
            print(f"Current CPU usage: {current_metrics.cpu_usage".1f"}%")
            print(f"Current memory usage: {current_metrics.memory_usage".1f"}%")
            print(f"Current efficiency score: {current_metrics.efficiency_score".2f"}")

        # Get system health
        health = monitor.get_system_health()
        print(f"System status: {health.overall_status}")
        print(f"Uptime: {health.uptime".1f"} seconds")

        # Stop monitoring
        monitor.stop_monitoring()
        print("Performance monitoring stopped")

    else:
        print("Failed to start performance monitoring")