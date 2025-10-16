"""
Snake Navigation and Path Planning System
Integrates depth vision with servo control for autonomous snake navigation.
"""

import numpy as np
import logging
import time
import threading
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import heapq
import math

class NavigationMode(Enum):
    """Navigation modes for the snake."""
    EXPLORATION = "exploration"
    GOAL_DIRECTED = "goal_directed"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    FOLLOW_PATH = "follow_path"
    SEARCH_PATTERN = "search_pattern"

class TerrainType(Enum):
    """Types of terrain the snake can navigate."""
    FLAT = "flat"
    ROUGH = "rough"
    OBSTRUCTED = "obstructed"
    NARROW = "narrow"
    OPEN = "open"

@dataclass
class NavigationGoal:
    """Goal for navigation."""
    position: Tuple[float, float, float]  # x, y, z coordinates
    orientation: Optional[Tuple[float, float, float]] = None  # yaw, pitch, roll
    priority: float = 1.0
    deadline: Optional[float] = None
    goal_type: str = "position"

@dataclass
class ObstacleInfo:
    """Information about detected obstacles."""
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]  # width, height, depth
    confidence: float
    obstacle_type: str = "unknown"

class PathNode:
    """Node for path planning A* algorithm."""

    def __init__(self, position: Tuple[float, float, float], parent=None):
        self.position = position
        self.parent = parent
        self.g_cost = 0.0  # Cost from start
        self.h_cost = 0.0  # Heuristic cost to goal
        self.f_cost = 0.0  # Total cost

    def __lt__(self, other):
        return self.f_cost < other.f_cost

class SnakeNavigator:
    """
    Navigation system for robotic snake integrating vision and control.
    """

    def __init__(self, snake_controller, depth_system, cpg_system=None, trajectory_generator=None):
        """
        Initialize snake navigator.

        Args:
            snake_controller: Servo controller instance
            depth_system: Depth estimation system
            cpg_system: CPG system for locomotion (optional)
            trajectory_generator: Trajectory generator (optional)
        """
        self.snake_controller = snake_controller
        self.depth_system = depth_system
        self.cpg_system = cpg_system
        self.trajectory_generator = trajectory_generator

        # Navigation state
        self.current_mode = NavigationMode.EXPLORATION
        self.current_goal: Optional[NavigationGoal] = None
        self.current_path: List[Tuple[float, float, float]] = []
        self.snake_position = (0.0, 0.0, 0.0)  # Current estimated position
        self.snake_orientation = (0.0, 0.0, 0.0)  # yaw, pitch, roll

        # Environment understanding
        self.obstacle_map: List[ObstacleInfo] = []
        self.terrain_map = np.zeros((100, 100))  # Simple 2D terrain representation
        self.safety_distance = 0.2  # Minimum safe distance from obstacles (meters)

        # Navigation parameters
        self.max_speed = 0.5  # m/s
        self.turn_rate = np.pi/4  # rad/s
        self.look_ahead_distance = 0.3  # meters

        # Performance tracking
        self.navigation_history = []
        self.collision_count = 0
        self.goal_reached_count = 0

        # Threading
        self.navigation_thread = None
        self.is_navigating = False
        self.navigation_lock = threading.Lock()

        # Setup logging
        self.logger = logging.getLogger('SnakeNavigator')

    def start_navigation(self, mode: NavigationMode = NavigationMode.EXPLORATION,
                        goal: Optional[NavigationGoal] = None) -> bool:
        """
        Start autonomous navigation.

        Args:
            mode: Navigation mode
            goal: Navigation goal (if applicable)

        Returns:
            True if navigation started successfully
        """
        try:
            self.current_mode = mode
            self.current_goal = goal

            if mode == NavigationMode.GOAL_DIRECTED and goal is None:
                self.logger.error("Goal-directed navigation requires a goal")
                return False

            # Start navigation thread
            self.is_navigating = True
            self.navigation_thread = threading.Thread(
                target=self._navigation_loop,
                daemon=True
            )
            self.navigation_thread.start()

            self.logger.info(f"Navigation started in {mode.value} mode")
            return True

        except Exception as e:
            self.logger.error(f"Error starting navigation: {e}")
            return False

    def stop_navigation(self):
        """Stop autonomous navigation."""
        try:
            self.is_navigating = False

            if self.navigation_thread and self.navigation_thread.is_alive:
                self.navigation_thread.join(timeout=2.0)

            # Stop snake motion
            if self.snake_controller:
                self.snake_controller.stop_motion()

            self.logger.info("Navigation stopped")

        except Exception as e:
            self.logger.error(f"Error stopping navigation: {e}")

    def _navigation_loop(self):
        """Main navigation loop."""
        try:
            while self.is_navigating:
                # Update environment understanding
                self._update_environment_map()

                # Execute navigation behavior based on mode
                if self.current_mode == NavigationMode.EXPLORATION:
                    self._execute_exploration()
                elif self.current_mode == NavigationMode.GOAL_DIRECTED:
                    self._execute_goal_directed()
                elif self.current_mode == NavigationMode.OBSTACLE_AVOIDANCE:
                    self._execute_obstacle_avoidance()
                elif self.current_mode == NavigationMode.FOLLOW_PATH:
                    self._execute_path_following()
                elif self.current_mode == NavigationMode.SEARCH_PATTERN:
                    self._execute_search_pattern()

                # Small delay for control loop
                time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error in navigation loop: {e}")
        finally:
            self.is_navigating = False

    def _update_environment_map(self):
        """Update environment map using depth and object detection data."""
        try:
            # Get current depth map and detections
            # This would integrate with the main depth SLAM system
            # For now, simulate environment updates

            # Clear old obstacles
            self.obstacle_map.clear()

            # Simulate obstacle detection (in real implementation, this would come from object detector)
            # For demonstration, create some sample obstacles
            sample_obstacles = [
                ObstacleInfo(position=(0.5, 0.0, 0.1), size=(0.2, 0.2, 0.2), confidence=0.8),
                ObstacleInfo(position=(-0.3, 0.2, 0.1), size=(0.15, 0.15, 0.15), confidence=0.7),
                ObstacleInfo(position=(0.8, -0.1, 0.1), size=(0.1, 0.3, 0.1), confidence=0.9)
            ]

            self.obstacle_map.extend(sample_obstacles)

            # Update terrain map based on depth information
            self._update_terrain_map()

        except Exception as e:
            self.logger.error(f"Error updating environment map: {e}")

    def _update_terrain_map(self):
        """Update 2D terrain representation."""
        try:
            # Simple terrain mapping based on obstacle positions
            self.terrain_map.fill(0)  # Reset terrain map

            for obstacle in self.obstacle_map:
                x, y, z = obstacle.position
                size_x, size_y, _ = obstacle.size

                # Convert world coordinates to map coordinates
                map_x = int((x + 5) * 10)  # Scale and offset
                map_y = int((y + 5) * 10)

                # Mark obstacle area in terrain map
                min_x = max(0, map_x - int(size_x * 10))
                max_x = min(99, map_x + int(size_x * 10))
                min_y = max(0, map_y - int(size_y * 10))
                max_y = min(99, map_y + int(size_y * 10))

                self.terrain_map[min_x:max_x, min_y:max_y] = 1.0

        except Exception as e:
            self.logger.error(f"Error updating terrain map: {e}")

    def _execute_exploration(self):
        """Execute exploration behavior."""
        try:
            # Simple exploration: move forward with random turns
            forward_speed = 0.3

            # Check for obstacles in path
            if self._check_path_clearance():
                # Path is clear, move forward
                self._move_forward(forward_speed)
            else:
                # Obstacle detected, turn to avoid
                self._avoid_obstacle()

        except Exception as e:
            self.logger.error(f"Error in exploration: {e}")

    def _execute_goal_directed(self):
        """Execute goal-directed navigation."""
        try:
            if not self.current_goal:
                return

            # Calculate direction to goal
            goal_x, goal_y, goal_z = self.current_goal.position
            current_x, current_y, current_z = self.snake_position

            # Calculate distance and direction to goal
            dx = goal_x - current_x
            dy = goal_y - current_y
            distance = np.sqrt(dx*dx + dy*dy)

            if distance < 0.1:  # Goal reached
                self._handle_goal_reached()
                return

            # Calculate desired heading
            desired_heading = np.arctan2(dy, dx)

            # Check for obstacles in path
            if self._check_path_clearance(desired_heading):
                # Path is clear, move toward goal
                self._move_toward_goal(desired_heading, min(self.max_speed, distance))
            else:
                # Obstacle in path, find alternative route
                self._find_alternative_path(desired_heading)

        except Exception as e:
            self.logger.error(f"Error in goal-directed navigation: {e}")

    def _execute_obstacle_avoidance(self):
        """Execute obstacle avoidance behavior."""
        try:
            # Focus on avoiding obstacles while maintaining forward motion
            self._avoid_obstacle()

        except Exception as e:
            self.logger.error(f"Error in obstacle avoidance: {e}")

    def _execute_path_following(self):
        """Execute path following behavior."""
        try:
            if not self.current_path:
                # No path to follow, switch to exploration
                self.current_mode = NavigationMode.EXPLORATION
                return

            # Follow the planned path
            self._follow_planned_path()

        except Exception as e:
            self.logger.error(f"Error in path following: {e}")

    def _execute_search_pattern(self):
        """Execute search pattern behavior."""
        try:
            # Implement search patterns like lawnmower, spiral, etc.
            self._execute_lawnmower_pattern()

        except Exception as e:
            self.logger.error(f"Error in search pattern: {e}")

    def _check_path_clearance(self, direction: float = 0.0) -> bool:
        """
        Check if path is clear of obstacles.

        Args:
            direction: Direction to check (radians)

        Returns:
            True if path is clear
        """
        try:
            # Check for obstacles within safety distance in the given direction
            check_distance = self.look_ahead_distance

            for obstacle in self.obstacle_map:
                obs_x, obs_y, obs_z = obstacle.position

                # Calculate distance to obstacle
                distance = np.sqrt((obs_x - self.snake_position[0])**2 +
                                 (obs_y - self.snake_position[1])**2)

                if distance < check_distance + self.safety_distance:
                    # Calculate angle to obstacle
                    angle_to_obstacle = np.arctan2(obs_y - self.snake_position[1],
                                                 obs_x - self.snake_position[0])

                    # Check if obstacle is in the direction we're checking
                    angle_diff = abs(angle_to_obstacle - direction)
                    angle_diff = min(angle_diff, 2*np.pi - angle_diff)

                    if angle_diff < np.pi/4:  # Within 45 degrees
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking path clearance: {e}")
            return True  # Default to clear if error

    def _move_forward(self, speed: float):
        """Move snake forward at given speed."""
        try:
            if self.cpg_system:
                # Use CPG for biologically-inspired locomotion
                self.cpg_system.set_gait_pattern(GaitPattern.SERPENTINE)
                # Adjust CPG frequency based on speed
                self.cpg_system.parameters.frequency = speed * 2.0

            elif self.trajectory_generator:
                # Use trajectory generator for smooth motion
                trajectory = self.trajectory_generator.generate_snake_trajectory(
                    TrajectoryType.SERPENTINE,
                    duration=1.0,
                    amplitude=0.5,
                    frequency=speed
                )
                self.snake_controller.execute_trajectory(trajectory)

        except Exception as e:
            self.logger.error(f"Error moving forward: {e}")

    def _move_toward_goal(self, heading: float, speed: float):
        """Move toward goal at given heading and speed."""
        try:
            # Calculate turn required
            current_heading = self.snake_orientation[0]  # yaw
            turn_angle = heading - current_heading
            turn_angle = (turn_angle + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

            # Execute turn if needed
            if abs(turn_angle) > 0.1:  # Minimum turn threshold
                self._execute_turn(turn_angle)

            # Move forward
            self._move_forward(speed)

        except Exception as e:
            self.logger.error(f"Error moving toward goal: {e}")

    def _avoid_obstacle(self):
        """Execute obstacle avoidance maneuver."""
        try:
            # Find nearest obstacle
            nearest_obstacle = None
            min_distance = float('inf')

            for obstacle in self.obstacle_map:
                distance = np.sqrt(
                    (obstacle.position[0] - self.snake_position[0])**2 +
                    (obstacle.position[1] - self.snake_position[1])**2
                )

                if distance < min_distance:
                    min_distance = distance
                    nearest_obstacle = obstacle

            if nearest_obstacle and min_distance < self.safety_distance * 2:
                # Calculate avoidance direction
                obs_x, obs_y, _ = nearest_obstacle.position
                avoidance_heading = np.arctan2(self.snake_position[1] - obs_y,
                                             self.snake_position[0] - obs_x)

                # Turn away from obstacle
                turn_angle = avoidance_heading - self.snake_orientation[0]
                turn_angle = (turn_angle + np.pi) % (2 * np.pi) - np.pi

                self._execute_turn(turn_angle * 0.5)  # Partial turn

                # Move at reduced speed
                self._move_forward(self.max_speed * 0.5)

        except Exception as e:
            self.logger.error(f"Error avoiding obstacle: {e}")

    def _execute_turn(self, turn_angle: float):
        """Execute turning maneuver."""
        try:
            # Calculate joint angles for turning
            turn_radius = 0.5  # meters
            joint_angles = []

            for i in range(self.snake_controller.num_joints):
                # Create turning wave pattern
                phase = (2 * np.pi * i) / self.snake_controller.num_joints
                angle_offset = np.sin(phase) * turn_angle * (1 - i / self.snake_controller.num_joints)
                joint_angles.append(angle_offset)

            # Apply turn for short duration
            self.snake_controller.set_joint_angles(joint_angles)
            time.sleep(0.2)

        except Exception as e:
            self.logger.error(f"Error executing turn: {e}")

    def _find_alternative_path(self, desired_heading: float):
        """Find alternative path when direct path is blocked."""
        try:
            # Simple path planning: try left and right deviations
            deviations = [-np.pi/3, np.pi/3]  # 60 degrees left and right

            for deviation in deviations:
                alternative_heading = desired_heading + deviation

                if self._check_path_clearance(alternative_heading):
                    # Found clear alternative path
                    self._move_toward_goal(alternative_heading, self.max_speed * 0.7)
                    return

            # If no clear path found, turn in place
            self._execute_turn(np.pi/2)  # 90 degree turn

        except Exception as e:
            self.logger.error(f"Error finding alternative path: {e}")

    def _follow_planned_path(self):
        """Follow pre-planned path."""
        try:
            if not self.current_path:
                return

            # Get next waypoint
            if len(self.current_path) > 0:
                next_waypoint = self.current_path[0]

                # Check if waypoint reached
                distance = np.sqrt(
                    (next_waypoint[0] - self.snake_position[0])**2 +
                    (next_waypoint[1] - self.snake_position[1])**2
                )

                if distance < 0.1:
                    # Waypoint reached, remove from path
                    self.current_path.pop(0)
                    return

                # Move toward waypoint
                heading = np.arctan2(
                    next_waypoint[1] - self.snake_position[1],
                    next_waypoint[0] - self.snake_position[0]
                )
                self._move_toward_goal(heading, self.max_speed)

        except Exception as e:
            self.logger.error(f"Error following planned path: {e}")

    def _execute_lawnmower_pattern(self):
        """Execute lawnmower search pattern."""
        try:
            # Simple back-and-forth search pattern
            static_heading = time.time() * 0.1  # Slow rotation

            # Move in current direction
            self._move_toward_goal(static_heading % (2*np.pi), self.max_speed * 0.6)

        except Exception as e:
            self.logger.error(f"Error executing lawnmower pattern: {e}")

    def _handle_goal_reached(self):
        """Handle when navigation goal is reached."""
        try:
            self.goal_reached_count += 1

            # Log success
            self.navigation_history.append({
                'timestamp': time.time(),
                'event': 'goal_reached',
                'position': self.snake_position,
                'goal': self.current_goal.position if self.current_goal else None
            })

            self.logger.info(f"Navigation goal reached! Total goals: {self.goal_reached_count}")

            # Stop or set new goal
            if self.current_goal and self.current_goal.goal_type == "position":
                self.stop_navigation()

        except Exception as e:
            self.logger.error(f"Error handling goal reached: {e}")

    def plan_path(self, start: Tuple[float, float, float],
                  goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Plan path from start to goal using A* algorithm.

        Args:
            start: Start position (x, y, z)
            goal: Goal position (x, y, z)

        Returns:
            List of waypoints forming the path
        """
        try:
            # Simple A* path planning in 2D (ignoring z for now)
            start_2d = (start[0], start[1])
            goal_2d = (goal[0], goal[1])

            # Create grid for path planning
            grid_size = 0.1  # meters per cell
            grid_width = int(10 / grid_size)  # 10m x 10m area
            grid_height = int(10 / grid_size)

            # Convert positions to grid coordinates
            start_grid = (
                int((start_2d[0] + 5) / grid_size),
                int((start_2d[1] + 5) / grid_size)
            )
            goal_grid = (
                int((goal_2d[0] + 5) / grid_size),
                int((goal_2d[1] + 5) / grid_size)
            )

            # Create obstacle grid
            obstacle_grid = np.zeros((grid_width, grid_height))
            for obstacle in self.obstacle_map:
                obs_x, obs_y, _ = obstacle.position
                obs_grid_x = int((obs_x + 5) / grid_size)
                obs_grid_y = int((obs_y + 5) / grid_size)

                # Mark obstacle cells
                min_x = max(0, obs_grid_x - 2)
                max_x = min(grid_width - 1, obs_grid_x + 2)
                min_y = max(0, obs_grid_y - 2)
                max_y = min(grid_height - 1, obs_grid_y + 2)

                obstacle_grid[min_x:max_x, min_y:max_y] = 1

            # A* path planning
            path = self._a_star_pathfinding(
                obstacle_grid, start_grid, goal_grid, grid_size
            )

            # Convert back to world coordinates
            world_path = []
            for grid_x, grid_y in path:
                world_x = (grid_x * grid_size) - 5
                world_y = (grid_y * grid_size) - 5
                world_path.append((world_x, world_y, 0.0))

            return world_path

        except Exception as e:
            self.logger.error(f"Error planning path: {e}")
            return []

    def _a_star_pathfinding(self, grid: np.ndarray, start: Tuple[int, int],
                          goal: Tuple[int, int], cell_size: float) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm."""
        try:
            rows, cols = grid.shape
            open_set = []
            closed_set = set()

            start_node = PathNode(start)
            start_node.h_cost = self._heuristic(start, goal)
            start_node.f_cost = start_node.g_cost + start_node.h_cost

            heapq.heappush(open_set, start_node)

            directions = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),           (0, 1),
                         (1, -1),  (1, 0), (1, 1)]

            while open_set:
                current_node = heapq.heappop(open_set)

                if current_node.position == goal:
                    # Reconstruct path
                    path = []
                    while current_node:
                        path.append(current_node.position)
                        current_node = current_node.parent
                    return path[::-1]

                closed_set.add(current_node.position)

                for dx, dy in directions:
                    new_x = current_node.position[0] + dx
                    new_y = current_node.position[1] + dy

                    if (0 <= new_x < rows and 0 <= new_y < cols and
                        grid[new_x, new_y] == 0 and
                        (new_x, new_y) not in closed_set):

                        neighbor = PathNode((new_x, new_y), current_node)
                        neighbor.g_cost = current_node.g_cost + np.sqrt(dx*dx + dy*dy)
                        neighbor.h_cost = self._heuristic((new_x, new_y), goal)
                        neighbor.f_cost = neighbor.g_cost + neighbor.h_cost

                        # Check if better path found
                        if neighbor not in open_set:
                            heapq.heappush(open_set, neighbor)

            return []  # No path found

        except Exception as e:
            self.logger.error(f"Error in A* pathfinding: {e}")
            return []

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Heuristic function for A* (Manhattan distance)."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def set_goal(self, goal: NavigationGoal):
        """Set navigation goal."""
        self.current_goal = goal
        self.current_mode = NavigationMode.GOAL_DIRECTED

        self.logger.info(f"Navigation goal set: {goal.position}")

    def set_path(self, path: List[Tuple[float, float, float]]):
        """Set path to follow."""
        self.current_path = path
        self.current_mode = NavigationMode.FOLLOW_PATH

        self.logger.info(f"Path set with {len(path)} waypoints")

    def get_navigation_status(self) -> Dict[str, Any]:
        """Get current navigation status."""
        return {
            'mode': self.current_mode.value,
            'is_navigating': self.is_navigating,
            'current_goal': self.current_goal.__dict__ if self.current_goal else None,
            'snake_position': self.snake_position,
            'snake_orientation': self.snake_orientation,
            'obstacle_count': len(self.obstacle_map),
            'collision_count': self.collision_count,
            'goal_reached_count': self.goal_reached_count,
            'current_path_length': len(self.current_path)
        }

# Example usage
if __name__ == "__main__":
    # This would be used with actual snake controller and depth system
    print("Snake Navigator - Integration module for autonomous navigation")
    print("This module integrates depth vision with servo control for autonomous snake movement")