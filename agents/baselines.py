"""Baseline search algorithms for SAR comparison.

These traditional search algorithms serve as baselines to compare against
RL-based methods. They don't use learning, just deterministic or heuristic strategies.
"""

import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class BaselineAgent(ABC):
    """Abstract base class for baseline search agents."""

    def __init__(self, grid_size: int, sensor_radius: int = 3):
        """Initialize baseline agent.

        Args:
            grid_size: Size of the search grid
            sensor_radius: Observation radius of the agent
        """
        self.grid_size = grid_size
        self.sensor_radius = sensor_radius
        self.reset()

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.visited: np.ndarray = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.current_pos: Optional[Tuple[int, int]] = None

    @abstractmethod
    def select_action(
        self, observation: dict, obstacles: Optional[np.ndarray] = None
    ) -> int:
        """Select an action based on the current observation.

        Args:
            observation: Environment observation dict
            obstacles: Optional obstacle map for path planning

        Returns:
            Action index (0-8)
        """
        pass

    def _get_action_to_target(
        self, current_pos: Tuple[int, int], target_pos: Tuple[int, int]
    ) -> int:
        """Get action that moves toward target position.

        Args:
            current_pos: Current (x, y) position
            target_pos: Target (x, y) position

        Returns:
            Action index that moves toward target
        """
        dx = np.sign(target_pos[0] - current_pos[0])
        dy = np.sign(target_pos[1] - current_pos[1])

        # Map (dx, dy) to action
        action_map = {
            (0, -1): 0,  # N
            (1, -1): 1,  # NE
            (1, 0): 2,  # E
            (1, 1): 3,  # SE
            (0, 1): 4,  # S
            (-1, 1): 5,  # SW
            (-1, 0): 6,  # W
            (-1, -1): 7,  # NW
            (0, 0): 8,  # Stay
        }

        return action_map.get((dx, dy), 8)


class GridSearchAgent(BaselineAgent):
    """Grid search - systematic row-by-row coverage.

    This is the traditional SAR pattern: cover the area systematically
    in a row-by-row manner (like a lawnmower pattern).
    """

    def __init__(self, grid_size: int, sensor_radius: int = 3):
        super().__init__(grid_size, sensor_radius)
        self.path: List[Tuple[int, int]] = []
        self.path_index: int = 0

    def reset(self) -> None:
        """Reset and generate grid search path."""
        super().reset()
        self.path = self._generate_grid_path()
        self.path_index = 0

    def _generate_grid_path(self) -> List[Tuple[int, int]]:
        """Generate systematic grid search path.

        Returns:
            List of (x, y) positions covering the grid
        """
        path = []
        for y in range(self.grid_size):
            if y % 2 == 0:
                # Left to right
                for x in range(self.grid_size):
                    path.append((x, y))
            else:
                # Right to left
                for x in range(self.grid_size - 1, -1, -1):
                    path.append((x, y))
        return path

    def select_action(
        self, observation: dict, obstacles: Optional[np.ndarray] = None
    ) -> int:
        """Select action following grid search path.

        Args:
            observation: Environment observation
            obstacles: Obstacle map (used to skip blocked cells)

        Returns:
            Action toward next grid position
        """
        # Get current position
        pos = observation["position"]
        current_pos = (int(pos[0]), int(pos[1]))
        self.current_pos = current_pos

        # Skip to next non-obstacle target in path
        while self.path_index < len(self.path):
            target = self.path[self.path_index]

            # Check if target is an obstacle
            if obstacles is not None and obstacles[target[1], target[0]]:
                self.path_index += 1
                continue

            # If we've reached this waypoint, move to next
            if current_pos == target:
                self.path_index += 1
                continue

            # Move toward target
            return self._get_action_to_target(current_pos, target)

        # Finished path - stay in place
        return 8


class SpiralSearchAgent(BaselineAgent):
    """Spiral search - search in expanding spiral from center.

    Often used when target is believed to be near a central point (e.g., last known position).
    """

    def __init__(self, grid_size: int, sensor_radius: int = 3, center: Optional[Tuple[int, int]] = None):
        super().__init__(grid_size, sensor_radius)
        self.center = center or (grid_size // 2, grid_size // 2)
        self.path: List[Tuple[int, int]] = []
        self.path_index: int = 0

    def reset(self) -> None:
        """Reset and generate spiral path."""
        super().reset()
        self.path = self._generate_spiral_path()
        self.path_index = 0

    def _generate_spiral_path(self) -> List[Tuple[int, int]]:
        """Generate spiral search path from center.

        Returns:
            List of (x, y) positions in spiral order
        """
        path = [self.center]
        x, y = self.center
        dx, dy = 1, 0  # Start moving right

        steps_in_direction = 1
        steps_taken = 0
        direction_changes = 0

        max_steps = self.grid_size * self.grid_size

        while len(path) < max_steps:
            x += dx
            y += dy

            # Add if in bounds
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                path.append((x, y))

            steps_taken += 1

            # Change direction?
            if steps_taken == steps_in_direction:
                steps_taken = 0
                direction_changes += 1

                # Rotate 90 degrees counter-clockwise
                dx, dy = -dy, dx

                # Increase step count every 2 direction changes
                if direction_changes % 2 == 0:
                    steps_in_direction += 1

        return path

    def select_action(
        self, observation: dict, obstacles: Optional[np.ndarray] = None
    ) -> int:
        """Select action following spiral path.

        Args:
            observation: Environment observation
            obstacles: Obstacle map

        Returns:
            Action toward next spiral position
        """
        pos = observation["position"]
        current_pos = (int(pos[0]), int(pos[1]))
        self.current_pos = current_pos

        while self.path_index < len(self.path):
            target = self.path[self.path_index]

            if obstacles is not None and obstacles[target[1], target[0]]:
                self.path_index += 1
                continue

            if current_pos == target:
                self.path_index += 1
                continue

            return self._get_action_to_target(current_pos, target)

        return 8


class RandomSearchAgent(BaselineAgent):
    """Random search - randomly explore the grid.

    This serves as a lower baseline - any decent algorithm should beat random search.
    """

    def __init__(self, grid_size: int, sensor_radius: int = 3, seed: Optional[int] = None):
        super().__init__(grid_size, sensor_radius)
        self.rng = np.random.RandomState(seed)

    def select_action(
        self, observation: dict, obstacles: Optional[np.ndarray] = None
    ) -> int:
        """Select random valid action.

        Args:
            observation: Environment observation
            obstacles: Obstacle map

        Returns:
            Random action (0-8)
        """
        # Try random actions until we find a valid one
        pos = observation["position"]
        current_x, current_y = int(pos[0]), int(pos[1])

        valid_actions = []
        action_deltas = [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, 0),
        ]

        for action, (dx, dy) in enumerate(action_deltas):
            new_x = current_x + dx
            new_y = current_y + dy

            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                if obstacles is None or not obstacles[new_y, new_x]:
                    valid_actions.append(action)

        if valid_actions:
            return self.rng.choice(valid_actions)
        return 8  # Stay if no valid actions


class ProbabilityWeightedAgent(BaselineAgent):
    """Probability-weighted search - move toward high-probability regions.

    Uses the belief map to guide search toward most likely target locations.
    This is a greedy heuristic baseline.
    """

    def __init__(self, grid_size: int, sensor_radius: int = 3):
        super().__init__(grid_size, sensor_radius)

    def select_action(
        self, observation: dict, obstacles: Optional[np.ndarray] = None
    ) -> int:
        """Select action toward highest probability region.

        Args:
            observation: Environment observation with belief_map
            obstacles: Obstacle map

        Returns:
            Action toward highest probability cell
        """
        pos = observation["position"]
        current_x, current_y = int(pos[0]), int(pos[1])
        belief_map = observation["belief_map"]

        # Find highest probability unvisited cell
        coverage = observation["coverage_map"]
        unvisited_belief = belief_map.copy()
        unvisited_belief[coverage > 0] = 0  # Zero out visited cells

        if obstacles is not None:
            unvisited_belief[obstacles] = 0

        # Find cell with highest belief
        if np.max(unvisited_belief) > 0:
            max_indices = np.argwhere(unvisited_belief == np.max(unvisited_belief))
            target_idx = max_indices[0]  # Take first max if multiple
            target_pos = (target_idx[1], target_idx[0])  # (x, y)

            return self._get_action_to_target((current_x, current_y), target_pos)

        # If no unvisited cells, explore randomly
        return RandomSearchAgent(self.grid_size, self.sensor_radius).select_action(
            observation, obstacles
        )


def create_baseline_agent(
    agent_type: str, grid_size: int, sensor_radius: int = 3, **kwargs
) -> BaselineAgent:
    """Factory function to create baseline agents.

    Args:
        agent_type: Type of agent ("grid", "spiral", "random", "probability")
        grid_size: Size of search grid
        sensor_radius: Observation radius
        **kwargs: Additional agent-specific parameters

    Returns:
        Baseline agent instance

    Raises:
        ValueError: If agent_type is unknown
    """
    agents = {
        "grid": GridSearchAgent,
        "spiral": SpiralSearchAgent,
        "random": RandomSearchAgent,
        "probability": ProbabilityWeightedAgent,
    }

    if agent_type not in agents:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Choose from {list(agents.keys())}"
        )

    return agents[agent_type](grid_size, sensor_radius, **kwargs)
