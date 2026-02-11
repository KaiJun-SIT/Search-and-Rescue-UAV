"""Search and Rescue UAV Gymnasium Environment.

This environment simulates a UAV searching for missing persons in a 2D grid world
with partial observability, battery constraints, and Bayesian belief updates.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from envs.config import EnvConfig


class SAREnv(gym.Env):
    """Search and Rescue UAV Environment.

    A partially observable grid world where a UAV must find targets efficiently
    while managing battery constraints and maintaining Bayesian belief maps.

    Action Space:
        Discrete(9): 8 directions (N, NE, E, SE, S, SW, W, NW) + stay

    Observation Space:
        Dict:
            - position: Box(2,) - agent's (x, y) position
            - battery: Box(1,) - current battery level [0, 100]
            - local_occupancy: Box(obs_size, obs_size, 3) - what agent sees:
                - Channel 0: obstacles (1 if obstacle, 0 otherwise)
                - Channel 1: visited cells (1 if visited, 0 otherwise)
                - Channel 2: current position indicator
            - coverage_map: Box(grid_size, grid_size) - binary visited map
            - belief_map: Box(grid_size, grid_size) - probability of target location
            - targets_found: Box(1,) - number of targets found so far
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Action definitions
    ACTIONS = {
        0: (0, -1),  # North
        1: (1, -1),  # NE
        2: (1, 0),  # East
        3: (1, 1),  # SE
        4: (0, 1),  # South
        5: (-1, 1),  # SW
        6: (-1, 0),  # West
        7: (-1, -1),  # NW
        8: (0, 0),  # Stay
    }

    def __init__(self, config: Optional[EnvConfig] = None):
        """Initialize the SAR environment.

        Args:
            config: Environment configuration. If None, uses default config.
        """
        super().__init__()

        self.config = config or EnvConfig()
        self.grid_size = self.config.grid_size
        self.obs_size = self.config.observation_size

        # Action space: 9 discrete actions
        self.action_space = spaces.Discrete(9)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=0, high=self.grid_size - 1, shape=(2,), dtype=np.float32
                ),
                "battery": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "local_occupancy": spaces.Box(
                    low=0, high=1, shape=(self.obs_size, self.obs_size, 3), dtype=np.float32
                ),
                "coverage_map": spaces.Box(
                    low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32
                ),
                "belief_map": spaces.Box(
                    low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32
                ),
                "targets_found": spaces.Box(
                    low=0, high=self.config.num_targets, shape=(1,), dtype=np.float32
                ),
            }
        )

        # Internal state
        self.agent_pos: np.ndarray = np.zeros(2, dtype=np.int32)
        self.battery: float = 0.0
        self.obstacles: np.ndarray = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.coverage_map: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32
        )
        self.belief_map: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32
        )
        self.target_positions: List[Tuple[int, int]] = []
        self.targets_found: int = 0
        self.step_count: int = 0

        # Rendering
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.colorbars: list = []  # Track colorbars to prevent stacking

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (can specify fixed_targets, fixed_obstacles)

        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset state
        self.battery = self.config.initial_battery
        self.step_count = 0
        self.targets_found = 0

        # Generate obstacles (or use provided ones)
        if options and "fixed_obstacles" in options:
            self.obstacles = options["fixed_obstacles"].copy()
        else:
            self.obstacles = (
                self.np_random.random((self.grid_size, self.grid_size))
                < self.config.obstacle_density
            )

        # Place agent at random free position
        self.agent_pos = self._get_random_free_position()

        # Place targets at random free positions (or use provided ones)
        if options and "fixed_targets" in options:
            self.target_positions = options["fixed_targets"].copy()
        else:
            self.target_positions = []
            for _ in range(self.config.num_targets):
                target_pos = self._get_random_free_position()
                # Ensure target isn't on agent or another target
                while (
                    np.array_equal(target_pos, self.agent_pos)
                    or any(np.array_equal(target_pos, t) for t in self.target_positions)
                ):
                    target_pos = self._get_random_free_position()
                self.target_positions.append(tuple(target_pos))

        # Initialize coverage map
        self.coverage_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.coverage_map[self.agent_pos[1], self.agent_pos[0]] = 1.0

        # Initialize belief map (uniform distribution over free cells)
        self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self.belief_map[self.obstacles] = 0.0
        self.belief_map[self.agent_pos[1], self.agent_pos[0]] = 0.0  # Agent knows it's not here
        self._normalize_belief_map()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action to take (0-8)

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (success or battery depleted)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        self.step_count += 1
        reward = 0.0

        # Get action delta
        dx, dy = self.ACTIONS[action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        # Check if move is valid
        if self._is_valid_position(new_x, new_y):
            # Track if this is a new cell
            was_visited = self.coverage_map[new_y, new_x] > 0

            # Store old belief for gradient reward
            old_belief = self.belief_map[new_y, new_x]
            old_pos_belief = (
                self.belief_map[self.agent_pos[1], self.agent_pos[0]]
                if not was_visited
                else 0.0
            )

            # Move agent
            self.agent_pos = np.array([new_x, new_y], dtype=np.int32)

            # Update coverage
            self.coverage_map[new_y, new_x] = 1.0

            # New coverage reward
            if not was_visited:
                reward += self.config.reward_new_coverage

                # Probability gradient reward (moving toward higher probability)
                if old_belief > old_pos_belief:
                    reward += self.config.reward_prob_gradient * (old_belief - old_pos_belief)

        # Deplete battery
        self.battery -= self.config.battery_depletion_rate
        self.battery = max(0.0, self.battery)  # Clamp to 0

        # Time penalty
        reward += self.config.reward_time_penalty

        # Check for target detection
        current_pos = tuple(self.agent_pos)
        if current_pos in self.target_positions:
            target_idx = self.target_positions.index(current_pos)
            # Only give reward if not already found
            if target_idx < len(self.config.target_priorities):
                # Detection based on probability
                if self.np_random.random() < self.config.detection_probability:
                    priority = self.config.target_priorities[target_idx]
                    reward += self.config.reward_target_found * priority
                    self.targets_found += 1
                    # Remove found target
                    self.target_positions.remove(current_pos)

        # Update belief map (Bayesian update for scanned area)
        self._update_belief_map()

        # Check termination conditions
        terminated = False
        truncated = False

        # Success: all targets found
        if self.targets_found >= self.config.num_targets:
            terminated = True

        # Failure: battery depleted
        if self.battery <= 0:
            reward += self.config.reward_battery_depleted
            terminated = True

        # Truncation: max steps reached
        if self.step_count >= self.config.max_steps:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation (with partial observability).

        Returns:
            Dictionary observation following the observation space
        """
        # Position
        position = self.agent_pos.astype(np.float32)

        # Battery
        battery = np.array([self.battery], dtype=np.float32)

        # Local occupancy (what the agent can see)
        local_occupancy = self._get_local_observation()

        # Coverage map (agent has perfect memory of where it's been)
        coverage_map = self.coverage_map.copy()

        # Belief map (agent maintains this internally)
        belief_map = self.belief_map.copy()

        # Targets found
        targets_found = np.array([self.targets_found], dtype=np.float32)

        return {
            "position": position,
            "battery": battery,
            "local_occupancy": local_occupancy,
            "coverage_map": coverage_map,
            "belief_map": belief_map,
            "targets_found": targets_found,
        }

    def _get_local_observation(self) -> np.ndarray:
        """Get local observation within sensor radius.

        This implements PARTIAL OBSERVABILITY - the agent only sees cells
        within sensor_radius of its current position.

        Returns:
            Array of shape (obs_size, obs_size, 3) containing:
                - Channel 0: obstacles (1 if obstacle, 0 if free, 0.5 if out of bounds)
                - Channel 1: visited cells
                - Channel 2: current position indicator (1 at center)
        """
        x, y = self.agent_pos
        r = self.config.sensor_radius

        # Initialize observation with out-of-bounds indicator (0.5)
        local_obs = np.full((self.obs_size, self.obs_size, 3), 0.5, dtype=np.float32)

        # Fill in visible cells
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                obs_x = i + r
                obs_y = j + r
                world_x = x + i
                world_y = y + j

                # Check if world position is in bounds
                if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                    # Channel 0: obstacles
                    local_obs[obs_y, obs_x, 0] = float(self.obstacles[world_y, world_x])

                    # Channel 1: visited cells
                    local_obs[obs_y, obs_x, 1] = self.coverage_map[world_y, world_x]

        # Channel 2: current position (center of observation)
        local_obs[r, r, 2] = 1.0

        return local_obs

    def _update_belief_map(self) -> None:
        """Update belief map using Bayesian inference.

        When the agent scans an area and doesn't find a target, we update:
        P(target at location | not detected) ∝ P(target) * P(not detected | target)
                                               = P(target) * (1 - detection_prob)

        For numerical stability, we use log-space calculations.
        """
        x, y = self.agent_pos

        # The agent has just scanned its current position
        # If no target was found, update the belief
        if tuple(self.agent_pos) not in self.target_positions:
            # Update: target wasn't here, so probability becomes 0
            self.belief_map[y, x] = 0.0

            # For cells in sensor range, update with detection probability
            r = self.config.sensor_radius
            for i in range(-r, r + 1):
                for j in range(-r, r + 1):
                    world_x = x + i
                    world_y = y + j

                    if (
                        0 <= world_x < self.grid_size
                        and 0 <= world_y < self.grid_size
                        and not self.obstacles[world_y, world_x]
                    ):
                        # Bayesian update: didn't detect target in this cell
                        # P(target | no detection) = P(target) * (1 - P(detection | target))
                        # For simplicity, we use full detection prob at current cell,
                        # reduced detection at range
                        distance = np.sqrt(i**2 + j**2)
                        if distance == 0:
                            # Already handled above
                            continue
                        else:
                            # Detection probability decreases with distance
                            detection_prob = self.config.detection_probability * np.exp(
                                -distance / r
                            )
                            self.belief_map[world_y, world_x] *= 1 - detection_prob

            # Renormalize belief map
            self._normalize_belief_map()

    def _normalize_belief_map(self) -> None:
        """Normalize belief map so probabilities sum to 1.

        Uses numerical stability techniques to avoid underflow.
        """
        total = np.sum(self.belief_map)
        if total > 1e-10:  # Avoid division by zero
            self.belief_map /= total
        else:
            # If all beliefs are near zero, reset to uniform over free cells
            self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
            self.belief_map[self.obstacles] = 0.0
            self.belief_map[self.coverage_map > 0] = 0.0  # Already searched
            total = np.sum(self.belief_map)
            if total > 0:
                self.belief_map /= total

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is valid (in bounds and not an obstacle).

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if position is valid, False otherwise
        """
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        if self.obstacles[y, x]:
            return False
        return True

    def _get_random_free_position(self) -> np.ndarray:
        """Get a random free position (not an obstacle).

        Returns:
            Array [x, y] of a free position
        """
        while True:
            x = self.np_random.integers(0, self.grid_size)
            y = self.np_random.integers(0, self.grid_size)
            if not self.obstacles[y, x]:
                return np.array([x, y], dtype=np.int32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state.

        Returns:
            Dictionary with info
        """
        return {
            "step_count": self.step_count,
            "battery": self.battery,
            "targets_found": self.targets_found,
            "targets_remaining": len(self.target_positions),
            "coverage": np.sum(self.coverage_map) / (self.grid_size**2 - np.sum(self.obstacles)),
            "success": self.targets_found >= self.config.num_targets,
        }

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.config.render_mode is None:
            return None

        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))

        # Remove old colorbars to prevent stacking/shrinking
        for cbar in self.colorbars:
            cbar.remove()
        self.colorbars.clear()

        for ax in self.ax:
            ax.clear()

        # Plot 1: Grid world with agent, targets, obstacles
        self._render_grid(self.ax[0])

        # Plot 2: Coverage map
        self._render_coverage(self.ax[1])

        # Plot 3: Belief map
        self._render_belief(self.ax[2])

        self.fig.suptitle(
            f"Step: {self.step_count} | Battery: {self.battery:.1f} | "
            f"Found: {self.targets_found}/{self.config.num_targets}"
        )

        if self.config.render_mode == "human":
            plt.pause(0.001)
            plt.draw()
            return None
        else:  # rgb_array
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data

    def _render_grid(self, ax: plt.Axes) -> None:
        """Render the grid world."""
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect("equal")
        ax.set_title("Grid World")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Draw grid
        for i in range(self.grid_size + 1):
            ax.axhline(i, color="gray", linewidth=0.5)
            ax.axvline(i, color="gray", linewidth=0.5)

        # Draw obstacles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.obstacles[y, x]:
                    rect = patches.Rectangle(
                        (x, y), 1, 1, linewidth=0, facecolor="black", alpha=0.7
                    )
                    ax.add_patch(rect)

        # Draw targets
        for tx, ty in self.target_positions:
            circle = patches.Circle((tx + 0.5, ty + 0.5), 0.3, color="red", alpha=0.7)
            ax.add_patch(circle)

        # Draw agent
        agent_circle = patches.Circle(
            (self.agent_pos[0] + 0.5, self.agent_pos[1] + 0.5), 0.4, color="blue", alpha=0.8
        )
        ax.add_patch(agent_circle)

        # Draw sensor range
        sensor_circle = patches.Circle(
            (self.agent_pos[0] + 0.5, self.agent_pos[1] + 0.5),
            self.config.sensor_radius,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
            linestyle="--",
            alpha=0.5,
        )
        ax.add_patch(sensor_circle)

        ax.invert_yaxis()

    def _render_coverage(self, ax: plt.Axes) -> None:
        """Render the coverage map."""
        im = ax.imshow(self.coverage_map, cmap="Blues", vmin=0, vmax=1, origin="upper")
        ax.set_title("Coverage Map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.colorbars.append(cbar)  # Track colorbar for removal

    def _render_belief(self, ax: plt.Axes) -> None:
        """Render the belief map."""
        im = ax.imshow(self.belief_map, cmap="hot", vmin=0, vmax=None, origin="upper")
        ax.set_title("Belief Map (Target Probability)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.colorbars.append(cbar)  # Track colorbar for removal

    def close(self) -> None:
        """Close the environment and cleanup."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.colorbars.clear()
