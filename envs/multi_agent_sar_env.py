"""Multi-Agent SAR Environment.

Coordinates multiple UAVs to search for targets more efficiently. Implements:
- Decentralized multi-agent RL
- Communication between agents
- Territory division
- Collaborative target search
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from envs.config import EnvConfig


class MultiAgentSAREnv(gym.Env):
    """Multi-Agent Search and Rescue Environment.

    Coordinates N UAVs to collaboratively search for targets. Agents can:
    - Share discovered information
    - Divide search territory
    - Avoid collisions
    - Coordinate target priorities

    Observation Space (per agent):
        Dict:
            - position: Own position
            - battery: Own battery level
            - local_occupancy: What agent sees
            - coverage_map: Shared coverage (all agents contribute)
            - belief_map: Shared belief map
            - targets_found: Total targets found by team
            - agent_positions: Positions of other agents (communication)
            - agent_batteries: Battery levels of other agents

    Action Space (per agent):
        Discrete(9): 8 directions + stay
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: Optional[EnvConfig] = None, num_agents: int = 3):
        """Initialize multi-agent SAR environment.

        Args:
            config: Environment configuration
            num_agents: Number of UAV agents
        """
        super().__init__()

        self.config = config or EnvConfig()
        self.grid_size = self.config.grid_size
        self.obs_size = self.config.observation_size
        self.num_agents = num_agents

        # Action space per agent
        self.action_space = spaces.Tuple([spaces.Discrete(9) for _ in range(num_agents)])

        # Observation space per agent (includes info about other agents)
        single_agent_obs = spaces.Dict(
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
                "agent_positions": spaces.Box(
                    low=0, high=self.grid_size - 1, shape=(num_agents, 2), dtype=np.float32
                ),
                "agent_batteries": spaces.Box(
                    low=0, high=100, shape=(num_agents,), dtype=np.float32
                ),
            }
        )

        self.observation_space = spaces.Tuple([single_agent_obs for _ in range(num_agents)])

        # Action mappings
        self.ACTIONS = {
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

        # Multi-agent state
        self.agent_positions: List[np.ndarray] = []
        self.agent_batteries: List[float] = []
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

        # Collision tracking
        self.collision_penalty: float = -100.0

        # Rendering
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
        """Reset multi-agent environment."""
        super().reset(seed=seed)

        # Reset state
        self.step_count = 0
        self.targets_found = 0

        # Generate obstacles
        self.obstacles = (
            self.np_random.random((self.grid_size, self.grid_size))
            < self.config.obstacle_density
        )

        # Place agents at different starting positions
        self.agent_positions = []
        self.agent_batteries = []

        for i in range(self.num_agents):
            pos = self._get_random_free_position()
            # Ensure agents don't start on same cell
            while any(np.array_equal(pos, p) for p in self.agent_positions):
                pos = self._get_random_free_position()
            self.agent_positions.append(pos)
            self.agent_batteries.append(self.config.initial_battery)

        # Place targets
        self.target_positions = []
        for _ in range(self.config.num_targets):
            target_pos = self._get_random_free_position()
            while any(np.array_equal(target_pos, t) for t in self.target_positions):
                target_pos = self._get_random_free_position()
            self.target_positions.append(tuple(target_pos))

        # Initialize shared maps
        self.coverage_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for pos in self.agent_positions:
            self.coverage_map[pos[1], pos[0]] = 1.0

        self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self.belief_map[self.obstacles] = 0.0
        self._normalize_belief_map()

        observations = self._get_observations()
        info = self._get_info()

        return observations, info

    def step(
        self, actions: Tuple[int, ...]
    ) -> Tuple[
        List[Dict[str, np.ndarray]], List[float], List[bool], List[bool], Dict[str, Any]
    ]:
        """Execute multi-agent step.

        Args:
            actions: Tuple of actions, one per agent

        Returns:
            observations: List of observations per agent
            rewards: List of rewards per agent
            terminated: List of termination flags per agent
            truncated: List of truncation flags per agent
            info: Shared info dictionary
        """
        self.step_count += 1
        rewards = [0.0] * self.num_agents
        terminated = [False] * self.num_agents
        truncated = [False] * self.num_agents

        # Track positions before movement for collision detection
        old_positions = [pos.copy() for pos in self.agent_positions]

        # Execute actions for all agents
        for i, action in enumerate(actions):
            if self.agent_batteries[i] <= 0:
                terminated[i] = True
                continue

            # Convert numpy action to Python int for dictionary lookup
            action = int(action)

            # Get action delta
            dx, dy = self.ACTIONS[action]
            new_x = self.agent_positions[i][0] + dx
            new_y = self.agent_positions[i][1] + dy

            # Check validity
            if self._is_valid_position(new_x, new_y):
                was_visited = self.coverage_map[new_y, new_x] > 0

                # Move agent
                self.agent_positions[i] = np.array([new_x, new_y], dtype=np.int32)

                # Update coverage
                self.coverage_map[new_y, new_x] = 1.0

                # Coverage reward (team reward)
                if not was_visited:
                    rewards[i] += self.config.reward_new_coverage

            # Deplete battery
            self.agent_batteries[i] -= self.config.battery_depletion_rate
            self.agent_batteries[i] = max(0.0, self.agent_batteries[i])

            # Time penalty
            rewards[i] += self.config.reward_time_penalty

        # Check for collisions (agents on same cell)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.array_equal(self.agent_positions[i], self.agent_positions[j]):
                    rewards[i] += self.collision_penalty
                    rewards[j] += self.collision_penalty

        # Check for target detection (any agent can detect)
        for target_pos in self.target_positions[:]:
            for i in range(self.num_agents):
                agent_pos = self.agent_positions[i]
                distance = np.linalg.norm(agent_pos - np.array(target_pos))

                if distance <= self.config.sensor_radius:
                    if self.np_random.random() < self.config.detection_probability:
                        target_idx = self.target_positions.index(target_pos)
                        if target_idx < len(self.config.target_priorities):
                            priority = self.config.target_priorities[target_idx]
                            # Team reward - all agents get the reward
                            team_reward = self.config.reward_target_found * priority
                            for j in range(self.num_agents):
                                rewards[j] += team_reward

                            self.targets_found += 1
                            self.target_positions.remove(target_pos)
                            break  # Target found, move to next

        # Update belief map from all agent observations
        self._update_belief_map_multiagent()

        # Check termination conditions
        # Mission success (all targets found)
        if self.targets_found >= self.config.num_targets:
            terminated = [True] * self.num_agents

        # Individual agent failures
        for i in range(self.num_agents):
            if self.agent_batteries[i] <= 0:
                rewards[i] += self.config.reward_battery_depleted
                terminated[i] = True

        # Episode truncation
        if self.step_count >= self.config.max_steps:
            truncated = [True] * self.num_agents

        observations = self._get_observations()
        info = self._get_info()

        return observations, rewards, terminated, truncated, info

    def _get_observations(self) -> List[Dict[str, np.ndarray]]:
        """Get observations for all agents."""
        observations = []

        for i in range(self.num_agents):
            obs = {
                "position": self.agent_positions[i].astype(np.float32),
                "battery": np.array([self.agent_batteries[i]], dtype=np.float32),
                "local_occupancy": self._get_local_observation(i),
                "coverage_map": self.coverage_map.copy(),
                "belief_map": self.belief_map.copy(),
                "targets_found": np.array([self.targets_found], dtype=np.float32),
                "agent_positions": np.array(self.agent_positions, dtype=np.float32),
                "agent_batteries": np.array(self.agent_batteries, dtype=np.float32),
            }
            observations.append(obs)

        return observations

    def _get_local_observation(self, agent_idx: int) -> np.ndarray:
        """Get local observation for specific agent."""
        x, y = self.agent_positions[agent_idx]
        r = self.config.sensor_radius

        local_obs = np.full((self.obs_size, self.obs_size, 3), 0.5, dtype=np.float32)

        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                obs_x = i + r
                obs_y = j + r
                world_x = x + i
                world_y = y + j

                if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                    # Channel 0: obstacles
                    local_obs[obs_y, obs_x, 0] = float(self.obstacles[world_y, world_x])

                    # Channel 1: coverage
                    local_obs[obs_y, obs_x, 1] = self.coverage_map[world_y, world_x]

                    # Channel 2: other agents
                    for j_agent, other_pos in enumerate(self.agent_positions):
                        if j_agent != agent_idx and np.array_equal(
                            other_pos, [world_x, world_y]
                        ):
                            local_obs[obs_y, obs_x, 2] = 1.0

        return local_obs

    def _update_belief_map_multiagent(self) -> None:
        """Update belief map considering all agents' observations."""
        for agent_pos in self.agent_positions:
            x, y = agent_pos
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
                        distance = np.sqrt(i**2 + j**2)
                        if distance <= r:
                            detection_prob = self.config.detection_probability * np.exp(
                                -distance / r
                            )
                            self.belief_map[world_y, world_x] *= 1 - detection_prob

        self._normalize_belief_map()

    def _normalize_belief_map(self) -> None:
        """Normalize belief map."""
        total = np.sum(self.belief_map)
        if total > 1e-10:
            self.belief_map /= total

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is valid."""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        if self.obstacles[y, x]:
            return False
        return True

    def _get_random_free_position(self) -> np.ndarray:
        """Get random free position."""
        while True:
            x = self.np_random.integers(0, self.grid_size)
            y = self.np_random.integers(0, self.grid_size)
            if not self.obstacles[y, x]:
                return np.array([x, y], dtype=np.int32)

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        return {
            "step_count": self.step_count,
            "targets_found": self.targets_found,
            "targets_remaining": len(self.target_positions),
            "coverage": np.sum(self.coverage_map) / (self.grid_size**2 - np.sum(self.obstacles)),
            "success": self.targets_found >= self.config.num_targets,
            "agent_batteries": self.agent_batteries.copy(),
            "active_agents": sum(1 for b in self.agent_batteries if b > 0),
        }

    def render(self) -> Optional[np.ndarray]:
        """Render multi-agent environment."""
        if self.config.render_mode is None:
            return None

        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 5))

        for ax in self.ax:
            ax.clear()

        # Plot 1: Grid world
        self._render_grid(self.ax[0])

        # Plot 2: Belief map
        self._render_belief(self.ax[1])

        self.fig.suptitle(
            f"Step: {self.step_count} | Agents: {self.num_agents} | "
            f"Found: {self.targets_found}/{self.config.num_targets}"
        )

        if self.config.render_mode == "human":
            plt.pause(0.001)
            plt.draw()
            return None
        else:
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data

    def _render_grid(self, ax: plt.Axes) -> None:
        """Render grid world."""
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect("equal")
        ax.set_title("Multi-Agent SAR Grid")

        # Grid lines
        for i in range(self.grid_size + 1):
            ax.axhline(i, color="gray", linewidth=0.5)
            ax.axvline(i, color="gray", linewidth=0.5)

        # Obstacles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.obstacles[y, x]:
                    rect = patches.Rectangle((x, y), 1, 1, facecolor="black", alpha=0.7)
                    ax.add_patch(rect)

        # Coverage
        coverage_overlay = self.coverage_map.copy()
        coverage_overlay[coverage_overlay == 0] = np.nan
        ax.imshow(
            coverage_overlay, cmap="Blues", alpha=0.3, vmin=0, vmax=1, origin="upper", extent=[0, self.grid_size, self.grid_size, 0]
        )

        # Targets
        for tx, ty in self.target_positions:
            circle = patches.Circle((tx + 0.5, ty + 0.5), 0.3, color="red", alpha=0.7)
            ax.add_patch(circle)

        # Agents (different colors)
        agent_colors = ["blue", "green", "orange", "purple", "cyan"]
        for i, pos in enumerate(self.agent_positions):
            color = agent_colors[i % len(agent_colors)]
            circle = patches.Circle(
                (pos[0] + 0.5, pos[1] + 0.5), 0.4, color=color, alpha=0.8
            )
            ax.add_patch(circle)
            # Add agent ID
            ax.text(pos[0] + 0.5, pos[1] + 0.5, str(i), ha="center", va="center", color="white", fontweight="bold")

        ax.invert_yaxis()

    def _render_belief(self, ax: plt.Axes) -> None:
        """Render belief map."""
        im = ax.imshow(self.belief_map, cmap="hot", origin="upper")
        ax.set_title("Shared Belief Map")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def close(self) -> None:
        """Close environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
