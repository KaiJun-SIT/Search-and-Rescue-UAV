"""3D Search and Rescue UAV Gymnasium Environment.

Extends the 2D SAR environment to 3D with altitude control, making it more
realistic for actual UAV operations. The drone can now move in 3D space and
must manage altitude along with horizontal position.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from envs.config import EnvConfig


class SAREnv3D(gym.Env):
    """3D Search and Rescue UAV Environment.

    Extension of the 2D SAR environment with altitude control. The UAV now operates
    in a 3D grid (X, Y, Z) where Z represents altitude levels.

    Action Space:
        Discrete(27): 26 directions (3D grid neighbors) + hover
        - 9 actions at current altitude (N, NE, E, SE, S, SW, W, NW, stay)
        - 9 actions at altitude+1 (climb while moving)
        - 9 actions at altitude-1 (descend while moving)

    Observation Space:
        Dict:
            - position: Box(3,) - agent's (x, y, z) position
            - battery: Box(1,) - current battery level [0, 100]
            - local_occupancy: Box(obs_size, obs_size, obs_size, 4) - 3D view
                - Channel 0: obstacles (buildings, trees)
                - Channel 1: visited cells
                - Channel 2: altitude layers
                - Channel 3: current position
            - coverage_map: Box(grid_size, grid_size, max_altitude) - 3D visited map
            - belief_map: Box(grid_size, grid_size) - 2D target probability (ground level)
            - targets_found: Box(1,) - number of targets found
            - altitude: Box(1,) - current altitude level
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        max_altitude: int = 5,
        altitude_sensor_range: int = 2,
    ):
        """Initialize 3D SAR environment.

        Args:
            config: Environment configuration
            max_altitude: Maximum altitude level (0 = ground)
            altitude_sensor_range: Vertical sensor range
        """
        super().__init__()

        self.config = config or EnvConfig()
        self.grid_size = self.config.grid_size
        self.obs_size = self.config.observation_size
        self.max_altitude = max_altitude
        self.altitude_sensor_range = altitude_sensor_range

        # 3D action space: 27 actions (3x3x3 grid of movements)
        self.action_space = spaces.Discrete(27)

        # Extended observation space with altitude
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([0, 0, 0]),
                    high=np.array([self.grid_size - 1, self.grid_size - 1, max_altitude]),
                    dtype=np.float32,
                ),
                "battery": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "local_occupancy": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.obs_size, self.obs_size, altitude_sensor_range * 2 + 1, 4),
                    dtype=np.float32,
                ),
                "coverage_map": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.grid_size, self.grid_size, max_altitude + 1),
                    dtype=np.float32,
                ),
                "belief_map": spaces.Box(
                    low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32
                ),
                "targets_found": spaces.Box(
                    low=0, high=self.config.num_targets, shape=(1,), dtype=np.float32
                ),
                "altitude": spaces.Box(low=0, high=max_altitude, shape=(1,), dtype=np.float32),
            }
        )

        # Generate 3D action mapping
        self._generate_action_map()

        # Internal state
        self.agent_pos: np.ndarray = np.zeros(3, dtype=np.int32)  # (x, y, z)
        self.battery: float = 0.0
        self.obstacles_3d: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size, max_altitude + 1), dtype=bool
        )
        self.coverage_map_3d: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size, max_altitude + 1), dtype=np.float32
        )
        self.belief_map: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32
        )
        self.target_positions: List[Tuple[int, int]] = []  # Ground-level targets
        self.targets_found: int = 0
        self.step_count: int = 0

        # Rendering
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[Axes3D] = None

    def _generate_action_map(self) -> None:
        """Generate 3D action mapping: 27 actions for 3x3x3 movement."""
        self.actions_3d = {}
        idx = 0

        for dz in [-1, 0, 1]:  # Altitude change
            for dy in [-1, 0, 1]:  # North-South
                for dx in [-1, 0, 1]:  # East-West
                    self.actions_3d[idx] = (dx, dy, dz)
                    idx += 1

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the 3D environment."""
        super().reset(seed=seed)

        # Reset state
        self.battery = self.config.initial_battery
        self.step_count = 0
        self.targets_found = 0

        # Generate 3D obstacles (buildings, trees at various heights)
        self.obstacles_3d = self._generate_3d_obstacles()

        # Place agent at random free position at medium altitude
        self.agent_pos = self._get_random_free_position_3d()

        # Place targets on ground level
        if options and "fixed_targets" in options:
            self.target_positions = options["fixed_targets"].copy()
        else:
            self.target_positions = []
            for _ in range(self.config.num_targets):
                target_pos = self._get_random_ground_position()
                while any(np.array_equal(target_pos, t) for t in self.target_positions):
                    target_pos = self._get_random_ground_position()
                self.target_positions.append(tuple(target_pos))

        # Initialize 3D coverage map
        self.coverage_map_3d = np.zeros(
            (self.grid_size, self.grid_size, self.max_altitude + 1), dtype=np.float32
        )
        self.coverage_map_3d[self.agent_pos[1], self.agent_pos[0], self.agent_pos[2]] = 1.0

        # Initialize 2D belief map (ground level)
        self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self.belief_map[self.obstacles_3d[:, :, 0]] = 0.0  # Ground obstacles
        self._normalize_belief_map()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _generate_3d_obstacles(self) -> np.ndarray:
        """Generate 3D obstacles (buildings at various heights)."""
        obstacles = np.zeros((self.grid_size, self.grid_size, self.max_altitude + 1), dtype=bool)

        # Generate buildings with random heights
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.np_random.random() < self.config.obstacle_density:
                    # Random building height
                    height = self.np_random.integers(1, self.max_altitude + 1)
                    obstacles[y, x, :height] = True

        return obstacles

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in 3D."""
        self.step_count += 1
        reward = 0.0

        # Get 3D action delta
        dx, dy, dz = self.actions_3d[action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        new_z = self.agent_pos[2] + dz

        # Check if move is valid
        if self._is_valid_position_3d(new_x, new_y, new_z):
            # Track if new cell
            was_visited = self.coverage_map_3d[new_y, new_x, new_z] > 0

            # Move agent
            self.agent_pos = np.array([new_x, new_y, new_z], dtype=np.int32)

            # Update coverage
            self.coverage_map_3d[new_y, new_x, new_z] = 1.0

            # Coverage reward
            if not was_visited:
                reward += self.config.reward_new_coverage

            # Extra battery cost for altitude changes
            if dz != 0:
                self.battery -= abs(dz) * 0.5  # Climbing/descending costs extra

        # Deplete battery
        self.battery -= self.config.battery_depletion_rate
        self.battery = max(0.0, self.battery)

        # Time penalty
        reward += self.config.reward_time_penalty

        # Check for target detection (can detect from any altitude with sensor)
        # Detection range decreases with altitude
        detection_range = max(1, self.config.sensor_radius - self.agent_pos[2])
        for target_pos in self.target_positions[:]:
            distance = np.sqrt(
                (self.agent_pos[0] - target_pos[0]) ** 2
                + (self.agent_pos[1] - target_pos[1]) ** 2
            )

            if distance <= detection_range:
                # Altitude affects detection probability
                altitude_penalty = 1.0 - (self.agent_pos[2] / self.max_altitude) * 0.5
                detection_prob = self.config.detection_probability * altitude_penalty

                if self.np_random.random() < detection_prob:
                    target_idx = self.target_positions.index(target_pos)
                    if target_idx < len(self.config.target_priorities):
                        priority = self.config.target_priorities[target_idx]
                        reward += self.config.reward_target_found * priority
                        self.targets_found += 1
                        self.target_positions.remove(target_pos)

        # Update belief map
        self._update_belief_map_3d()

        # Termination
        terminated = False
        truncated = False

        if self.targets_found >= self.config.num_targets:
            terminated = True

        if self.battery <= 0:
            reward += self.config.reward_battery_depleted
            terminated = True

        if self.step_count >= self.config.max_steps:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get 3D observation."""
        position = self.agent_pos.astype(np.float32)
        battery = np.array([self.battery], dtype=np.float32)
        altitude = np.array([self.agent_pos[2]], dtype=np.float32)

        # 3D local occupancy
        local_occupancy = self._get_local_observation_3d()

        # 3D coverage map
        coverage_map = self.coverage_map_3d.copy()

        # 2D belief map
        belief_map = self.belief_map.copy()

        targets_found = np.array([self.targets_found], dtype=np.float32)

        return {
            "position": position,
            "battery": battery,
            "local_occupancy": local_occupancy,
            "coverage_map": coverage_map,
            "belief_map": belief_map,
            "targets_found": targets_found,
            "altitude": altitude,
        }

    def _get_local_observation_3d(self) -> np.ndarray:
        """Get 3D local observation within sensor range."""
        x, y, z = self.agent_pos
        r = self.config.sensor_radius
        r_alt = self.altitude_sensor_range

        # Initialize with out-of-bounds indicator
        local_obs = np.full(
            (self.obs_size, self.obs_size, r_alt * 2 + 1, 4), 0.5, dtype=np.float32
        )

        # Fill visible cells
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                for k in range(-r_alt, r_alt + 1):
                    obs_x = i + r
                    obs_y = j + r
                    obs_z = k + r_alt

                    world_x = x + i
                    world_y = y + j
                    world_z = z + k

                    if (
                        0 <= world_x < self.grid_size
                        and 0 <= world_y < self.grid_size
                        and 0 <= world_z <= self.max_altitude
                    ):
                        # Channel 0: obstacles
                        local_obs[obs_y, obs_x, obs_z, 0] = float(
                            self.obstacles_3d[world_y, world_x, world_z]
                        )

                        # Channel 1: visited
                        local_obs[obs_y, obs_x, obs_z, 1] = self.coverage_map_3d[
                            world_y, world_x, world_z
                        ]

                        # Channel 2: altitude layer indicator
                        local_obs[obs_y, obs_x, obs_z, 2] = world_z / self.max_altitude

        # Channel 3: current position
        local_obs[r, r, r_alt, 3] = 1.0

        return local_obs

    def _update_belief_map_3d(self) -> None:
        """Update belief map based on 3D sensor observations."""
        x, y, z = self.agent_pos

        # Detection range based on altitude
        detection_range = max(1, self.config.sensor_radius - z // 2)

        for i in range(-detection_range, detection_range + 1):
            for j in range(-detection_range, detection_range + 1):
                world_x = x + i
                world_y = y + j

                if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                    # Altitude affects detection probability
                    distance = np.sqrt(i**2 + j**2)
                    altitude_factor = 1.0 - (z / self.max_altitude) * 0.5

                    if distance <= detection_range:
                        detection_prob = (
                            self.config.detection_probability
                            * altitude_factor
                            * np.exp(-distance / detection_range)
                        )
                        self.belief_map[world_y, world_x] *= 1 - detection_prob

        self._normalize_belief_map()

    def _normalize_belief_map(self) -> None:
        """Normalize belief map."""
        total = np.sum(self.belief_map)
        if total > 1e-10:
            self.belief_map /= total
        else:
            self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
            self.belief_map[self.obstacles_3d[:, :, 0]] = 0.0
            total = np.sum(self.belief_map)
            if total > 0:
                self.belief_map /= total

    def _is_valid_position_3d(self, x: int, y: int, z: int) -> bool:
        """Check if 3D position is valid."""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size and 0 <= z <= self.max_altitude):
            return False
        if self.obstacles_3d[y, x, z]:
            return False
        return True

    def _get_random_free_position_3d(self) -> np.ndarray:
        """Get random free 3D position."""
        while True:
            x = self.np_random.integers(0, self.grid_size)
            y = self.np_random.integers(0, self.grid_size)
            z = self.np_random.integers(1, self.max_altitude + 1)  # Start above ground

            if not self.obstacles_3d[y, x, z]:
                return np.array([x, y, z], dtype=np.int32)

    def _get_random_ground_position(self) -> Tuple[int, int]:
        """Get random ground position for target."""
        while True:
            x = self.np_random.integers(0, self.grid_size)
            y = self.np_random.integers(0, self.grid_size)
            if not self.obstacles_3d[y, x, 0]:
                return (x, y)

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        return {
            "step_count": self.step_count,
            "battery": self.battery,
            "targets_found": self.targets_found,
            "targets_remaining": len(self.target_positions),
            "altitude": self.agent_pos[2],
            "success": self.targets_found >= self.config.num_targets,
        }

    def render(self) -> Optional[np.ndarray]:
        """Render 3D environment."""
        if self.config.render_mode is None:
            return None

        if self.fig is None:
            self.fig = plt.figure(figsize=(15, 5))
            self.ax = [
                self.fig.add_subplot(131, projection="3d"),
                self.fig.add_subplot(132),
                self.fig.add_subplot(133),
            ]

        for ax in self.ax:
            if hasattr(ax, "clear"):
                ax.clear()

        self._render_3d_view(self.ax[0])
        self._render_top_view(self.ax[1])
        self._render_belief(self.ax[2])

        self.fig.suptitle(
            f"Step: {self.step_count} | Alt: {self.agent_pos[2]} | "
            f"Battery: {self.battery:.1f} | Found: {self.targets_found}/{self.config.num_targets}"
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

    def _render_3d_view(self, ax) -> None:
        """Render 3D visualization."""
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_zlim(0, self.max_altitude)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Altitude")
        ax.set_title("3D View")

        # Plot obstacles
        obs_coords = np.where(self.obstacles_3d)
        if len(obs_coords[0]) > 0:
            ax.scatter(
                obs_coords[1], obs_coords[0], obs_coords[2], c="gray", marker="s", alpha=0.3, s=20
            )

        # Plot targets
        for tx, ty in self.target_positions:
            ax.scatter([tx], [ty], [0], c="red", marker="*", s=200)

        # Plot agent
        ax.scatter(
            [self.agent_pos[0]], [self.agent_pos[1]], [self.agent_pos[2]], c="blue", marker="o", s=100
        )

    def _render_top_view(self, ax) -> None:
        """Render top-down coverage view."""
        coverage_2d = np.max(self.coverage_map_3d, axis=2)
        im = ax.imshow(coverage_2d, cmap="Blues", vmin=0, vmax=1, origin="upper")
        ax.set_title("Coverage (Top View)")
        ax.plot(self.agent_pos[0], self.agent_pos[1], "bo", markersize=10)
        for tx, ty in self.target_positions:
            ax.plot(tx, ty, "r*", markersize=15)

    def _render_belief(self, ax) -> None:
        """Render belief map."""
        im = ax.imshow(self.belief_map, cmap="hot", origin="upper")
        ax.set_title("Belief Map")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def close(self) -> None:
        """Close environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
