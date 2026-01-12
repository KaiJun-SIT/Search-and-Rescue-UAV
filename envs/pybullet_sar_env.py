"""SAR Environment with Gym-Pybullet-Drones integration.

Provides physics-based drone simulation using the Gym-Pybullet-Drones library.
This allows for realistic aerodynamics, wind effects, and hardware validation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Any, Tuple, List

try:
    from gym_pybullet_drones.envs.BaseAviary import BaseAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    BaseAviary = object  # Fallback

from envs.config import EnvConfig


class PyBulletSAREnv(gym.Env if not PYBULLET_AVAILABLE else BaseAviary):
    """SAR Environment with PyBullet physics simulation.

    Integrates SAR search logic with realistic drone physics from
    gym-pybullet-drones. Supports:
    - Realistic aerodynamics
    - Wind disturbances
    - Battery consumption based on thrust
    - Multiple drone models (CF2X, CF2P, etc.)

    Note: Requires gym-pybullet-drones to be installed:
        pip install gym-pybullet-drones
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 48}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        drone_model: str = "cf2x",
        physics: str = "pyb",
        gui: bool = False,
        record: bool = False,
    ):
        """Initialize PyBullet SAR environment.

        Args:
            config: SAR environment configuration
            drone_model: Drone model ("cf2x", "cf2p", "hb")
            physics: Physics engine ("pyb", "dyn", "pyb_gnd", "pyb_drag", "pyb_dw", "pyb_gnd_drag_dw")
            gui: Whether to show PyBullet GUI
            record: Whether to record video
        """
        if not PYBULLET_AVAILABLE:
            raise ImportError(
                "gym-pybullet-drones is not installed. "
                "Install it with: pip install gym-pybullet-drones"
            )

        self.config = config or EnvConfig()
        self.grid_size = self.config.grid_size
        self.search_area_size = 10.0  # meters

        # Map drone model string to enum
        drone_models = {
            "cf2x": DroneModel.CF2X,
            "cf2p": DroneModel.CF2P,
            "hb": DroneModel.HB,
        }

        # Map physics string to enum
        physics_types = {
            "pyb": Physics.PYB,
            "dyn": Physics.DYN,
            "pyb_gnd": Physics.PYB_GND,
            "pyb_drag": Physics.PYB_DRAG,
            "pyb_dw": Physics.PYB_DW,
            "pyb_gnd_drag_dw": Physics.PYB_GND_DRAG_DW,
        }

        # Initialize BaseAviary
        super().__init__(
            drone_model=drone_models.get(drone_model, DroneModel.CF2X),
            num_drones=1,
            neighbourhood_radius=10,
            initial_xyzs=np.array([[0, 0, 1]]),  # Start at 1m altitude
            initial_rpys=np.array([[0, 0, 0]]),
            physics=physics_types.get(physics, Physics.PYB),
            freq=240,
            aggregate_phy_steps=1,
            gui=gui,
            record=record,
        )

        # SAR-specific state
        self.grid_resolution = self.search_area_size / self.grid_size
        self.target_positions: List[np.ndarray] = []  # 3D positions in meters
        self.targets_found: int = 0
        self.coverage_map: np.ndarray = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.belief_map: np.ndarray = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self.step_count: int = 0
        self.battery_level: float = 100.0

        # Define high-level action space (waypoint navigation)
        self.action_space = spaces.Discrete(9)  # 8 directions + hover

        # Observation space (similar to 2D SAR but with physics state)
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "orientation": spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
                "battery": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
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

        # Waypoint controller state
        self.current_waypoint: Optional[np.ndarray] = None
        self.waypoint_reached_threshold = 0.2  # meters

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment."""
        # Reset PyBullet simulation
        obs = super().reset(seed=seed)

        # Reset SAR state
        self.step_count = 0
        self.targets_found = 0
        self.battery_level = 100.0
        self.coverage_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)

        # Place targets
        self.target_positions = []
        for _ in range(self.config.num_targets):
            target = np.array(
                [
                    np.random.uniform(-self.search_area_size / 2, self.search_area_size / 2),
                    np.random.uniform(-self.search_area_size / 2, self.search_area_size / 2),
                    0.0,  # Ground level
                ]
            )
            self.target_positions.append(target)

        self._normalize_belief_map()

        # Get SAR observation
        sar_obs = self._get_sar_observation()
        info = self._get_info()

        return sar_obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute high-level action (waypoint navigation)."""
        self.step_count += 1
        reward = 0.0

        # Convert discrete action to waypoint
        waypoint = self._action_to_waypoint(action)
        self.current_waypoint = waypoint

        # Execute low-level control to reach waypoint
        # (simplified - in practice you'd use a PID controller)
        for _ in range(10):  # 10 physics steps per high-level step
            low_level_action = self._compute_control(waypoint)
            obs, _, _, _, _ = super().step(low_level_action)

            # Check if waypoint reached
            drone_pos = self._getDroneStateVector(0)[0:3]
            if np.linalg.norm(drone_pos[:2] - waypoint[:2]) < self.waypoint_reached_threshold:
                break

        # Update coverage
        drone_pos = self._getDroneStateVector(0)[0:3]
        grid_x, grid_y = self._world_to_grid(drone_pos[:2])
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            if self.coverage_map[grid_y, grid_x] == 0:
                self.coverage_map[grid_y, grid_x] = 1.0
                reward += self.config.reward_new_coverage

        # Check for target detection
        for target_pos in self.target_positions[:]:
            distance = np.linalg.norm(drone_pos[:2] - target_pos[:2])
            if distance < 1.0:  # 1 meter detection range
                if np.random.random() < self.config.detection_probability:
                    idx = self.target_positions.index(target_pos)
                    priority = self.config.target_priorities[idx]
                    reward += self.config.reward_target_found * priority
                    self.targets_found += 1
                    self.target_positions.remove(target_pos)

        # Update belief map
        self._update_belief_map(drone_pos)

        # Battery consumption (based on thrust)
        # Simplified: consume more battery for fast movements
        velocity = self._getDroneStateVector(0)[10:13]
        speed = np.linalg.norm(velocity)
        battery_consumption = 1.0 + speed * 0.5  # Base + speed penalty
        self.battery_level -= battery_consumption
        self.battery_level = max(0.0, self.battery_level)

        # Time penalty
        reward += self.config.reward_time_penalty

        # Termination
        terminated = False
        truncated = False

        if self.targets_found >= self.config.num_targets:
            terminated = True

        if self.battery_level <= 0:
            reward += self.config.reward_battery_depleted
            terminated = True

        if self.step_count >= self.config.max_steps:
            truncated = True

        sar_obs = self._get_sar_observation()
        info = self._get_info()

        return sar_obs, reward, terminated, truncated, info

    def _action_to_waypoint(self, action: int) -> np.ndarray:
        """Convert discrete action to waypoint."""
        # Get current position
        drone_pos = self._getDroneStateVector(0)[0:3]

        # Action mapping
        action_map = {
            0: (0, 1),  # North
            1: (1, 1),  # NE
            2: (1, 0),  # East
            3: (1, -1),  # SE
            4: (0, -1),  # South
            5: (-1, -1),  # SW
            6: (-1, 0),  # West
            7: (-1, -1),  # NW
            8: (0, 0),  # Hover
        }

        dx, dy = action_map[action]
        step_size = self.grid_resolution

        waypoint = drone_pos.copy()
        waypoint[0] += dx * step_size
        waypoint[1] += dy * step_size
        waypoint[2] = 1.0  # Maintain altitude

        return waypoint

    def _compute_control(self, target_waypoint: np.ndarray) -> np.ndarray:
        """Compute low-level control to reach waypoint.

        This is a simplified controller. For real deployment, use a proper
        PID or MPC controller.
        """
        drone_state = self._getDroneStateVector(0)
        current_pos = drone_state[0:3]
        current_vel = drone_state[10:13]

        # Simple proportional controller
        pos_error = target_waypoint - current_pos
        desired_vel = pos_error * 2.0  # Proportional gain

        # Velocity control (simplified)
        vel_error = desired_vel - current_vel

        # Convert to RPM commands (very simplified)
        # In practice, use proper attitude control
        rpm = np.array([16000, 16000, 16000, 16000])  # Hover RPMs

        # Add corrections based on vel_error (simplified)
        rpm[0] += vel_error[0] * 100
        rpm[1] -= vel_error[0] * 100
        rpm[2] += vel_error[1] * 100
        rpm[3] -= vel_error[1] * 100

        # Add altitude correction
        rpm += vel_error[2] * 200

        # Clamp RPMs
        rpm = np.clip(rpm, 0, self.MAX_RPM)

        return rpm.reshape(1, 4)

    def _world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        # Shift from [-size/2, size/2] to [0, size]
        shifted = world_pos + self.search_area_size / 2

        # Scale to grid
        grid_x = int(shifted[0] / self.grid_resolution)
        grid_y = int(shifted[1] / self.grid_resolution)

        return grid_x, grid_y

    def _update_belief_map(self, drone_pos: np.ndarray) -> None:
        """Update belief map based on observation."""
        grid_x, grid_y = self._world_to_grid(drone_pos[:2])

        # Update surrounding cells
        sensor_radius = self.config.sensor_radius
        for i in range(-sensor_radius, sensor_radius + 1):
            for j in range(-sensor_radius, sensor_radius + 1):
                gx = grid_x + i
                gy = grid_y + j

                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    distance = np.sqrt(i**2 + j**2)
                    detection_prob = self.config.detection_probability * np.exp(
                        -distance / sensor_radius
                    )
                    self.belief_map[gy, gx] *= 1 - detection_prob

        self._normalize_belief_map()

    def _normalize_belief_map(self) -> None:
        """Normalize belief map."""
        total = np.sum(self.belief_map)
        if total > 1e-10:
            self.belief_map /= total

    def _get_sar_observation(self) -> Dict[str, np.ndarray]:
        """Get SAR-specific observation."""
        drone_state = self._getDroneStateVector(0)

        return {
            "position": drone_state[0:3].astype(np.float32),
            "velocity": drone_state[10:13].astype(np.float32),
            "orientation": drone_state[7:10].astype(np.float32),
            "battery": np.array([self.battery_level], dtype=np.float32),
            "coverage_map": self.coverage_map.copy(),
            "belief_map": self.belief_map.copy(),
            "targets_found": np.array([self.targets_found], dtype=np.float32),
        }

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        return {
            "step_count": self.step_count,
            "battery": self.battery_level,
            "targets_found": self.targets_found,
            "targets_remaining": len(self.target_positions),
            "success": self.targets_found >= self.config.num_targets,
        }


def create_pybullet_sar_env(
    config: Optional[EnvConfig] = None,
    drone_model: str = "cf2x",
    physics: str = "pyb",
    gui: bool = False,
) -> PyBulletSAREnv:
    """Factory function to create PyBullet SAR environment.

    Args:
        config: SAR environment configuration
        drone_model: Drone model to simulate
        physics: Physics engine to use
        gui: Whether to show GUI

    Returns:
        PyBullet SAR environment

    Example:
        >>> config = EnvConfig(grid_size=10, num_targets=1)
        >>> env = create_pybullet_sar_env(config, gui=True)
        >>> obs, info = env.reset()
        >>> for _ in range(100):
        ...     action = env.action_space.sample()
        ...     obs, reward, done, truncated, info = env.step(action)
        ...     if done or truncated:
        ...         break
    """
    return PyBulletSAREnv(
        config=config,
        drone_model=drone_model,
        physics=physics,
        gui=gui,
    )
