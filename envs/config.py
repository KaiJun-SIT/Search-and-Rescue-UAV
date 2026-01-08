"""Configuration for SAR UAV Environment."""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class EnvConfig:
    """Configuration for the SAR UAV environment.

    Attributes:
        grid_size: Size of the square grid (NxN)
        sensor_radius: How many cells the UAV can observe around itself
        initial_battery: Starting battery level (0-100)
        battery_depletion_rate: Battery units lost per step
        max_steps: Maximum steps before episode terminates
        num_targets: Number of targets to find
        target_priorities: Priority weights for each target (higher = more important)
        obstacle_density: Probability of a cell being an obstacle (0-1)
        detection_probability: P(detect | target in cell) when UAV is on target

        # Reward weights
        reward_target_found: Reward for finding a target (scaled by priority)
        reward_new_coverage: Reward for visiting a new cell
        reward_time_penalty: Penalty applied each step
        reward_battery_depleted: Penalty for running out of battery
        reward_prob_gradient: Reward for moving toward high-probability regions

        # Rendering
        render_mode: "human", "rgb_array", or None
        cell_size: Pixels per grid cell for rendering
    """

    # Grid and observation
    grid_size: int = 10
    sensor_radius: int = 3

    # Battery
    initial_battery: float = 100.0
    battery_depletion_rate: float = 1.0

    # Episode limits
    max_steps: int = 500

    # Targets
    num_targets: int = 1
    target_priorities: Optional[List[float]] = None

    # Environment dynamics
    obstacle_density: float = 0.1
    detection_probability: float = 0.95

    # Rewards
    reward_target_found: float = 1000.0
    reward_new_coverage: float = 1.0
    reward_time_penalty: float = -1.0
    reward_battery_depleted: float = -500.0
    reward_prob_gradient: float = 5.0

    # Rendering
    render_mode: Optional[str] = None
    cell_size: int = 30

    def __post_init__(self) -> None:
        """Validate and set default values."""
        if self.target_priorities is None:
            self.target_priorities = [1.0] * self.num_targets

        if len(self.target_priorities) != self.num_targets:
            raise ValueError(
                f"target_priorities length ({len(self.target_priorities)}) "
                f"must match num_targets ({self.num_targets})"
            )

        if not 0 <= self.obstacle_density <= 1:
            raise ValueError(f"obstacle_density must be in [0, 1], got {self.obstacle_density}")

        if not 0 < self.detection_probability <= 1:
            raise ValueError(
                f"detection_probability must be in (0, 1], got {self.detection_probability}"
            )

        # Set reasonable max_steps based on grid size if using default
        if self.max_steps == 500 and self.grid_size != 10:
            self.max_steps = self.grid_size * self.grid_size * 5

    @property
    def observation_size(self) -> int:
        """Size of the local observation window (always odd)."""
        return 2 * self.sensor_radius + 1
