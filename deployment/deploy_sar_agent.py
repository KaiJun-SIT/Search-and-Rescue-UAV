"""Deploy trained SAR agent on real drone.

This script loads a trained RL model and deploys it on a real drone
for actual SAR missions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import time
from typing import Optional, Tuple
from stable_baselines3 import PPO, SAC, TD3

from envs import EnvConfig
from deployment.drone_interface import create_drone_interface, DroneInterface


class SARDeployment:
    """Deploy trained SAR agent on real drone."""

    def __init__(
        self,
        model_path: str,
        algorithm: str,
        drone_interface: DroneInterface,
        config: EnvConfig,
        grid_to_world_scale: float = 2.0,  # meters per grid cell
    ):
        """Initialize deployment.

        Args:
            model_path: Path to trained model
            algorithm: Algorithm type ("PPO", "SAC", "TD3")
            drone_interface: Drone interface instance
            config: Environment configuration
            grid_to_world_scale: Meters per grid cell
        """
        self.config = config
        self.drone = drone_interface
        self.grid_to_world_scale = grid_to_world_scale

        # Load trained model
        print(f"Loading {algorithm} model from {model_path}")
        if algorithm == "PPO":
            self.model = PPO.load(model_path)
        elif algorithm == "SAC":
            self.model = SAC.load(model_path)
        elif algorithm == "TD3":
            self.model = TD3.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Initialize state
        self.grid_position = np.array([0, 0], dtype=np.int32)
        self.coverage_map = np.zeros((config.grid_size, config.grid_size), dtype=np.float32)
        self.belief_map = np.ones((config.grid_size, config.grid_size), dtype=np.float32)
        self.belief_map /= np.sum(self.belief_map)

        self.targets_found = 0
        self.step_count = 0

        # Action mapping (same as SAREnv)
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

    def grid_to_world(self, grid_pos: np.ndarray) -> Tuple[float, float]:
        """Convert grid position to world coordinates (meters)."""
        x = grid_pos[0] * self.grid_to_world_scale
        y = grid_pos[1] * self.grid_to_world_scale
        return (x, y)

    def world_to_grid(self, world_pos: Tuple[float, float]) -> np.ndarray:
        """Convert world coordinates to grid position."""
        grid_x = int(world_pos[0] / self.grid_to_world_scale)
        grid_y = int(world_pos[1] / self.grid_to_world_scale)
        return np.array([grid_x, grid_y], dtype=np.int32)

    def get_observation(self) -> dict:
        """Get current observation for the model."""
        # Simplified local observation (would need camera/sensors in real deployment)
        obs_size = 2 * self.config.sensor_radius + 1
        local_occupancy = np.zeros((obs_size, obs_size, 3), dtype=np.float32)

        # Mark center as current position
        center = obs_size // 2
        local_occupancy[center, center, 2] = 1.0

        # Build observation dict
        observation = {
            "position": self.grid_position.astype(np.float32),
            "battery": np.array([self.drone.get_battery()], dtype=np.float32),
            "local_occupancy": local_occupancy,
            "coverage_map": self.coverage_map,
            "belief_map": self.belief_map,
            "targets_found": np.array([self.targets_found], dtype=np.float32),
        }

        return observation

    def execute_mission(
        self,
        flight_altitude: float = 1.5,
        max_steps: int = 500,
        step_duration: float = 2.0,
    ) -> bool:
        """Execute SAR mission.

        Args:
            flight_altitude: Flight altitude in meters
            max_steps: Maximum mission steps
            step_duration: Duration per step in seconds

        Returns:
            True if mission completed successfully
        """
        print("\n" + "="*60)
        print("SAR MISSION DEPLOYMENT")
        print("="*60)

        # Connect to drone
        if not self.drone.connect():
            print("Failed to connect to drone")
            return False

        # Takeoff
        print(f"\nTaking off to {flight_altitude}m...")
        if not self.drone.takeoff(flight_altitude):
            print("Takeoff failed")
            self.drone.disconnect()
            return False

        time.sleep(3)  # Stabilize after takeoff

        # Mission loop
        print("\nStarting SAR mission...")
        print("="*60)

        try:
            for step in range(max_steps):
                # Get observation
                obs = self.get_observation()

                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)

                # Execute action
                dx, dy = self.ACTIONS[action]
                new_grid_pos = self.grid_position + np.array([dx, dy])

                # Check bounds
                if not (
                    0 <= new_grid_pos[0] < self.config.grid_size
                    and 0 <= new_grid_pos[1] < self.config.grid_size
                ):
                    print(f"Step {step}: Out of bounds, staying in place")
                    continue

                # Update grid position
                self.grid_position = new_grid_pos

                # Convert to world coordinates
                target_x, target_y = self.grid_to_world(self.grid_position)

                # Move drone
                print(
                    f"Step {step}: Moving to grid ({self.grid_position[0]}, {self.grid_position[1]}) "
                    f"-> world ({target_x:.1f}m, {target_y:.1f}m)"
                )

                if not self.drone.move_to(target_x, target_y, flight_altitude, speed=1.0):
                    print("Movement failed")
                    break

                # Update coverage
                self.coverage_map[self.grid_position[1], self.grid_position[0]] = 1.0

                # Update belief map (simplified - would use real sensor data)
                self._update_belief_map_simulated()

                # Check battery
                battery = self.drone.get_battery()
                print(f"  Battery: {battery:.1f}%")

                if battery < 20.0:
                    print("Low battery! Returning to base...")
                    break

                # Wait for step duration
                time.sleep(step_duration)

                # Check coverage
                coverage = np.sum(self.coverage_map) / (self.config.grid_size ** 2)
                print(f"  Coverage: {coverage:.1%}")

                if coverage > 0.95:
                    print("\nSearch area fully covered!")
                    break

        except KeyboardInterrupt:
            print("\nMission interrupted by user")

        finally:
            # Land drone
            print("\nLanding...")
            self.drone.land()
            time.sleep(3)

            # Disconnect
            self.drone.disconnect()

            # Mission summary
            print("\n" + "="*60)
            print("MISSION SUMMARY")
            print("="*60)
            print(f"Steps completed: {step}")
            print(f"Coverage achieved: {np.sum(self.coverage_map) / (self.config.grid_size ** 2):.1%}")
            print(f"Final battery: {battery:.1f}%")
            print("="*60)

        return True

    def _update_belief_map_simulated(self) -> None:
        """Update belief map (simplified for deployment).

        In real deployment, this would use actual sensor data (camera, thermal, etc.)
        """
        x, y = self.grid_position
        r = self.config.sensor_radius

        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                world_x = x + i
                world_y = y + j

                if 0 <= world_x < self.config.grid_size and 0 <= world_y < self.config.grid_size:
                    distance = np.sqrt(i**2 + j**2)
                    if distance <= r:
                        detection_prob = self.config.detection_probability * np.exp(-distance / r)
                        self.belief_map[world_y, world_x] *= 1 - detection_prob

        # Renormalize
        total = np.sum(self.belief_map)
        if total > 1e-10:
            self.belief_map /= total


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy trained SAR agent on real drone")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["PPO", "SAC", "TD3"],
        default="SAC",
        help="RL algorithm",
    )
    parser.add_argument(
        "--drone-type",
        type=str,
        choices=["simulated", "crazyflie", "mavlink"],
        default="simulated",
        help="Type of drone to use",
    )
    parser.add_argument(
        "--connection",
        type=str,
        default=None,
        help="Connection string for real drone (e.g., 'radio://0/80/2M' for Crazyflie)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Grid size",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=1.5,
        help="Flight altitude in meters",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum mission steps",
    )
    parser.add_argument(
        "--grid-scale",
        type=float,
        default=2.0,
        help="Meters per grid cell",
    )

    args = parser.parse_args()

    # Create environment config
    config = EnvConfig(
        grid_size=args.grid_size,
        sensor_radius=3,
        detection_probability=0.95,
    )

    # Create drone interface
    if args.drone_type == "simulated":
        drone = create_drone_interface("simulated")
    elif args.drone_type == "crazyflie":
        if not args.connection:
            args.connection = "radio://0/80/2M/E7E7E7E7E7"
        drone = create_drone_interface("crazyflie", uri=args.connection)
    elif args.drone_type == "mavlink":
        if not args.connection:
            args.connection = "udp:127.0.0.1:14550"
        drone = create_drone_interface("mavlink", connection_string=args.connection)

    # Create deployment
    deployment = SARDeployment(
        model_path=args.model,
        algorithm=args.algorithm,
        drone_interface=drone,
        config=config,
        grid_to_world_scale=args.grid_scale,
    )

    # Execute mission
    success = deployment.execute_mission(
        flight_altitude=args.altitude,
        max_steps=args.max_steps,
        step_duration=2.0,
    )

    if success:
        print("\n✅ Mission completed successfully!")
    else:
        print("\n❌ Mission failed")


if __name__ == "__main__":
    main()
