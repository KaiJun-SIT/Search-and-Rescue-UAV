"""Example: Using the 3D SAR environment.

Demonstrates altitude control and 3D search patterns.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from envs.sar_env_3d import SAREnv3D
from envs import EnvConfig


def main():
    """Run 3D SAR environment example."""
    print("\n" + "="*60)
    print("3D SAR Environment Example")
    print("="*60)

    # Create 3D environment
    config = EnvConfig(
        grid_size=10,
        sensor_radius=3,
        num_targets=2,
        initial_battery=200.0,  # More battery for 3D
    )

    env = SAREnv3D(config, max_altitude=5, altitude_sensor_range=2)

    # Reset environment
    obs, info = env.reset(seed=42)

    print(f"\nEnvironment initialized:")
    print(f"  Grid size: {config.grid_size}x{config.grid_size}x{env.max_altitude}")
    print(f"  Action space: {env.action_space.n} actions (3D movement)")
    print(f"  Starting position: {obs['position']}")
    print(f"  Starting altitude: {obs['altitude'][0]}")

    # Run episode with random actions
    print("\nRunning episode with random actions...")
    print("-"*60)

    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Print status every 10 steps
        if step % 10 == 0:
            print(f"Step {step}:")
            print(f"  Position (X,Y,Z): ({obs['position'][0]:.0f}, {obs['position'][1]:.0f}, {obs['position'][2]:.0f})")
            print(f"  Battery: {obs['battery'][0]:.1f}")
            print(f"  Targets found: {int(obs['targets_found'][0])}")
            print(f"  Reward: {reward:.2f}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break

    # Mission summary
    print("\n" + "="*60)
    print("Mission Summary")
    print("="*60)
    print(f"Success: {info['success']}")
    print(f"Targets found: {info['targets_found']}/{config.num_targets}")
    print(f"Final battery: {info['battery']:.1f}")
    print(f"Final altitude: {info['altitude']}")
    print("="*60)

    env.close()


if __name__ == "__main__":
    main()
