"""Example: Using the PyBullet SAR environment.

Demonstrates realistic physics-based simulation with:
- Realistic aerodynamics and drone physics
- 3D visualization with PyBullet
- Physics-based battery consumption
- Wind disturbances (optional)
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from envs.pybullet_sar_env import PyBulletSAREnv
from envs import EnvConfig


def main():
    """Run PyBullet SAR environment example."""
    print("\n" + "="*60)
    print("PyBullet SAR Environment Example")
    print("Realistic Physics Simulation")
    print("="*60)

    # Create PyBullet environment with physics simulation
    config = EnvConfig(
        grid_size=10,
        sensor_radius=3,
        num_targets=2,
        initial_battery=200.0,
        render_mode="human",  # Enable visualization
    )

    # Initialize PyBullet environment with GUI enabled
    env = PyBulletSAREnv(
        config=config,
        drone_model="cf2x",  # Crazyflie 2.X drone model
        physics="pyb",        # PyBullet physics
        gui=True,            # Enable 3D visualization window
        record=False,        # Set to True to record video
    )

    # Reset environment
    obs, info = env.reset(seed=42)

    print(f"\nEnvironment initialized:")
    print(f"  Grid size: {config.grid_size}x{config.grid_size}")
    print(f"  Drone model: Crazyflie 2.X")
    print(f"  Physics: PyBullet realistic simulation")
    print(f"  Action space: {env.action_space}")
    print(f"  Starting position: {obs['position']}")
    print(f"\nControls:")
    print("  - PyBullet window will show 3D drone simulation")
    print("  - Press Ctrl+C to stop")

    # Run episode with random actions
    print("\nRunning episode with random actions...")
    print("-"*60)

    try:
        for step in range(200):  # More steps for physics simulation
            # Random action (replace with trained agent for better performance)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Render is automatic with gui=True, but we can call it explicitly
            env.render()

            # Slower update rate for physics simulation
            time.sleep(0.05)

            # Print status every 20 steps
            if step % 20 == 0:
                print(f"Step {step}:")
                print(f"  Position (X,Y): ({obs['position'][0]:.2f}, {obs['position'][1]:.2f})")
                print(f"  Battery: {obs['battery'][0]:.1f}%")
                print(f"  Targets found: {int(obs['targets_found'][0])}")
                print(f"  Reward: {reward:.2f}")

            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                break

    except KeyboardInterrupt:
        print("\nStopping simulation...")

    # Mission summary
    print("\n" + "="*60)
    print("Mission Summary")
    print("="*60)
    print(f"Success: {info.get('success', False)}")
    print(f"Targets found: {info.get('targets_found', 0)}/{config.num_targets}")
    print(f"Final battery: {info.get('battery', 0):.1f}%")
    print("="*60)

    env.close()


if __name__ == "__main__":
    main()
