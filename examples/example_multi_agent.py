"""Example: Multi-agent SAR coordination.

Demonstrates multiple drones working together to search for targets.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from envs.multi_agent_sar_env import MultiAgentSAREnv
from envs import EnvConfig


def main():
    """Run multi-agent SAR example."""
    print("\n" + "="*60)
    print("Multi-Agent SAR Environment Example")
    print("="*60)

    # Create multi-agent environment
    config = EnvConfig(
        grid_size=15,
        sensor_radius=3,
        num_targets=3,
        initial_battery=150.0,
    )

    num_agents = 3
    env = MultiAgentSAREnv(config, num_agents=num_agents)

    # Reset environment
    observations, info = env.reset(seed=42)

    print(f"\nEnvironment initialized:")
    print(f"  Grid size: {config.grid_size}x{config.grid_size}")
    print(f"  Number of agents: {num_agents}")
    print(f"  Number of targets: {config.num_targets}")

    print(f"\nStarting positions:")
    for i, obs in enumerate(observations):
        print(f"  Agent {i}: ({obs['position'][0]:.0f}, {obs['position'][1]:.0f})")

    # Run episode with random actions
    print("\nRunning episode with random actions...")
    print("-"*60)

    for step in range(200):
        # Random actions for all agents
        actions = tuple(env.action_space[i].sample() for i in range(num_agents))

        observations, rewards, terminated, truncated, info = env.step(actions)

        # Print status every 20 steps
        if step % 20 == 0:
            print(f"\nStep {step}:")
            for i in range(num_agents):
                print(f"  Agent {i}: Pos=({observations[i]['position'][0]:.0f},{observations[i]['position'][1]:.0f}) "
                      f"Battery={observations[i]['battery'][0]:.1f} Reward={rewards[i]:.1f}")
            print(f"  Targets found: {info['targets_found']}/{config.num_targets}")
            print(f"  Coverage: {info['coverage']:.1%}")
            print(f"  Active agents: {info['active_agents']}/{num_agents}")

        # Check if all agents are done
        if all(terminated) or all(truncated):
            print(f"\nEpisode ended at step {step}")
            break

    # Mission summary
    print("\n" + "="*60)
    print("Mission Summary")
    print("="*60)
    print(f"Success: {info['success']}")
    print(f"Targets found: {info['targets_found']}/{config.num_targets}")
    print(f"Coverage achieved: {info['coverage']:.1%}")
    print(f"Active agents at end: {info['active_agents']}/{num_agents}")
    print("\nAgent battery levels:")
    for i, battery in enumerate(info['agent_batteries']):
        print(f"  Agent {i}: {battery:.1f}%")
    print("="*60)

    env.close()


if __name__ == "__main__":
    main()
