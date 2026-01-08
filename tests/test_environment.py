"""Test script to verify SAR environment works correctly.

Run this to ensure the environment is functioning properly before training.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from envs import SAREnv, EnvConfig


def test_basic_environment():
    """Test basic environment functionality."""
    print("\n" + "="*60)
    print("Testing SAR Environment")
    print("="*60)

    # Create environment
    config = EnvConfig(
        grid_size=10,
        sensor_radius=3,
        num_targets=1,
        obstacle_density=0.1,
    )
    env = SAREnv(config)

    print("\n✓ Environment created successfully")
    print(f"  Grid size: {config.grid_size}x{config.grid_size}")
    print(f"  Sensor radius: {config.sensor_radius}")
    print(f"  Observation size: {config.observation_size}x{config.observation_size}")

    # Test reset
    obs, info = env.reset(seed=42)

    print("\n✓ Environment reset successfully")
    print(f"  Agent position: {obs['position']}")
    print(f"  Battery: {obs['battery'][0]}")
    print(f"  Local obs shape: {obs['local_occupancy'].shape}")
    print(f"  Coverage map shape: {obs['coverage_map'].shape}")
    print(f"  Belief map shape: {obs['belief_map'].shape}")
    print(f"  Targets found: {obs['targets_found'][0]}")

    # Verify observation space
    assert env.observation_space.contains(obs), "Observation not in observation space!"
    print("\n✓ Observation space validation passed")

    # Test steps
    print("\nTesting episode simulation...")
    total_reward = 0
    steps = 0
    max_steps = 100

    while steps < max_steps:
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            print(f"\n✓ Episode terminated after {steps} steps")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Success: {info['success']}")
            print(f"  Targets found: {info['targets_found']}")
            print(f"  Coverage: {info['coverage']:.2%}")
            print(f"  Battery remaining: {info['battery']:.1f}")
            break

    if steps == max_steps:
        print(f"\n⚠️  Episode did not terminate in {max_steps} steps")

    # Test partial observability
    print("\nTesting partial observability...")
    obs, _ = env.reset(seed=123)
    local_obs = obs["local_occupancy"]

    # Verify local observation is correct size
    expected_size = config.observation_size
    assert local_obs.shape == (expected_size, expected_size, 3), \
        f"Local observation has wrong shape: {local_obs.shape}"

    print(f"✓ Local observation has correct shape: {local_obs.shape}")

    # Verify center cell is marked as current position
    center = expected_size // 2
    assert local_obs[center, center, 2] == 1.0, "Center cell should mark agent position"
    print("✓ Agent position correctly marked in center")

    # Test belief map updates
    print("\nTesting Bayesian belief map updates...")
    obs, _ = env.reset(seed=456)
    initial_belief_sum = np.sum(obs["belief_map"])

    print(f"  Initial belief sum: {initial_belief_sum:.6f}")
    assert abs(initial_belief_sum - 1.0) < 1e-5, "Belief map should sum to 1"
    print("✓ Belief map properly normalized")

    # Take a few steps and check belief updates
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

    updated_belief_sum = np.sum(obs["belief_map"])
    print(f"  Updated belief sum: {updated_belief_sum:.6f}")
    assert abs(updated_belief_sum - 1.0) < 1e-5, "Belief map should remain normalized"
    print("✓ Belief map updates maintain normalization")

    # Test battery depletion
    print("\nTesting battery mechanics...")
    obs, _ = env.reset(seed=789)
    initial_battery = obs["battery"][0]

    obs, _, _, _, _ = env.step(0)  # Take one step
    new_battery = obs["battery"][0]

    battery_used = initial_battery - new_battery
    print(f"  Battery used per step: {battery_used}")
    assert battery_used == config.battery_depletion_rate, "Battery depletion incorrect"
    print("✓ Battery depletion working correctly")

    env.close()

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")


def test_multi_target():
    """Test multi-target scenario."""
    print("\n" + "="*60)
    print("Testing Multi-Target Scenario")
    print("="*60)

    config = EnvConfig(
        grid_size=15,
        num_targets=3,
        target_priorities=[1.0, 1.5, 2.0],
        obstacle_density=0.15,
    )
    env = SAREnv(config)

    print(f"\n✓ Environment with {config.num_targets} targets created")
    print(f"  Priorities: {config.target_priorities}")

    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset with {len(env.target_positions)} targets placed")

    env.close()
    print("\n✓ Multi-target test passed\n")


def test_with_baselines():
    """Test environment with baseline agents."""
    print("\n" + "="*60)
    print("Testing with Baseline Agents")
    print("="*60)

    from agents.baselines import create_baseline_agent

    config = EnvConfig(grid_size=10)
    env = SAREnv(config)

    for baseline_type in ["grid", "spiral", "random", "probability"]:
        print(f"\nTesting {baseline_type} search...")

        agent = create_baseline_agent(baseline_type, config.grid_size, config.sensor_radius)
        obs, _ = env.reset(seed=42)

        for _ in range(50):
            action = agent.select_action(obs, obstacles=env.obstacles)
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

        print(f"  ✓ {baseline_type} agent completed episode")

    env.close()
    print("\n✓ All baseline agents working\n")


if __name__ == "__main__":
    test_basic_environment()
    test_multi_target()
    test_with_baselines()

    print("="*60)
    print("🎉 Environment is ready for training!")
    print("="*60)
