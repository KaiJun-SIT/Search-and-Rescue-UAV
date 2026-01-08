"""Evaluation framework for comparing SAR search algorithms.

Compares RL agents (PPO, SAC, TD3) against baselines (grid, spiral, random, probability-weighted).
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import os
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, SAC, TD3

from envs import SAREnv, EnvConfig
from agents.baselines import create_baseline_agent


class SAREvaluator:
    """Evaluate and compare SAR search algorithms."""

    def __init__(self, env_config: EnvConfig, n_episodes: int = 100, seed: int = 42):
        """Initialize evaluator.

        Args:
            env_config: Environment configuration
            n_episodes: Number of episodes to evaluate
            seed: Random seed
        """
        self.env_config = env_config
        self.n_episodes = n_episodes
        self.seed = seed
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate_rl_agent(
        self, agent_path: str, agent_name: str, algorithm: str
    ) -> Dict[str, Any]:
        """Evaluate an RL agent.

        Args:
            agent_path: Path to saved model
            agent_name: Name for results
            algorithm: "PPO", "SAC", or "TD3"

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {agent_name} ({algorithm})")
        print(f"{'='*60}")

        # Load model
        if algorithm == "PPO":
            model = PPO.load(agent_path)
        elif algorithm == "SAC":
            model = SAC.load(agent_path)
        elif algorithm == "TD3":
            model = TD3.load(agent_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Run evaluation episodes
        metrics = self._run_episodes(lambda obs: model.predict(obs, deterministic=True)[0])

        self.results[agent_name] = metrics
        self._print_metrics(agent_name, metrics)

        return metrics

    def evaluate_baseline(self, baseline_type: str) -> Dict[str, Any]:
        """Evaluate a baseline agent.

        Args:
            baseline_type: "grid", "spiral", "random", or "probability"

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Baseline: {baseline_type}")
        print(f"{'='*60}")

        agent = create_baseline_agent(
            baseline_type, self.env_config.grid_size, self.env_config.sensor_radius
        )

        # Create policy function
        def baseline_policy(obs):
            # For baselines, we need to pass obstacles from environment
            # We'll handle this in _run_episodes
            return agent.select_action(obs)

        metrics = self._run_episodes(baseline_policy, is_baseline=True)

        agent_name = f"{baseline_type}_search"
        self.results[agent_name] = metrics
        self._print_metrics(agent_name, metrics)

        return metrics

    def _run_episodes(
        self, policy_fn, is_baseline: bool = False
    ) -> Dict[str, Any]:
        """Run evaluation episodes.

        Args:
            policy_fn: Policy function (obs -> action)
            is_baseline: Whether this is a baseline agent

        Returns:
            Dictionary of metrics
        """
        env = SAREnv(self.env_config)

        episode_rewards = []
        episode_lengths = []
        successes = []
        search_times = []
        coverages = []
        battery_remaining = []
        targets_found_list = []

        for ep in range(self.n_episodes):
            obs, info = env.reset(seed=self.seed + ep)
            done = False
            episode_reward = 0
            episode_length = 0

            # For baselines, create agent per episode
            if is_baseline:
                agent = create_baseline_agent(
                    policy_fn.__name__.split("_")[0],  # Extract type from function name
                    self.env_config.grid_size,
                    self.env_config.sensor_radius,
                )
                # Monkey-patch to pass obstacles
                original_policy = policy_fn

                def policy_with_obstacles(obs):
                    return agent.select_action(obs, obstacles=env.obstacles)

                policy_fn = policy_with_obstacles

            while not done:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            # Record metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            success = info["success"]
            successes.append(1 if success else 0)
            coverages.append(info["coverage"])
            battery_remaining.append(info["battery"])
            targets_found_list.append(info["targets_found"])

            if success:
                search_times.append(episode_length)

            # Print progress
            if (ep + 1) % 10 == 0:
                print(f"  Completed {ep + 1}/{self.n_episodes} episodes...")

        env.close()

        # Calculate statistics
        metrics = {
            "success_rate": np.mean(successes),
            "success_rate_std": np.std(successes),
            "avg_reward": np.mean(episode_rewards),
            "avg_reward_std": np.std(episode_rewards),
            "avg_episode_length": np.mean(episode_lengths),
            "avg_episode_length_std": np.std(episode_lengths),
            "avg_search_time": np.mean(search_times) if search_times else np.mean(episode_lengths),
            "avg_search_time_std": np.std(search_times) if search_times else np.std(episode_lengths),
            "avg_coverage": np.mean(coverages),
            "avg_coverage_std": np.std(coverages),
            "avg_battery_remaining": np.mean(battery_remaining),
            "avg_battery_remaining_std": np.std(battery_remaining),
            "avg_targets_found": np.mean(targets_found_list),
            "n_successful_episodes": len(search_times),
        }

        return metrics

    def _print_metrics(self, agent_name: str, metrics: Dict[str, Any]) -> None:
        """Print metrics in a nice format."""
        print(f"\nResults for {agent_name}:")
        print(f"  Success Rate: {metrics['success_rate']:.2%} ± {metrics['success_rate_std']:.2%}")
        print(f"  Avg Search Time: {metrics['avg_search_time']:.1f} ± {metrics['avg_search_time_std']:.1f} steps")
        print(f"  Avg Coverage: {metrics['avg_coverage']:.2%} ± {metrics['avg_coverage_std']:.2%}")
        print(f"  Avg Battery Remaining: {metrics['avg_battery_remaining']:.1f} ± {metrics['avg_battery_remaining_std']:.1f}")
        print(f"  Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['avg_reward_std']:.2f}")
        print(f"  Successful Episodes: {metrics['n_successful_episodes']}/{self.n_episodes}")

    def generate_comparison_report(self, save_dir: str) -> None:
        """Generate comparison plots and CSV report.

        Args:
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)

        # Create DataFrame
        df_data = []
        for agent_name, metrics in self.results.items():
            df_data.append({
                "Agent": agent_name,
                "Success Rate": metrics["success_rate"],
                "Avg Search Time": metrics["avg_search_time"],
                "Avg Coverage": metrics["avg_coverage"],
                "Avg Battery": metrics["avg_battery_remaining"],
                "Avg Reward": metrics["avg_reward"],
            })

        df = pd.DataFrame(df_data)

        # Save CSV
        csv_path = os.path.join(save_dir, "comparison_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Results saved to: {csv_path}")

        # Create plots
        self._plot_comparison(df, save_dir)

        # Print summary
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)

        # Find best performer
        best_success = df.loc[df["Success Rate"].idxmax()]
        best_speed = df.loc[df["Avg Search Time"].idxmin()]

        print(f"\n🏆 Best Success Rate: {best_success['Agent']} ({best_success['Success Rate']:.2%})")
        print(f"⚡ Fastest Search: {best_speed['Agent']} ({best_speed['Avg Search Time']:.1f} steps)")

        # Calculate improvement over baseline
        if "grid_search" in df["Agent"].values:
            grid_time = df[df["Agent"] == "grid_search"]["Avg Search Time"].values[0]
            for _, row in df.iterrows():
                if row["Agent"] != "grid_search":
                    improvement = ((grid_time - row["Avg Search Time"]) / grid_time) * 100
                    print(f"  {row['Agent']}: {improvement:+.1f}% vs grid search")

    def _plot_comparison(self, df: pd.DataFrame, save_dir: str) -> None:
        """Create comparison plots."""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Success rate
        axes[0, 0].bar(df["Agent"], df["Success Rate"])
        axes[0, 0].set_title("Success Rate", fontsize=14, fontweight="bold")
        axes[0, 0].set_ylabel("Success Rate")
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Search time
        axes[0, 1].bar(df["Agent"], df["Avg Search Time"])
        axes[0, 1].set_title("Average Search Time", fontsize=14, fontweight="bold")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Coverage
        axes[1, 0].bar(df["Agent"], df["Avg Coverage"])
        axes[1, 0].set_title("Average Coverage", fontsize=14, fontweight="bold")
        axes[1, 0].set_ylabel("Coverage")
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Reward
        axes[1, 1].bar(df["Agent"], df["Avg Reward"])
        axes[1, 1].set_title("Average Reward", fontsize=14, fontweight="bold")
        axes[1, 1].set_ylabel("Reward")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "comparison_plots.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"📈 Plots saved to: {plot_path}")
        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate SAR agents")
    parser.add_argument(
        "--rl-agents",
        nargs="+",
        help="Paths to RL models (format: name:algorithm:path)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=["grid", "spiral", "random", "probability"],
        default=["grid", "spiral", "probability"],
        help="Baseline agents to evaluate",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Grid size",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="experiments/evaluation",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create environment config
    env_config = EnvConfig(
        grid_size=args.grid_size,
        sensor_radius=3,
        initial_battery=100.0,
        battery_depletion_rate=1.0,
        num_targets=1,
        obstacle_density=0.1,
        detection_probability=0.95,
    )

    # Create evaluator
    evaluator = SAREvaluator(env_config, n_episodes=args.n_episodes, seed=args.seed)

    # Evaluate baselines
    for baseline in args.baselines:
        evaluator.evaluate_baseline(baseline)

    # Evaluate RL agents
    if args.rl_agents:
        for agent_spec in args.rl_agents:
            parts = agent_spec.split(":")
            if len(parts) != 3:
                print(f"⚠️  Skipping invalid agent spec: {agent_spec}")
                print("    Format should be: name:algorithm:path")
                continue

            name, algorithm, path = parts
            evaluator.evaluate_rl_agent(path, name, algorithm.upper())

    # Generate report
    evaluator.generate_comparison_report(args.save_dir)


if __name__ == "__main__":
    main()
