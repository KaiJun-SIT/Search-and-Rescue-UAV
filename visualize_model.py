"""Visualize trained RL models solving SAR tasks with live rendering.

This script loads trained RL models and displays them solving SAR search tasks
with real-time matplotlib visualization and console statistics.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import argparse
import time
import numpy as np
from typing import Optional, Dict, Any, List

from stable_baselines3 import PPO, SAC, TD3, A2C

try:
    from stable_baselines3 import DQN
    DQN_AVAILABLE = True
except ImportError:
    DQN = None
    DQN_AVAILABLE = False

from envs import SAREnv, EnvConfig


def auto_detect_algorithm(model_path: str) -> Optional[str]:
    """Auto-detect algorithm from model filename or path.

    Args:
        model_path: Path to model file

    Returns:
        Algorithm name (uppercase) or None if not detected
    """
    path_lower = model_path.lower()
    for algo in ['ppo', 'sac', 'td3', 'a2c', 'dqn']:
        if algo in path_lower:
            return algo.upper()
    return None


def find_best_model(exp_dir: str, algorithm: str) -> Optional[str]:
    """Find best model in experiment directory.

    Priority order:
    1. best_models/best_model_*.zip (highest success rate)
    2. {algorithm}_final.zip
    3. checkpoints/{algorithm}_model_*_steps.zip (latest)

    Args:
        exp_dir: Experiment directory path
        algorithm: Algorithm name (PPO, SAC, etc.)

    Returns:
        Path to best model or None if not found
    """
    exp_path = Path(exp_dir)

    # Check if directory exists
    if not exp_path.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return None

    # Priority 1: best_models directory
    best_models_dir = exp_path / "best_models"
    if best_models_dir.exists():
        models = list(best_models_dir.glob("best_model_*.zip"))
        if models:
            # Parse success rates from filenames (e.g., best_model_0.920.zip)
            def get_success_rate(path):
                try:
                    # Extract number from filename
                    return float(path.stem.split('_')[-1])
                except:
                    return 0.0

            # Return model with highest success rate
            best_model = max(models, key=get_success_rate)
            print(f"✅ Found best model: {best_model}")
            return str(best_model)

    # Priority 2: final model
    final_model = exp_path / f"{algorithm}_final.zip"
    if final_model.exists():
        print(f"✅ Found final model: {final_model}")
        return str(final_model)

    # Priority 3: latest checkpoint
    checkpoints_dir = exp_path / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob(f"{algorithm}_model_*_steps.zip"))
        if checkpoints:
            # Extract step number and sort
            def get_steps(path):
                try:
                    parts = path.stem.split('_')
                    return int(parts[-2])
                except:
                    return 0

            latest_checkpoint = max(checkpoints, key=get_steps)
            print(f"✅ Found latest checkpoint: {latest_checkpoint}")
            return str(latest_checkpoint)

    print(f"❌ No model found in {exp_dir}")
    return None


def load_model(model_path: str, algorithm: str):
    """Load model based on algorithm type with error handling.

    Args:
        model_path: Path to model file (.zip)
        algorithm: Algorithm name (PPO, SAC, TD3, A2C, DQN)

    Returns:
        Loaded model

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If algorithm is unknown
        ImportError: If DQN is not available
    """
    algorithm = algorithm.upper()

    # Check file exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading {algorithm} model from: {model_path}")

    # Load based on algorithm
    try:
        if algorithm == "PPO":
            model = PPO.load(model_path)
        elif algorithm == "SAC":
            model = SAC.load(model_path)
        elif algorithm == "TD3":
            model = TD3.load(model_path)
        elif algorithm == "A2C":
            model = A2C.load(model_path)
        elif algorithm == "DQN":
            if not DQN_AVAILABLE:
                raise ImportError("DQN not available. Install with: pip install stable-baselines3[extra]")
            model = DQN.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Supported: PPO, SAC, TD3, A2C, DQN")

        print(f"✅ Model loaded successfully!")
        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def print_episode_header(episode: int, total: int):
    """Print formatted episode header."""
    print("\n" + "=" * 70)
    print(f"  EPISODE {episode}/{total}")
    print("=" * 70)


def print_step_stats(step: int, obs: Dict, reward: float, info: Dict):
    """Print real-time step statistics (overwrites same line).

    Args:
        step: Current step number
        obs: Current observation dict
        reward: Current step reward
        info: Info dict from environment
    """
    battery = obs["battery"][0]
    targets = int(obs["targets_found"][0])
    coverage = info.get("coverage", 0.0)

    # Use \r to overwrite same line
    print(f"\r  Step {step:3d} | Battery: {battery:5.1f}% | "
          f"Targets: {targets} | Coverage: {coverage:5.1%} | "
          f"Reward: {reward:7.2f}", end="", flush=True)


def print_episode_summary(episode: int, stats: Dict[str, Any]):
    """Print episode summary statistics.

    Args:
        episode: Episode number
        stats: Episode statistics dict
    """
    print("\n")  # New line after step updates
    print(f"  {'-' * 66}")
    print(f"  Episode {episode} Summary:")
    print(f"    Success:          {'✅ Yes' if stats['success'] else '❌ No'}")
    print(f"    Steps:            {stats['steps']}")
    print(f"    Total Reward:     {stats['total_reward']:.2f}")
    print(f"    Targets Found:    {stats['targets_found']}")
    print(f"    Coverage:         {stats['coverage']:.1%}")
    print(f"    Battery Remaining: {stats['battery_remaining']:.1f}%")
    print(f"  {'-' * 66}")


class ModelVisualizer:
    """Visualize trained RL models solving SAR tasks."""

    def __init__(
        self,
        model_path: str,
        algorithm: str,
        env_config: EnvConfig,
        speed: float = 0.1,
        show_stats: bool = True,
    ):
        """Initialize visualizer.

        Args:
            model_path: Path to trained model
            algorithm: Algorithm name
            env_config: Environment configuration
            speed: Pause duration between steps (seconds)
            show_stats: Whether to show console statistics
        """
        self.model_path = model_path
        self.algorithm = algorithm
        self.env_config = env_config
        self.speed = speed
        self.show_stats = show_stats

        # Load model
        self.model = load_model(model_path, algorithm)

        # Create environment with render mode
        self.env = SAREnv(env_config)

        print(f"\n🎬 Visualization ready!")
        print(f"   Grid size: {env_config.grid_size}x{env_config.grid_size}")
        print(f"   Targets: {env_config.num_targets}")
        print(f"   Obstacles: {env_config.obstacle_density:.1%}")
        print(f"   Speed: {speed}s per step")

    def run_episode(self, episode_num: int = 0, seed: Optional[int] = None) -> Dict[str, Any]:
        """Run single episode with visualization.

        Args:
            episode_num: Episode number (for display)
            seed: Random seed for episode

        Returns:
            Episode statistics dict
        """
        obs, info = self.env.reset(seed=seed)

        episode_stats = {
            "steps": 0,
            "total_reward": 0.0,
            "success": False,
            "targets_found": 0,
            "coverage": 0.0,
            "battery_remaining": 0.0,
        }

        done = False
        step = 0

        while not done:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Render
            self.env.render()

            # Update stats
            episode_stats["total_reward"] += reward
            step += 1

            # Display stats
            if self.show_stats:
                print_step_stats(step, obs, reward, info)

            # Speed control
            time.sleep(self.speed)

        # Final stats
        episode_stats.update({
            "steps": step,
            "success": info.get("success", False),
            "targets_found": int(obs["targets_found"][0]),
            "coverage": info.get("coverage", 0.0),
            "battery_remaining": obs["battery"][0],
        })

        return episode_stats

    def run_multiple_episodes(self, n_episodes: int, base_seed: int = 42) -> Dict[str, Any]:
        """Run multiple episodes and aggregate statistics.

        Args:
            n_episodes: Number of episodes to run
            base_seed: Base random seed (incremented for each episode)

        Returns:
            Aggregated statistics dict
        """
        all_stats = []

        for ep in range(n_episodes):
            print_episode_header(ep + 1, n_episodes)

            # Run episode
            stats = self.run_episode(episode_num=ep, seed=base_seed + ep)
            all_stats.append(stats)

            # Print summary
            print_episode_summary(ep + 1, stats)

        # Aggregate statistics
        aggregated = {
            "n_episodes": n_episodes,
            "success_rate": np.mean([s["success"] for s in all_stats]),
            "avg_steps": np.mean([s["steps"] for s in all_stats]),
            "std_steps": np.std([s["steps"] for s in all_stats]),
            "avg_reward": np.mean([s["total_reward"] for s in all_stats]),
            "std_reward": np.std([s["total_reward"] for s in all_stats]),
            "avg_coverage": np.mean([s["coverage"] for s in all_stats]),
            "avg_battery": np.mean([s["battery_remaining"] for s in all_stats]),
        }

        # Print overall summary
        print("\n" + "=" * 70)
        print("  OVERALL SUMMARY")
        print("=" * 70)
        print(f"  Episodes:         {n_episodes}")
        print(f"  Success Rate:     {aggregated['success_rate']:.1%}")
        print(f"  Avg Steps:        {aggregated['avg_steps']:.1f} ± {aggregated['std_steps']:.1f}")
        print(f"  Avg Reward:       {aggregated['avg_reward']:.2f} ± {aggregated['std_reward']:.2f}")
        print(f"  Avg Coverage:     {aggregated['avg_coverage']:.1%}")
        print(f"  Avg Battery Left: {aggregated['avg_battery']:.1f}%")
        print("=" * 70 + "\n")

        return aggregated

    def close(self):
        """Close environment."""
        self.env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize trained RL models solving SAR tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize PPO model with default settings
  python visualize_model.py --model experiments/runs/ppo_exp/best_models/best_model_0.920.zip --algorithm PPO

  # Auto-detect algorithm and run 3 episodes slowly
  python visualize_model.py --model path/to/PPO_final.zip --episodes 3 --speed 0.5

  # Find and visualize best model from experiment
  python visualize_model.py --find-best experiments/runs/ppo_exp --algorithm PPO

  # Larger grid with custom settings
  python visualize_model.py --model path/to/model.zip --algorithm PPO --grid-size 15 --num-targets 2
        """
    )

    # Model selection arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        type=str,
        help="Path to trained model (.zip file)",
    )
    model_group.add_argument(
        "--find-best",
        type=str,
        metavar="EXP_DIR",
        help="Auto-find best model in experiment directory",
    )

    # Algorithm
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["PPO", "SAC", "TD3", "A2C", "DQN"],
        help="RL algorithm (auto-detect if omitted with --model)",
    )

    # Environment configuration
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Grid size (default: 10)",
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        default=1,
        help="Number of targets (default: 1)",
    )
    parser.add_argument(
        "--obstacle-density",
        type=float,
        default=0.1,
        help="Obstacle density 0-1 (default: 0.1)",
    )

    # Visualization settings
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to visualize (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.1,
        help="Pause duration between steps in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable console statistics output",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.grid_size < 5 or args.grid_size > 50:
        print("❌ Error: Grid size must be between 5 and 50")
        return

    if args.episodes < 1:
        print("❌ Error: Episodes must be at least 1")
        return

    if args.speed < 0:
        print("❌ Error: Speed must be non-negative")
        return

    if args.obstacle_density < 0 or args.obstacle_density > 1:
        print("❌ Error: Obstacle density must be between 0 and 1")
        return

    # Determine model path
    if args.find_best:
        if not args.algorithm:
            print("❌ Error: --algorithm required when using --find-best")
            return
        model_path = find_best_model(args.find_best, args.algorithm)
        if not model_path:
            return
    else:
        model_path = args.model

    # Auto-detect algorithm if not provided
    algorithm = args.algorithm
    if not algorithm:
        algorithm = auto_detect_algorithm(model_path)
        if not algorithm:
            print("❌ Error: Could not auto-detect algorithm from filename.")
            print("   Please specify --algorithm explicitly (PPO, SAC, TD3, A2C, or DQN)")
            return
        print(f"🔍 Auto-detected algorithm: {algorithm}")

    # Create environment config
    env_config = EnvConfig(
        grid_size=args.grid_size,
        num_targets=args.num_targets,
        obstacle_density=args.obstacle_density,
        render_mode="human",  # Critical: must be set before env creation
    )

    # Create visualizer
    try:
        visualizer = ModelVisualizer(
            model_path=model_path,
            algorithm=algorithm,
            env_config=env_config,
            speed=args.speed,
            show_stats=not args.no_stats,
        )

        # Run visualization
        if args.episodes == 1:
            print_episode_header(1, 1)
            stats = visualizer.run_episode(episode_num=0, seed=args.seed)
            print_episode_summary(1, stats)
        else:
            visualizer.run_multiple_episodes(args.episodes, args.seed)

        # Cleanup
        visualizer.close()
        print("\n✅ Visualization complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
