"""Main training script for SAR UAV RL agents.

Supports training PPO, SAC, and TD3 agents on the SAR environment.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import os
from datetime import datetime
from typing import Optional

import gymnasium as gym
import yaml
from stable_baselines3 import PPO, SAC, TD3, A2C
try:
    from stable_baselines3 import DQN
    DQN_AVAILABLE = True
except ImportError:
    DQN = None
    DQN_AVAILABLE = False

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from envs import SAREnv, EnvConfig
from agents.rl_agent import create_policy_kwargs
from training.callbacks import SARMetricsCallback, BestModelCallback, ProgressBarCallback


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_env(env_config: dict, render_mode: Optional[str] = None) -> gym.Env:
    """Create SAR environment from config.

    Args:
        env_config: Environment configuration dict
        render_mode: Rendering mode

    Returns:
        SAR environment
    """
    config = EnvConfig(**env_config)
    config.render_mode = render_mode
    return SAREnv(config)


def train(
    algorithm: str,
    env_config: dict,
    training_config: dict,
    exp_name: Optional[str] = None,
    seed: int = 42,
) -> None:
    """Train an RL agent on the SAR environment.

    Args:
        algorithm: "PPO", "SAC", or "TD3"
        env_config: Environment configuration dict
        training_config: Training hyperparameters dict
        exp_name: Experiment name for logging
        seed: Random seed
    """
    # Create experiment directory
    if exp_name is None:
        exp_name = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exp_dir = os.path.join("experiments", "runs", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save configs
    config_save_path = os.path.join(exp_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump({"env": env_config, "training": training_config}, f)

    print(f"\n{'='*60}")
    print(f"Training {algorithm} on SAR Environment")
    print(f"Experiment: {exp_name}")
    print(f"Config saved to: {config_save_path}")
    print(f"{'='*60}\n")

    # Create training environment
    env = make_vec_env(
        lambda: create_env(env_config),
        n_envs=training_config.get("n_envs", 4),
        seed=seed,
    )
    env = VecMonitor(env)

    # Create evaluation environment
    eval_env = create_env(env_config)

    # Check if algorithm supports the action space
    test_env = create_env(env_config)
    action_space = test_env.action_space

    # Define which algorithms support which action spaces
    discrete_only = ["DQN", "A2C"]
    continuous_only = ["SAC", "TD3"]
    both = ["PPO"]

    from gymnasium.spaces import Discrete, Box

    if isinstance(action_space, Discrete):
        if algorithm in continuous_only:
            print("\n" + "="*60)
            print("❌ ERROR: Action Space Mismatch")
            print("="*60)
            print(f"Algorithm: {algorithm} only supports CONTINUOUS actions (Box)")
            print(f"Your environment has: DISCRETE actions (Discrete({action_space.n}))")
            print("\n💡 SOLUTION: Use an algorithm that supports discrete actions:")
            print("   - PPO (Recommended for SAR tasks)")
            print("   - A2C")
            print("   - DQN" + (" (available)" if DQN_AVAILABLE else " (install: pip install stable-baselines3[extra])"))
            print("\nTry running:")
            print(f"  python training/train.py --algorithm PPO --grid-size {env_config['grid_size']} --timesteps {training_config.get('total_timesteps', 100000)}")
            print("="*60 + "\n")
            test_env.close()
            env.close()
            eval_env.close()
            return

    test_env.close()

    # Get policy kwargs
    policy_kwargs = create_policy_kwargs(
        extractor_type=training_config.get("feature_extractor", "compact"),
        features_dim=training_config.get("features_dim", 128),
    )

    # Create algorithm
    total_timesteps = training_config.get("total_timesteps", 1_000_000)

    common_args = {
        "policy": "MultiInputPolicy",
        "env": env,
        "verbose": 1,
        "seed": seed,
        "tensorboard_log": os.path.join(exp_dir, "tensorboard"),
        "policy_kwargs": policy_kwargs,
    }

    if algorithm == "PPO":
        model = PPO(
            **common_args,
            learning_rate=training_config.get("learning_rate", 3e-4),
            n_steps=training_config.get("n_steps", 2048),
            batch_size=training_config.get("batch_size", 64),
            n_epochs=training_config.get("n_epochs", 10),
            gamma=training_config.get("gamma", 0.99),
            gae_lambda=training_config.get("gae_lambda", 0.95),
            clip_range=training_config.get("clip_range", 0.2),
            ent_coef=training_config.get("ent_coef", 0.01),
        )
    elif algorithm == "SAC":
        model = SAC(
            **common_args,
            learning_rate=training_config.get("learning_rate", 3e-4),
            buffer_size=training_config.get("buffer_size", 100000),
            batch_size=training_config.get("batch_size", 256),
            gamma=training_config.get("gamma", 0.99),
            tau=training_config.get("tau", 0.005),
            ent_coef=training_config.get("ent_coef", "auto"),  # Auto-tuning
            train_freq=training_config.get("train_freq", 1),
            gradient_steps=training_config.get("gradient_steps", 1),
        )
    elif algorithm == "TD3":
        model = TD3(
            **common_args,
            learning_rate=training_config.get("learning_rate", 3e-4),
            buffer_size=training_config.get("buffer_size", 100000),
            batch_size=training_config.get("batch_size", 256),
            gamma=training_config.get("gamma", 0.99),
            tau=training_config.get("tau", 0.005),
            policy_delay=training_config.get("policy_delay", 2),
            train_freq=training_config.get("train_freq", 1),
            gradient_steps=training_config.get("gradient_steps", 1),
        )
    elif algorithm == "A2C":
        model = A2C(
            **common_args,
            learning_rate=training_config.get("learning_rate", 7e-4),
            n_steps=training_config.get("n_steps", 5),
            gamma=training_config.get("gamma", 0.99),
            gae_lambda=training_config.get("gae_lambda", 1.0),
            ent_coef=training_config.get("ent_coef", 0.0),
            vf_coef=training_config.get("vf_coef", 0.5),
            max_grad_norm=training_config.get("max_grad_norm", 0.5),
        )
    elif algorithm == "DQN":
        if not DQN_AVAILABLE:
            print("❌ DQN not available. Install with: pip install stable-baselines3[extra]")
            env.close()
            eval_env.close()
            return

        model = DQN(
            **common_args,
            learning_rate=training_config.get("learning_rate", 1e-4),
            buffer_size=training_config.get("buffer_size", 100000),
            learning_starts=training_config.get("learning_starts", 50000),
            batch_size=training_config.get("batch_size", 32),
            tau=training_config.get("tau", 1.0),
            gamma=training_config.get("gamma", 0.99),
            train_freq=training_config.get("train_freq", 4),
            gradient_steps=training_config.get("gradient_steps", 1),
            target_update_interval=training_config.get("target_update_interval", 10000),
            exploration_fraction=training_config.get("exploration_fraction", 0.1),
            exploration_final_eps=training_config.get("exploration_final_eps", 0.05),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"\n{algorithm} model created with config:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Feature extractor: {training_config.get('feature_extractor', 'compact')}")
    print(f"  Learning rate: {training_config.get('learning_rate', 3e-4)}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config.get("checkpoint_freq", 50000),
        save_path=os.path.join(exp_dir, "checkpoints"),
        name_prefix=f"{algorithm}_model",
    )

    metrics_callback = SARMetricsCallback(
        eval_env=eval_env,
        eval_freq=training_config.get("eval_freq", 10000),
        n_eval_episodes=training_config.get("n_eval_episodes", 20),
        verbose=1,
    )

    best_model_callback = BestModelCallback(
        eval_env=eval_env,
        save_path=os.path.join(exp_dir, "best_models"),
        eval_freq=training_config.get("eval_freq", 10000),
        n_eval_episodes=training_config.get("n_eval_episodes", 20),
        verbose=1,
    )

    progress_callback = ProgressBarCallback(total_timesteps)

    callbacks = CallbackList(
        [checkpoint_callback, metrics_callback, best_model_callback, progress_callback]
    )

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    # Save final model
    final_model_path = os.path.join(exp_dir, f"{algorithm}_final.zip")
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")

    # Cleanup
    env.close()
    eval_env.close()

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Results saved to: {exp_dir}")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train SAR UAV RL agents")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["PPO", "SAC", "TD3", "A2C", "DQN"],
        default="PPO",
        help="RL algorithm to use (PPO/A2C/DQN for discrete, SAC/TD3 for continuous)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Grid size (overrides config)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Load or create default config
    if args.config:
        config = load_config(args.config)
        env_config = config["env"]
        training_config = config["training"]
    else:
        # Default configs
        env_config = {
            "grid_size": args.grid_size,
            "sensor_radius": 3,
            "initial_battery": 100.0,
            "battery_depletion_rate": 1.0,
            "num_targets": 1,
            "obstacle_density": 0.1,
            "detection_probability": 0.95,
        }

        training_config = {
            "total_timesteps": args.timesteps or 500_000,
            "n_envs": 4,
            "feature_extractor": "compact",
            "features_dim": 128,
            "learning_rate": 3e-4,
            "eval_freq": 10000,
            "n_eval_episodes": 20,
            "checkpoint_freq": 50000,
        }

        # Algorithm-specific defaults
        if args.algorithm == "PPO":
            training_config.update({
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "ent_coef": 0.01,
            })
        elif args.algorithm == "A2C":
            training_config.update({
                "n_steps": 5,
                "learning_rate": 7e-4,
            })
        elif args.algorithm == "DQN":
            training_config.update({
                "buffer_size": 100000,
                "batch_size": 32,
                "learning_starts": 50000,
                "train_freq": 4,
                "target_update_interval": 10000,
                "exploration_fraction": 0.1,
            })
        elif args.algorithm in ["SAC", "TD3"]:
            training_config.update({
                "buffer_size": 100000,
                "batch_size": 256,
                "train_freq": 1,
                "gradient_steps": 1,
            })

    # Override timesteps if provided
    if args.timesteps:
        training_config["total_timesteps"] = args.timesteps

    # Train
    train(
        algorithm=args.algorithm,
        env_config=env_config,
        training_config=training_config,
        exp_name=args.exp_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
