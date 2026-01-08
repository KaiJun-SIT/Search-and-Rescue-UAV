"""Custom callbacks for SAR training."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Optional, Dict, Any
import os


class SARMetricsCallback(BaseCallback):
    """Custom callback for logging SAR-specific metrics.

    Logs:
    - Success rate (% of episodes where all targets found)
    - Average search time (steps to find all targets)
    - Coverage efficiency (% of grid covered)
    - Battery usage
    - Targets found per episode
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        verbose: int = 1,
    ):
        """Initialize metrics callback.

        Args:
            eval_env: Environment to evaluate on
            eval_freq: Evaluate every n steps
            n_eval_episodes: Number of episodes for evaluation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        # Track metrics
        self.success_rates: list = []
        self.avg_search_times: list = []
        self.avg_coverages: list = []
        self.avg_battery_remaining: list = []

    def _on_step(self) -> bool:
        """Called at every step."""
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_log()
        return True

    def _evaluate_and_log(self) -> None:
        """Evaluate policy and log metrics."""
        episode_rewards = []
        episode_lengths = []
        successes = []
        search_times = []
        coverages = []
        battery_remaining = []

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            # Record metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            successes.append(1 if info["success"] else 0)
            coverages.append(info["coverage"])
            battery_remaining.append(info["battery"])

            if info["success"]:
                search_times.append(episode_length)

        # Calculate statistics
        success_rate = np.mean(successes)
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        avg_coverage = np.mean(coverages)
        avg_battery = np.mean(battery_remaining)
        avg_search_time = np.mean(search_times) if search_times else avg_length

        # Store history
        self.success_rates.append(success_rate)
        self.avg_search_times.append(avg_search_time)
        self.avg_coverages.append(avg_coverage)
        self.avg_battery_remaining.append(avg_battery)

        # Log to tensorboard
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_reward", avg_reward)
        self.logger.record("eval/mean_ep_length", avg_length)
        self.logger.record("eval/mean_search_time", avg_search_time)
        self.logger.record("eval/mean_coverage", avg_coverage)
        self.logger.record("eval/mean_battery_remaining", avg_battery)

        if self.verbose > 0:
            print(f"\n--- Eval at step {self.n_calls} ---")
            print(f"Success rate: {success_rate:.2%}")
            print(f"Avg search time: {avg_search_time:.1f} steps")
            print(f"Avg coverage: {avg_coverage:.2%}")
            print(f"Avg battery remaining: {avg_battery:.1f}")
            print(f"Avg reward: {avg_reward:.2f}")


class BestModelCallback(BaseCallback):
    """Save model when performance improves.

    Saves the model whenever the success rate improves beyond the best seen so far.
    """

    def __init__(
        self,
        eval_env,
        save_path: str,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        verbose: int = 1,
    ):
        """Initialize best model callback.

        Args:
            eval_env: Environment for evaluation
            save_path: Directory to save models
            eval_freq: Evaluate every n steps
            n_eval_episodes: Number of eval episodes
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_success_rate = -np.inf

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_save()
        return True

    def _evaluate_and_save(self) -> None:
        """Evaluate and save if best."""
        successes = []

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

            successes.append(1 if info["success"] else 0)

        success_rate = np.mean(successes)

        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            save_file = os.path.join(
                self.save_path, f"best_model_{success_rate:.3f}.zip"
            )
            self.model.save(save_file)

            if self.verbose > 0:
                print(
                    f"\n🎉 New best model! Success rate: {success_rate:.2%} "
                    f"(saved to {save_file})"
                )


class ProgressBarCallback(BaseCallback):
    """Simple progress tracking callback."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.last_log = 0

    def _on_step(self) -> bool:
        # Log every 1% of training
        if self.n_calls - self.last_log >= self.total_timesteps / 100:
            progress = (self.n_calls / self.total_timesteps) * 100
            print(f"Progress: {progress:.1f}% ({self.n_calls}/{self.total_timesteps})")
            self.last_log = self.n_calls
        return True
