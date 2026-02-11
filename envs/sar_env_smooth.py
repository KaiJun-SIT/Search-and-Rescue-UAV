"""SAR Environment with Smooth Rendering (No Glitching).

This is a modified version of sar_env.py with improved rendering that eliminates
flickering and glitching during visualization.
"""

# Copy of sar_env.py with modified render() method
# Import and inherit from the original, then override render methods

from envs.sar_env import SAREnv as OriginalSAREnv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional


class SAREnvSmooth(OriginalSAREnv):
    """SAR Environment with smooth, flicker-free rendering.

    Changes from original:
    - Uses interactive mode (plt.ion())
    - Increased pause time (50ms instead of 1ms)
    - Better canvas management
    """

    def render(self) -> Optional[np.ndarray]:
        """Render the environment with smooth animation (no flickering).

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.config.render_mode is None:
            return None

        # First time setup
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))
            plt.ion()  # Interactive mode for smoother updates
            plt.show(block=False)

        # Clear axes
        for ax in self.ax:
            ax.clear()

        # Plot 1: Grid world with agent, targets, obstacles
        self._render_grid(self.ax[0])

        # Plot 2: Coverage map
        self._render_coverage(self.ax[1])

        # Plot 3: Belief map
        self._render_belief(self.ax[2])

        # Update title
        self.fig.suptitle(
            f"Step: {self.step_count} | Battery: {self.battery:.1f} | "
            f"Found: {self.targets_found}/{self.config.num_targets}"
        )

        if self.config.render_mode == "human":
            # Smoother rendering with proper timing
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.05)  # 50ms = 20 FPS (smooth, no flicker)
            return None
        else:  # rgb_array
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data

    def close(self) -> None:
        """Close the environment and cleanup."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        plt.ioff()  # Turn off interactive mode


# Example usage
if __name__ == "__main__":
    from envs.config import EnvConfig

    print("Testing Smooth SAR Environment Rendering...")
    print("=" * 60)

    config = EnvConfig(
        grid_size=10,
        num_targets=1,
        render_mode="human"
    )

    env = SAREnvSmooth(config)

    obs, info = env.reset(seed=42)
    print("Environment initialized. Watch for smooth rendering...")

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        if step % 20 == 0:
            print(f"Step {step}: No glitching should be visible!")

        if done or truncated:
            print(f"\nEpisode completed at step {step}")
            print(f"Success: {info['success']}")
            break

    env.close()
    print("\n✅ Test complete - rendering should be smooth!")
