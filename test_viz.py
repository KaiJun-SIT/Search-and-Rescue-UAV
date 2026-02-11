"""Test visualization rendering to check for glitching."""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from envs import SAREnv, EnvConfig
import time

config = EnvConfig(
    grid_size=10,
    num_targets=1,
    render_mode="human"
)
env = SAREnv(config)

print("Starting visualization test...")
print("Watch for any glitching in the matplotlib window...")

obs, info = env.reset(seed=42)
for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    # Add small delay to see rendering
    time.sleep(0.1)

    if step % 10 == 0:
        print(f"Step {step}: Position ({obs['position'][0]:.0f}, {obs['position'][1]:.0f})")

    if done or truncated:
        print(f"\nEpisode ended at step {step}")
        break

env.close()
print("Visualization test complete.")
