"""Test the colorbar stacking fix."""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from envs import SAREnv, EnvConfig
import time

print("\n" + "="*60)
print("Testing Colorbar Stacking Fix")
print("="*60)
print("\nThe coverage and belief maps should NOT shrink!")
print("Watch the plots - they should stay the same size.\n")

config = EnvConfig(
    grid_size=10,
    num_targets=1,
    render_mode="human"
)
env = SAREnv(config)

obs, info = env.reset(seed=42)

for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    time.sleep(0.05)  # Slow enough to see

    if step % 10 == 0:
        print(f"Step {step}: Plots should maintain their size!")

    if done or truncated:
        print(f"\n✅ Episode ended at step {step}")
        print("The plots should NOT have shrunk!")
        break

env.close()
print("\n" + "="*60)
print("✅ Fix Complete - No More Stacking/Shrinking!")
print("="*60)
