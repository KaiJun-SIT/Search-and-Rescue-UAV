"""Demonstration of visualization glitching and the fix.

This script shows the difference between glitchy and smooth rendering.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from envs import SAREnv, EnvConfig
from envs.sar_env_smooth import SAREnvSmooth


def test_original_glitchy():
    """Test original rendering (may have glitching)."""
    print("\n" + "="*60)
    print("ORIGINAL RENDERING (May Show Glitching)")
    print("="*60)
    print("Watch for flickering/flashing in the window...")
    print("Press Ctrl+C to stop early if too glitchy\n")

    config = EnvConfig(grid_size=10, num_targets=1, render_mode="human")
    env = SAREnv(config)

    try:
        obs, info = env.reset(seed=42)
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            env.render()

            if done or truncated:
                break
    except KeyboardInterrupt:
        print("\n⚠️  Stopped due to glitching")
    finally:
        env.close()


def test_smooth_fixed():
    """Test smooth rendering (no glitching)."""
    print("\n" + "="*60)
    print("SMOOTH RENDERING (No Glitching)")
    print("="*60)
    print("This should be smooth with no flickering!\n")

    config = EnvConfig(grid_size=10, num_targets=1, render_mode="human")
    env = SAREnvSmooth(config)

    try:
        obs, info = env.reset(seed=42)
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            env.render()

            if done or truncated:
                break
    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")
    finally:
        env.close()


def explain_issue():
    """Explain the glitching issue."""
    print("\n" + "="*60)
    print("WHY THE GLITCHING HAPPENS")
    print("="*60)
    print("""
The original code has these issues:

1. ❌ ax.clear() - Clears entire plot every frame (visible flash)
2. ❌ plt.pause(0.001) - Too fast (1ms = up to 1000 FPS!)
3. ❌ Redraws static elements (grid lines, obstacles) every frame

This causes:
- Screen briefly goes blank → white flash
- Matplotlib can't keep up → lag
- Fast loop → continuous flickering

THE FIX:

1. ✅ plt.ion() - Interactive mode for smoother updates
2. ✅ plt.pause(0.05) - 50ms = 20 FPS (smooth for human eye)
3. ✅ canvas.draw_idle() + flush_events() - Better update method

Result: Smooth animation with no visible flickering!
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo visualization glitching fix")
    parser.add_argument(
        "--test",
        choices=["glitchy", "smooth", "both"],
        default="both",
        help="Which version to test"
    )

    args = parser.parse_args()

    explain_issue()

    if args.test in ["glitchy", "both"]:
        input("\nPress Enter to test ORIGINAL (glitchy) version...")
        test_original_glitchy()

    if args.test in ["smooth", "both"]:
        input("\nPress Enter to test SMOOTH (fixed) version...")
        test_smooth_fixed()

    print("\n" + "="*60)
    print("✅ DEMO COMPLETE")
    print("="*60)
    print("\nTo use smooth rendering in your code:")
    print("  from envs.sar_env_smooth import SAREnvSmooth")
    print("  env = SAREnvSmooth(config)")
    print("\nOr apply the fix to envs/sar_env.py (see VISUALIZATION_FIX.md)")
