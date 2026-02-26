"""Evaluation dashboard for SAR UAV RL models.

Scans experiments/runs/ for all trained models, evaluates them against
baselines, generates a multi-panel dashboard image, and writes a
EVALUATION_REPORT.md. Supports --watch mode to auto-update on new runs.

Usage:
    python evaluation/run_dashboard.py
    python evaluation/run_dashboard.py --watch --watch-interval 60
    python evaluation/run_dashboard.py --force-reeval --episodes 100
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend so it works without a display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO, SAC, TD3, A2C

try:
    from stable_baselines3 import DQN
    DQN_AVAILABLE = True
except ImportError:
    DQN = None
    DQN_AVAILABLE = False

from agents.baselines import create_baseline_agent
from envs import EnvConfig, SAREnv

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RL_COLORS = {
    "PPO": "#4169E1",        # royalblue
    "A2C": "#6495ED",        # cornflowerblue
    "DQN": "#4682B4",        # steelblue
    "SAC": "#1E90FF",        # dodgerblue
    "TD3": "#00BFFF",        # deepskyblue
}

BASELINE_COLORS = {
    "Grid Search":        "#FF8C00",   # darkorange
    "Spiral Search":      "#FF6347",   # tomato
    "Random Search":      "#FFA07A",   # lightsalmon
    "Probability Search": "#DAA520",   # goldenrod
}

BASELINE_TYPES = [
    ("grid",        "Grid Search"),
    ("spiral",      "Spiral Search"),
    ("random",      "Random Search"),
    ("probability", "Probability Search"),
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model discovery
# ──────────────────────────────────────────────────────────────────────────────

def _detect_algorithm(path: str) -> Optional[str]:
    """Return uppercase algorithm name found in path, or None."""
    p = path.lower()
    for algo in ("ppo", "sac", "td3", "a2c", "dqn"):
        if algo in p:
            return algo.upper()
    return None


def _best_model_in_exp(exp_path: Path) -> Optional[Path]:
    """Find the single best model file inside an experiment directory."""
    # Priority 1: best_models/*.zip – pick highest success-rate number
    best_dir = exp_path / "best_models"
    if best_dir.exists():
        zips = sorted(best_dir.glob("best_model_*.zip"))
        if zips:
            def _rate(p: Path) -> float:
                try:
                    return float(p.stem.split("_")[-1])
                except ValueError:
                    return 0.0
            return max(zips, key=_rate)

    # Priority 2: <ALGO>_final.zip
    for f in exp_path.glob("*_final.zip"):
        return f

    # Priority 3: latest checkpoint
    ckpt_dir = exp_path / "checkpoints"
    if ckpt_dir.exists():
        zips = sorted(ckpt_dir.glob("*_model_*_steps.zip"))
        if zips:
            def _steps(p: Path) -> int:
                parts = p.stem.split("_")
                try:
                    return int(parts[-2])
                except (ValueError, IndexError):
                    return 0
            return max(zips, key=_steps)

    return None


def discover_models(exp_dir: str) -> List[Dict[str, str]]:
    """Walk exp_dir and return one best model per experiment sub-directory.

    Each entry: {"name": str, "algorithm": str, "path": str}
    """
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        return []

    found = []
    for sub in sorted(exp_path.iterdir()):
        if not sub.is_dir():
            continue
        model_file = _best_model_in_exp(sub)
        if model_file is None:
            continue
        algo = _detect_algorithm(str(model_file)) or _detect_algorithm(sub.name)
        if algo is None:
            # Try parent directory name
            algo = "UNKNOWN"
        found.append({
            "name":      sub.name,
            "algorithm": algo,
            "path":      str(model_file),
        })
    return found


# ──────────────────────────────────────────────────────────────────────────────
# 2. Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, algorithm: str):
    """Load an SB3 model from disk."""
    algorithm = algorithm.upper()
    loaders = {
        "PPO": PPO,
        "SAC": SAC,
        "TD3": TD3,
        "A2C": A2C,
    }
    if DQN_AVAILABLE:
        loaders["DQN"] = DQN

    if algorithm not in loaders:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {list(loaders)}")

    return loaders[algorithm].load(model_path)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Evaluation engine
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_agent(
    policy_fn: Callable,
    env_config: EnvConfig,
    n_episodes: int,
    base_seed: int,
) -> Dict[str, Any]:
    """Run n_episodes and return a metrics dict."""
    env = SAREnv(env_config)

    rewards, lengths, successes = [], [], []
    coverages, batteries, search_times = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_len += 1

        rewards.append(ep_reward)
        lengths.append(ep_len)
        success = bool(info["success"])
        successes.append(1 if success else 0)
        coverages.append(float(info["coverage"]))
        batteries.append(float(info["battery"]))
        if success:
            search_times.append(ep_len)

        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"    {ep + 1}/{n_episodes} episodes", end="\r", flush=True)

    print()
    env.close()

    return {
        "success_rate":            float(np.mean(successes)),
        "success_rate_std":        float(np.std(successes)),
        "avg_reward":              float(np.mean(rewards)),
        "avg_reward_std":          float(np.std(rewards)),
        "avg_steps":               float(np.mean(lengths)),
        "avg_steps_std":           float(np.std(lengths)),
        "avg_search_time":         float(np.mean(search_times)) if search_times else float(np.mean(lengths)),
        "avg_search_time_std":     float(np.std(search_times)) if search_times else float(np.std(lengths)),
        "avg_coverage":            float(np.mean(coverages)),
        "avg_coverage_std":        float(np.std(coverages)),
        "avg_battery_remaining":   float(np.mean(batteries)),
        "avg_battery_remaining_std": float(np.std(batteries)),
        "n_successful":            len(search_times),
        "n_episodes":              n_episodes,
    }


def make_rl_policy(model) -> Callable:
    """Wrap a trained SB3 model as a simple policy function."""
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def make_baseline_policy(baseline_type: str, env_config: EnvConfig, env: SAREnv) -> Callable:
    """Create a stateful baseline policy that resets automatically each episode.

    Returns a closure that, on every call, checks whether a new episode started
    (position jumped back to start) and re-creates the baseline agent.
    """
    agent_holder = [None]

    def policy(obs):
        if agent_holder[0] is None:
            agent_holder[0] = create_baseline_agent(
                baseline_type,
                env_config.grid_size,
                env_config.sensor_radius,
            )
        return agent_holder[0].select_action(obs, obstacles=env.obstacles)

    return policy, agent_holder


def evaluate_baseline(
    baseline_type: str,
    baseline_label: str,
    env_config: EnvConfig,
    n_episodes: int,
    base_seed: int,
) -> Dict[str, Any]:
    """Evaluate a baseline agent."""
    env = SAREnv(env_config)

    rewards, lengths, successes = [], [], []
    coverages, batteries, search_times = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0

        # Fresh agent per episode
        agent = create_baseline_agent(
            baseline_type,
            env_config.grid_size,
            env_config.sensor_radius,
        )

        while not done:
            action = agent.select_action(obs, obstacles=env.obstacles)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_len += 1

        rewards.append(ep_reward)
        lengths.append(ep_len)
        success = bool(info["success"])
        successes.append(1 if success else 0)
        coverages.append(float(info["coverage"]))
        batteries.append(float(info["battery"]))
        if success:
            search_times.append(ep_len)

        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"    {ep + 1}/{n_episodes} episodes", end="\r", flush=True)

    print()
    env.close()

    return {
        "success_rate":              float(np.mean(successes)),
        "success_rate_std":          float(np.std(successes)),
        "avg_reward":                float(np.mean(rewards)),
        "avg_reward_std":            float(np.std(rewards)),
        "avg_steps":                 float(np.mean(lengths)),
        "avg_steps_std":             float(np.std(lengths)),
        "avg_search_time":           float(np.mean(search_times)) if search_times else float(np.mean(lengths)),
        "avg_search_time_std":       float(np.std(search_times)) if search_times else float(np.std(lengths)),
        "avg_coverage":              float(np.mean(coverages)),
        "avg_coverage_std":          float(np.std(coverages)),
        "avg_battery_remaining":     float(np.mean(batteries)),
        "avg_battery_remaining_std": float(np.std(batteries)),
        "n_successful":              len(search_times),
        "n_episodes":                n_episodes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Results cache
# ──────────────────────────────────────────────────────────────────────────────

def _cache_key(model_path: str, n_episodes: int, grid_size: int) -> str:
    mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0
    return f"{model_path}|mtime={mtime:.0f}|ep={n_episodes}|gs={grid_size}"


def load_cache(cache_path: str) -> Dict:
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_cache(cache_path: str, cache: Dict) -> None:
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def is_cached(key: str, cache: Dict) -> bool:
    return key in cache.get("results", {})


def get_cached(key: str, cache: Dict) -> Optional[Dict]:
    return cache.get("results", {}).get(key)


def set_cached(key: str, data: Dict, cache: Dict) -> None:
    cache.setdefault("results", {})[key] = data


# ──────────────────────────────────────────────────────────────────────────────
# 5. Dashboard generation
# ──────────────────────────────────────────────────────────────────────────────

def _bar_chart(
    ax,
    names: List[str],
    values: List[float],
    errors: List[float],
    colors: List[str],
    title: str,
    ylabel: str,
    best_baseline: Optional[float] = None,
    fmt_pct: bool = False,
):
    """Draw a bar chart with error bars and optional baseline reference line."""
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colors, width=0.6, zorder=2, alpha=0.85)
    ax.errorbar(x, values, yerr=errors, fmt="none", color="#333333",
                capsize=4, linewidth=1.5, zorder=3)

    # Reference line for best baseline
    if best_baseline is not None:
        ax.axhline(best_baseline, color="#555555", linestyle="--",
                   linewidth=1.2, alpha=0.7, label="Best Baseline")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace(" ", "\n") for n in names],
        fontsize=7.5, ha="center"
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Format y-axis
    if fmt_pct:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
        )
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)

    # Improvement labels above RL bars (skip baselines)
    if best_baseline is not None:
        for bar, name, val in zip(bars, names, values):
            if name in BASELINE_COLORS:
                continue
            improvement = val - best_baseline
            sign = "+" if improvement >= 0 else ""
            label = f"{sign}{improvement:.1%}" if fmt_pct else f"{sign}{improvement:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                label,
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color="#1a5276" if improvement >= 0 else "#922b21",
            )

    return bars


def _radar_chart(ax, names: List[str], scores_matrix: np.ndarray, colors: List[str]):
    """Draw a radar/spider chart with one polygon per agent."""
    metrics = ["Success\nRate", "Coverage", "Reward", "Battery", "Speed"]
    n_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title("Agent Comparison\n(Normalized)", fontsize=11, fontweight="bold", pad=15)

    handles = []
    for name, scores, color in zip(names, scores_matrix, colors):
        values = scores.tolist() + scores[:1].tolist()
        ax.plot(angles, values, "o-", linewidth=1.5, color=color, alpha=0.9)
        ax.fill(angles, values, alpha=0.1, color=color)
        handles.append(mpatches.Patch(color=color, label=name, alpha=0.7))

    ax.legend(handles=handles, loc="lower right",
              bbox_to_anchor=(1.3, -0.05), fontsize=7, framealpha=0.8)


def generate_dashboard(
    results: Dict[str, Dict],
    output_path: str,
    grid_size: int,
    n_episodes: int,
) -> None:
    """Generate a 2×3 multi-panel dashboard PNG and save to output_path."""

    # ── Separate RL vs baseline entries ─────────────────────────────────────
    rl_entries   = [(k, v) for k, v in results.items() if v.get("agent_type") == "rl"]
    base_entries = [(k, v) for k, v in results.items() if v.get("agent_type") == "baseline"]

    all_entries = rl_entries + base_entries
    names   = [k for k, _ in all_entries]
    metrics = [v for _, v in all_entries]

    def _color(name: str, algo: str) -> str:
        if name in BASELINE_COLORS:
            return BASELINE_COLORS[name]
        return RL_COLORS.get(algo, "#4169E1")

    colors = [_color(k, v.get("algorithm", "PPO")) for k, v in all_entries]

    # ── Best baseline reference ───────────────────────────────────────────────
    baseline_success = [v["success_rate"] for _, v in base_entries] or [0.0]
    best_baseline_sr = max(baseline_success)

    baseline_time = [v["avg_search_time"] for _, v in base_entries] or [100.0]
    best_baseline_time = min(baseline_time)

    baseline_cov = [v["avg_coverage"] for _, v in base_entries] or [0.0]
    best_baseline_cov = max(baseline_cov)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("#f8f9fa")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"SAR UAV RL Evaluation Dashboard\n"
        f"Grid {grid_size}×{grid_size}  |  {n_episodes} episodes per agent  |  Updated: {ts}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # 2×3 grid: leave room for radar (polar) in [1,2]
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.88, bottom=0.07)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2], polar=True)

    # ── Panel 0: Success Rate ─────────────────────────────────────────────────
    _bar_chart(
        ax0, names,
        [m["success_rate"] for m in metrics],
        [m["success_rate_std"] for m in metrics],
        colors, "Success Rate", "Success Rate",
        best_baseline=best_baseline_sr, fmt_pct=True,
    )

    # ── Panel 1: Avg Search Time ──────────────────────────────────────────────
    _bar_chart(
        ax1, names,
        [m["avg_search_time"] for m in metrics],
        [m["avg_search_time_std"] for m in metrics],
        colors, "Avg Search Time", "Steps to Find Target",
        best_baseline=best_baseline_time, fmt_pct=False,
    )

    # ── Panel 2: Coverage ─────────────────────────────────────────────────────
    _bar_chart(
        ax2, names,
        [m["avg_coverage"] for m in metrics],
        [m["avg_coverage_std"] for m in metrics],
        colors, "Grid Coverage", "Coverage %",
        best_baseline=best_baseline_cov, fmt_pct=True,
    )

    # ── Panel 3: Avg Reward ───────────────────────────────────────────────────
    baseline_reward = [v["avg_reward"] for _, v in base_entries] or [0.0]
    _bar_chart(
        ax3, names,
        [m["avg_reward"] for m in metrics],
        [m["avg_reward_std"] for m in metrics],
        colors, "Avg Episode Reward", "Reward",
        best_baseline=max(baseline_reward), fmt_pct=False,
    )

    # ── Panel 4: Battery Remaining ────────────────────────────────────────────
    _bar_chart(
        ax4, names,
        [m["avg_battery_remaining"] for m in metrics],
        [m["avg_battery_remaining_std"] for m in metrics],
        colors, "Battery Remaining", "Battery %",
        best_baseline=None, fmt_pct=False,
    )

    # ── Panel 5: Radar chart ──────────────────────────────────────────────────
    # Normalise five metrics across all agents to [0, 1]
    def _norm(vals):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [0.5] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    sr_norm  = _norm([m["success_rate"]          for m in metrics])
    cov_norm = _norm([m["avg_coverage"]           for m in metrics])
    rew_norm = _norm([m["avg_reward"]             for m in metrics])
    bat_norm = _norm([m["avg_battery_remaining"]  for m in metrics])
    # Speed: lower steps = faster = better, so invert
    all_steps = [m["avg_steps"] for m in metrics]
    spd_norm = _norm([-s for s in all_steps])  # invert

    radar_matrix = np.column_stack([sr_norm, cov_norm, rew_norm, bat_norm, spd_norm])
    _radar_chart(ax5, names, radar_matrix, colors)

    # ── Footer ────────────────────────────────────────────────────────────────
    if rl_entries:
        best_rl_name, best_rl = max(rl_entries, key=lambda x: x[1]["success_rate"])
        best_base = max(base_entries, key=lambda x: x[1]["success_rate"])[1]["success_rate"] if base_entries else 0
        improvement = best_rl["success_rate"] - best_base
        sign = "+" if improvement >= 0 else ""
        footer = (
            f"Best RL Agent: {best_rl_name}  |  "
            f"Success Rate: {best_rl['success_rate']:.1%}  |  "
            f"vs Best Baseline: {sign}{improvement:.1%}"
        )
        fig.text(0.5, 0.01, footer, ha="center", fontsize=9,
                 color="#2c3e50", style="italic")

    # ── Legend ────────────────────────────────────────────────────────────────
    rl_patch   = mpatches.Patch(color="#4169E1", alpha=0.8, label="RL Models")
    base_patch = mpatches.Patch(color="#FF8C00", alpha=0.8, label="Baselines")
    fig.legend(handles=[rl_patch, base_patch], loc="upper right",
               bbox_to_anchor=(0.98, 0.97), fontsize=9, framealpha=0.9)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Dashboard] Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Markdown report
# ──────────────────────────────────────────────────────────────────────────────

def generate_markdown(
    results: Dict[str, Dict],
    dashboard_img_path: str,
    report_path: str,
    grid_size: int,
    n_episodes: int,
    seed: int,
) -> None:
    """Write EVALUATION_REPORT.md."""

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rl_entries   = [(k, v) for k, v in results.items() if v.get("agent_type") == "rl"]
    base_entries = [(k, v) for k, v in results.items() if v.get("agent_type") == "baseline"]
    all_entries  = rl_entries + base_entries

    # ── Helpers ───────────────────────────────────────────────────────────────
    def pct(v, std=None):
        s = f"{v:.1%}"
        if std is not None:
            s += f" ± {std:.1%}"
        return s

    def flt(v, std=None, dp=1):
        s = f"{v:.{dp}f}"
        if std is not None:
            s += f" ± {std:.{dp}f}"
        return s

    # ── Best performers ───────────────────────────────────────────────────────
    if rl_entries:
        best_sr   = max(rl_entries, key=lambda x: x[1]["success_rate"])
        best_time = min(rl_entries, key=lambda x: x[1]["avg_search_time"])
        best_cov  = max(rl_entries, key=lambda x: x[1]["avg_coverage"])
        best_bat  = max(rl_entries, key=lambda x: x[1]["avg_battery_remaining"])
    else:
        best_sr = best_time = best_cov = best_bat = (None, {})

    # Baseline rates for improvement table
    grid_rate   = results.get("Grid Search",        {}).get("success_rate", None)
    spiral_rate = results.get("Spiral Search",       {}).get("success_rate", None)
    prob_rate   = results.get("Probability Search",  {}).get("success_rate", None)

    def _improvement_str(agent_rate, baseline_rate):
        if baseline_rate is None:
            return "N/A"
        diff = agent_rate - baseline_rate
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.1%}"

    # ── Relative image path from report location ──────────────────────────────
    report_dir   = Path(report_path).parent
    try:
        img_rel = Path(dashboard_img_path).relative_to(report_dir)
    except ValueError:
        img_rel = Path(dashboard_img_path)

    # ── Build markdown ─────────────────────────────────────────────────────────
    lines = [
        "# SAR UAV RL Evaluation Report",
        "",
        f"> **Last updated:** {ts}  |  **Grid:** {grid_size}×{grid_size}  "
        f"|  **Episodes per agent:** {n_episodes}",
        "",
        "---",
        "",
        "## Dashboard",
        "",
        f"![Evaluation Dashboard]({img_rel})",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Agent | Type | Success Rate | Avg Reward | Avg Steps | Coverage | Battery Left | Episodes |",
        "|-------|------|-------------|------------|-----------|----------|-------------|----------|",
    ]

    for name, m in all_entries:
        agent_type = "RL" if m.get("agent_type") == "rl" else "Baseline"
        sr_str = (
            f"**{pct(m['success_rate'], m['success_rate_std'])}**"
            if m.get("agent_type") == "rl" and name == best_sr[0]
            else pct(m["success_rate"], m["success_rate_std"])
        )
        lines.append(
            f"| {name} | {agent_type} "
            f"| {sr_str} "
            f"| {flt(m['avg_reward'], m['avg_reward_std'], dp=1)} "
            f"| {flt(m['avg_steps'], m['avg_steps_std'], dp=1)} "
            f"| {pct(m['avg_coverage'], m['avg_coverage_std'])} "
            f"| {flt(m['avg_battery_remaining'], m['avg_battery_remaining_std'], dp=1)}% "
            f"| {m['n_successful']}/{m['n_episodes']} |"
        )

    lines += ["", "---", "", "## Best Performers", ""]

    if rl_entries:
        lines += [
            f"- **Highest Success Rate:** {best_sr[0]} ({pct(best_sr[1]['success_rate'])})",
            f"- **Fastest Search:**        {best_time[0]} ({flt(best_time[1]['avg_search_time'])} avg steps)",
            f"- **Best Coverage:**         {best_cov[0]} ({pct(best_cov[1]['avg_coverage'])})",
            f"- **Best Battery Efficiency:** {best_bat[0]} ({flt(best_bat[1]['avg_battery_remaining'])}% remaining)",
        ]
    else:
        lines.append("_No RL models evaluated yet – run training first._")

    lines += ["", "---", "", "## Improvement vs Baselines", ""]

    if rl_entries and any(x is not None for x in [grid_rate, spiral_rate, prob_rate]):
        lines += [
            "| Agent | Algorithm | vs Grid | vs Spiral | vs Probability |",
            "|-------|-----------|---------|-----------|----------------|",
        ]
        for name, m in rl_entries:
            lines.append(
                f"| {name} | {m.get('algorithm', '?')} "
                f"| {_improvement_str(m['success_rate'], grid_rate)} "
                f"| {_improvement_str(m['success_rate'], spiral_rate)} "
                f"| {_improvement_str(m['success_rate'], prob_rate)} |"
            )
    else:
        lines.append("_Baseline comparison unavailable._")

    lines += [
        "",
        "---",
        "",
        "## Evaluation Settings",
        "",
        f"- **Grid Size:** {grid_size}×{grid_size}",
        f"- **Episodes per Agent:** {n_episodes}",
        f"- **Random Seed:** {seed}",
        "- **Environment:** Standard SAR (1 target, 10% obstacles)",
        "",
        "---",
        "",
        "> *This report is auto-generated by `evaluation/run_dashboard.py`.*  ",
        "> *Re-run to refresh, or use `--watch` for continuous updates.*",
    ]

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Report]    Saved: {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main orchestration
# ──────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    exp_dir: str,
    n_episodes: int,
    grid_size: int,
    seed: int,
    output_dir: str,
    report_path: str,
    force_reeval: bool,
    evaluate_baselines: bool,
    cache: Dict,
    cache_path: str,
) -> Dict:
    """Discover, evaluate, cache, then generate dashboard + report."""

    env_config = EnvConfig(
        grid_size=grid_size,
        render_mode=None,       # no rendering during eval
    )

    results: Dict[str, Dict] = {}

    # ── Discover RL models ────────────────────────────────────────────────────
    models = discover_models(exp_dir)
    if models:
        print(f"\n[Discovery] Found {len(models)} RL model(s):")
        for m in models:
            print(f"  • {m['name']} ({m['algorithm']}): {m['path']}")
    else:
        print(f"\n[Discovery] No trained models found in '{exp_dir}'.")

    # ── Evaluate RL models ────────────────────────────────────────────────────
    for m_info in models:
        name = m_info["name"]
        algo = m_info["algorithm"]
        path = m_info["path"]

        key = _cache_key(path, n_episodes, grid_size)
        if not force_reeval and is_cached(key, cache):
            print(f"[Cache]      {name} – using cached results")
            entry = get_cached(key, cache)
        else:
            print(f"\n[Evaluation] {name} ({algo}) – {n_episodes} episodes...")
            try:
                model = load_model(path, algo)
                policy = make_rl_policy(model)
                entry = evaluate_agent(policy, env_config, n_episodes, seed)
                entry["algorithm"] = algo
                set_cached(key, entry, cache)
                save_cache(cache_path, cache)
                rate = entry["success_rate"]
                print(f"             ✅ Success Rate: {rate:.1%}")
            except Exception as exc:
                warnings.warn(f"Failed to evaluate {name}: {exc}")
                continue

        entry["agent_type"] = "rl"
        entry.setdefault("algorithm", algo)
        results[name] = entry

    # ── Evaluate baselines ────────────────────────────────────────────────────
    if evaluate_baselines:
        for b_type, b_label in BASELINE_TYPES:
            b_key = f"baseline:{b_label}|ep={n_episodes}|gs={grid_size}"
            if not force_reeval and is_cached(b_key, cache):
                print(f"[Cache]      {b_label} – using cached results")
                entry = get_cached(b_key, cache)
            else:
                print(f"\n[Evaluation] {b_label} – {n_episodes} episodes...")
                try:
                    entry = evaluate_baseline(b_type, b_label, env_config, n_episodes, seed)
                    set_cached(b_key, entry, cache)
                    save_cache(cache_path, cache)
                    rate = entry["success_rate"]
                    print(f"             ✅ Success Rate: {rate:.1%}")
                except Exception as exc:
                    warnings.warn(f"Failed to evaluate {b_label}: {exc}")
                    continue

            entry["agent_type"] = "baseline"
            entry["algorithm"]  = "Baseline"
            results[b_label]    = entry

    # ── Generate outputs ──────────────────────────────────────────────────────
    if not results:
        print("\n[Warning] No results to display.")
        return results

    os.makedirs(output_dir, exist_ok=True)
    dashboard_path = os.path.join(output_dir, "dashboard.png")

    print()
    generate_dashboard(results, dashboard_path, grid_size, n_episodes)
    generate_markdown(results, dashboard_path, report_path, grid_size, n_episodes, seed)

    # ── Print summary ─────────────────────────────────────────────────────────
    rl_results = [(k, v) for k, v in results.items() if v.get("agent_type") == "rl"]
    if rl_results:
        best_name, best = max(rl_results, key=lambda x: x[1]["success_rate"])
        base_results = [(k, v) for k, v in results.items() if v.get("agent_type") == "baseline"]
        best_base_sr = max((v["success_rate"] for _, v in base_results), default=0.0)
        improvement = best["success_rate"] - best_base_sr
        sign = "+" if improvement >= 0 else ""
        print(f"\n{'─'*60}")
        print(f"Best RL Agent : {best_name}")
        print(f"Success Rate  : {best['success_rate']:.1%}")
        print(f"vs Best Baseline: {sign}{improvement:.1%}")
        print(f"{'─'*60}\n")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 8. Watch mode
# ──────────────────────────────────────────────────────────────────────────────

def watch_mode(
    exp_dir: str,
    n_episodes: int,
    grid_size: int,
    seed: int,
    output_dir: str,
    report_path: str,
    evaluate_baselines: bool,
    cache_path: str,
    interval: int,
) -> None:
    """Continuously poll for new models and re-run evaluation."""
    print(f"\n[Watch] Monitoring '{exp_dir}' every {interval}s…  (Ctrl+C to stop)\n")

    known_paths: set = set()
    cache = load_cache(cache_path)

    while True:
        models = discover_models(exp_dir)
        current_paths = {m["path"] for m in models}
        new_paths = current_paths - known_paths

        if new_paths or not known_paths:
            if new_paths:
                print(f"\n[Watch] New model(s) detected:")
                for p in new_paths:
                    print(f"  • {p}")
            else:
                print(f"[Watch] Initial scan – running evaluation…")

            cache = load_cache(cache_path)  # reload in case externally updated
            run_full_evaluation(
                exp_dir=exp_dir,
                n_episodes=n_episodes,
                grid_size=grid_size,
                seed=seed,
                output_dir=output_dir,
                report_path=report_path,
                force_reeval=False,
                evaluate_baselines=evaluate_baselines,
                cache=cache,
                cache_path=cache_path,
            )
            known_paths = current_paths
            print(f"\n[Watch] Dashboard updated. Next check in {interval}s…")

        time.sleep(interval)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation dashboard for all SAR trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-shot evaluation
  python evaluation/run_dashboard.py

  # 100 evaluation episodes
  python evaluation/run_dashboard.py --episodes 100

  # Force re-evaluate (ignore cache)
  python evaluation/run_dashboard.py --force-reeval

  # Auto-update whenever a new model appears (every 60 s)
  python evaluation/run_dashboard.py --watch --watch-interval 60

  # Custom experiment directory
  python evaluation/run_dashboard.py --exp-dir my_experiments/runs
        """,
    )

    parser.add_argument("--exp-dir",        default="experiments/runs",
                        help="Experiment runs root directory (default: experiments/runs)")
    parser.add_argument("--episodes",  "-n", type=int, default=50,
                        help="Episodes per agent (default: 50)")
    parser.add_argument("--grid-size",       type=int, default=10,
                        help="Grid size (default: 10)")
    parser.add_argument("--seed",            type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--output-dir",      default="evaluation",
                        help="Output directory for dashboard image (default: evaluation/)")
    parser.add_argument("--report",          default="EVALUATION_REPORT.md",
                        help="Output markdown report path (default: EVALUATION_REPORT.md)")
    parser.add_argument("--force-reeval",    action="store_true",
                        help="Ignore cache and re-evaluate all agents")
    parser.add_argument("--watch",           action="store_true",
                        help="Watch mode: auto-update when new models appear")
    parser.add_argument("--watch-interval",  type=int, default=30,
                        help="Seconds between watch checks (default: 30)")
    parser.add_argument("--no-baselines",    action="store_true",
                        help="Skip baseline agent evaluation")

    args = parser.parse_args()

    cache_path = os.path.join(args.output_dir, "results_cache.json")
    cache = load_cache(cache_path)

    if args.watch:
        watch_mode(
            exp_dir=args.exp_dir,
            n_episodes=args.episodes,
            grid_size=args.grid_size,
            seed=args.seed,
            output_dir=args.output_dir,
            report_path=args.report,
            evaluate_baselines=not args.no_baselines,
            cache_path=cache_path,
            interval=args.watch_interval,
        )
    else:
        run_full_evaluation(
            exp_dir=args.exp_dir,
            n_episodes=args.episodes,
            grid_size=args.grid_size,
            seed=args.seed,
            output_dir=args.output_dir,
            report_path=args.report,
            force_reeval=args.force_reeval,
            evaluate_baselines=not args.no_baselines,
            cache=cache,
            cache_path=cache_path,
        )


if __name__ == "__main__":
    main()
