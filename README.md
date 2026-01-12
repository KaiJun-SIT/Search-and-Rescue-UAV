# 🚁 SAR Drone RL - Deep Reinforcement Learning for Search and Rescue

A comprehensive deep reinforcement learning system for autonomous UAV search and rescue missions. Train RL agents (PPO, SAC, TD3) to search for missing persons faster than traditional grid patterns using partial observability and Bayesian belief maps.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project implements a custom Gymnasium environment for SAR missions with:
- **Partial observability** - Drone only sees within sensor radius (3-5 cells)
- **Battery constraints** - Limited flight time
- **Bayesian belief maps** - Probabilistic target location tracking
- **Multi-target scenarios** - Find multiple targets with different priorities
- **Obstacle avoidance** - Navigate around blocked areas

### Key Features

✅ **Custom Gymnasium Environment**
- Configurable grid sizes (10×10 to 30×30)
- Partial observability with sensor radius
- Bayesian belief map updates with numerical stability
- Battery mechanics with proper termination
- Rich observation space (position, battery, local view, coverage, belief map)

✅ **RL Algorithms (Stable-Baselines3)**
- **PPO** - Proximal Policy Optimization
- **SAC** - Soft Actor-Critic (with automatic entropy tuning)
- **TD3** - Twin Delayed DDPG

✅ **Baseline Algorithms**
- Grid Search (lawnmower pattern)
- Spiral Search (expanding from center)
- Random Search
- Probability-Weighted Search

✅ **Training Infrastructure**
- Custom feature extractors for Dict observation spaces
- Tensorboard logging
- Automatic checkpointing
- Best model saving
- Progress tracking

✅ **Evaluation Framework**
- Compare all methods systematically
- Generate comparison plots
- Export results to CSV
- Statistical analysis

## 📁 Project Structure

```
sar-rl-project/
├── envs/
│   ├── __init__.py
│   ├── config.py              # EnvConfig dataclass
│   └── sar_env.py             # Main SAR Gymnasium environment
├── agents/
│   ├── __init__.py
│   ├── baselines.py           # Baseline search algorithms
│   └── rl_agent.py            # Feature extractors for SB3
├── training/
│   ├── __init__.py
│   ├── callbacks.py           # Custom training callbacks
│   └── train.py               # Main training script
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py            # Evaluation framework
├── experiments/
│   └── configs/               # YAML training configs
│       ├── ppo_10x10.yaml
│       ├── sac_10x10.yaml
│       └── sac_30x30.yaml
├── tests/
│   └── test_environment.py    # Environment tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Search-and-Rescue-UAV

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Test the Environment

```bash
# Run environment tests
python tests/test_environment.py
```

You should see output confirming all tests pass:
```
Testing SAR Environment
====================================================================
✓ Environment created successfully
✓ Environment reset successfully
✓ Observation space validation passed
...
✅ All tests passed!
```

### Train an Agent

#### Quick Start (Default Config)

```bash
# Train PPO on 10x10 grid
python training/train.py --algorithm PPO --grid-size 10 --timesteps 500000

# Train SAC on 10x10 grid
python training/train.py --algorithm SAC --grid-size 10 --timesteps 500000

# Train TD3 on 10x10 grid
python training/train.py --algorithm TD3 --grid-size 10 --timesteps 500000
```

#### Using Config Files

```bash
# Train with YAML config
python training/train.py --algorithm SAC --config experiments/configs/sac_10x10.yaml

# Train on larger grid with multiple targets
python training/train.py --algorithm SAC --config experiments/configs/sac_30x30.yaml
```

#### Custom Experiment

```bash
python training/train.py \
    --algorithm SAC \
    --grid-size 20 \
    --timesteps 1000000 \
    --exp-name "sac_20x20_experiment" \
    --seed 42
```

### Monitor Training

Training logs to Tensorboard automatically:

```bash
# In a separate terminal
tensorboard --logdir experiments/runs/
```

Open `http://localhost:6006` to view:
- Success rate over time
- Average search time
- Coverage efficiency
- Battery usage
- Reward curves

### Evaluate Agents

#### Evaluate Baselines Only

```bash
python evaluation/evaluate.py \
    --baselines grid spiral probability random \
    --grid-size 10 \
    --n-episodes 100 \
    --save-dir experiments/evaluation/baselines
```

#### Compare RL Agent vs Baselines

```bash
python evaluation/evaluate.py \
    --rl-agents "my_sac:SAC:experiments/runs/my_sac/best_models/best_model_0.85.zip" \
    --baselines grid spiral probability \
    --grid-size 10 \
    --n-episodes 100 \
    --save-dir experiments/evaluation/sac_vs_baselines
```

This generates:
- `comparison_results.csv` - Detailed metrics table
- `comparison_plots.png` - Visual comparison
- Console output with statistical summary

## 📊 Understanding the Environment

### Observation Space

The environment provides a Dict observation with:

```python
{
    "position": (x, y),                    # Agent's grid position
    "battery": float,                       # 0-100% battery level
    "local_occupancy": 7×7×3 array,        # Partial observation within sensor radius
                                           # Channel 0: obstacles
                                           # Channel 1: visited cells
                                           # Channel 2: current position
    "coverage_map": N×N array,             # Complete history of visited cells
    "belief_map": N×N array,               # Bayesian probability of target location
    "targets_found": int                    # Number of targets found so far
}
```

### Action Space

Discrete(9) actions:
- 0: North
- 1: Northeast
- 2: East
- 3: Southeast
- 4: South
- 5: Southwest
- 6: West
- 7: Northwest
- 8: Stay

### Reward Structure

```python
Rewards:
  +1000 * priority   # Target found (scaled by target priority)
  +1                 # New cell visited (exploration bonus)
  -1                 # Time penalty (every step)
  -500               # Battery depleted (failure)
  +5 * Δprob         # Moving toward high-probability regions
```

### Termination Conditions

Episode ends when:
- ✅ **Success**: All targets found
- ❌ **Failure**: Battery depleted (reaches 0)
- ⏱️ **Truncated**: Max steps reached (default: 5 × grid_size²)

## 🔧 Configuration

### Environment Configuration

Create custom configs in `experiments/configs/`:

```yaml
env:
  grid_size: 10
  sensor_radius: 3
  initial_battery: 100.0
  battery_depletion_rate: 1.0
  max_steps: 500
  num_targets: 1
  target_priorities: [1.0]  # Optional: different priorities per target
  obstacle_density: 0.1
  detection_probability: 0.95

  # Reward weights (tune these!)
  reward_target_found: 1000.0
  reward_new_coverage: 1.0
  reward_time_penalty: -1.0
  reward_battery_depleted: -500.0
  reward_prob_gradient: 5.0

training:
  total_timesteps: 500000
  n_envs: 4
  feature_extractor: "compact"  # or "standard" for larger networks
  features_dim: 128
  learning_rate: 0.0003

  # Evaluation
  eval_freq: 10000
  n_eval_episodes: 20
  checkpoint_freq: 50000
```

### Algorithm-Specific Parameters

**PPO:**
```yaml
n_steps: 2048
batch_size: 64
n_epochs: 10
ent_coef: 0.01  # Entropy coefficient for exploration
```

**SAC:**
```yaml
buffer_size: 100000
batch_size: 256
ent_coef: "auto"  # Automatic entropy tuning (recommended!)
train_freq: 1
gradient_steps: 1
```

**TD3:**
```yaml
buffer_size: 100000
batch_size: 256
policy_delay: 2
train_freq: 1
gradient_steps: 1
```

## 📈 Expected Results

Based on Ewers et al. (2024) - "Deep RL for time-critical wilderness SAR using drones":

| Metric | Baseline (Grid) | Target (SAC) | Your Goal |
|--------|----------------|--------------|-----------|
| **Success Rate** | ~70% | 91.2% | 85%+ |
| **Avg Search Time** | 287 steps | 138 steps | 50% improvement |
| **Coverage** | ~95% | ~60% | Efficient |

### Tuning Tips

If your agent isn't learning well:

1. **Low success rate (<50%)**:
   - Increase `reward_prob_gradient` (helps agent use belief map)
   - Decrease `reward_time_penalty` (less pressure early on)
   - Increase `ent_coef` for more exploration

2. **High success but slow search**:
   - Increase `reward_time_penalty` (more urgency)
   - Reduce `reward_new_coverage` (less random exploration)

3. **Agent runs out of battery**:
   - Increase `initial_battery` or decrease `battery_depletion_rate`
   - Add battery-aware rewards

4. **Training unstable**:
   - Reduce learning rate
   - Use smaller batch sizes
   - Try PPO instead of SAC/TD3

## 🧪 Advanced Usage

### Custom Feature Extractors

The environment uses Dict observations. Two feature extractors are provided:

- **CompactSARFeatureExtractor**: Fast, lightweight (good for 10×10 grids)
- **SARFeatureExtractor**: Larger, more expressive (use for 30×30 grids)

Modify in `agents/rl_agent.py` to customize architecture.

### Multi-Target Scenarios

```python
from envs import EnvConfig, SAREnv

config = EnvConfig(
    grid_size=20,
    num_targets=3,
    target_priorities=[1.0, 1.5, 2.0],  # Higher priority = more reward
)

env = SAREnv(config)
```

### Visualization

```python
from envs import SAREnv, EnvConfig

config = EnvConfig(grid_size=10, render_mode="human")
env = SAREnv(config)

obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        break

env.close()
```

## 📚 Implementation Details

### Partial Observability

The agent only sees within `sensor_radius` cells. Implemented in `envs/sar_env.py:_get_local_observation()`:

```python
# Returns 7×7×3 array centered on agent (for sensor_radius=3)
# Cells outside bounds filled with 0.5 (out-of-bounds indicator)
# Agent maintains full memory via coverage_map
```

### Bayesian Belief Updates

When agent scans an area without finding a target:

```python
P(target | no detection) = P(target) × (1 - P(detection | present))
# Then renormalize to sum to 1
```

Implemented with numerical stability in `envs/sar_env.py:_update_belief_map()`.

### Battery Mechanics

- Depletes `battery_depletion_rate` per step
- Episode terminates when battery ≤ 0
- Large penalty applied on depletion

## 🤝 Contributing

This is a capstone project. Feel free to:
- Report bugs
- Suggest improvements
- Share your training results
- Extend to 3D environments or physics simulators

## 📖 References

- **Ewers et al. (2024)** - "Deep RL for time-critical wilderness SAR using drones"
  - SAC with continuous PDMs achieved 91.2% success
  - 52% faster than grid search (138s vs 287s)
  - Entropy maximization critical for exploration

- **Stable-Baselines3** - https://stable-baselines3.readthedocs.io/
- **Gymnasium** - https://gymnasium.farama.org/

## 📝 Citation

If you use this code in your research:

```bibtex
@misc{sar-rl-uav,
  title={Deep Reinforcement Learning for Autonomous UAV Search and Rescue},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/Search-and-Rescue-UAV}}
}
```

## 📄 License

MIT License - See LICENSE file for details

---

## 🎯 Roadmap

- [x] Custom Gymnasium environment
- [x] Partial observability
- [x] Bayesian belief maps
- [x] Battery mechanics
- [x] RL algorithms (PPO, SAC, TD3)
- [x] Baseline algorithms
- [x] Training pipeline
- [x] Evaluation framework
- [x] **3D environment support** ✨ NEW!
- [x] **Gym-Pybullet-Drones integration** ✨ NEW!
- [x] **Real drone deployment** ✨ NEW!
- [x] **Multi-agent coordination** ✨ NEW!

---

## 🌟 Advanced Features

This project now includes cutting-edge capabilities for real-world SAR deployment!

### 🆕 3D Environment (`SAREnv3D`)
- Full 3D navigation with altitude control
- 27 actions (3×3×3 movement grid)
- Altitude-dependent detection and battery costs
- Example: `python examples/example_3d_env.py`

### 🆕 Physics Simulation (`PyBulletSAREnv`)
- Realistic aerodynamics via Gym-Pybullet-Drones
- Multiple drone models (Crazyflie, Hummingbird)
- Wind effects and thrust dynamics
- Hardware validation before deployment

### 🆕 Multi-Agent Coordination (`MultiAgentSAREnv`)
- Coordinate 2-10 UAVs simultaneously
- Shared coverage and belief maps
- Collision avoidance and team rewards
- Example: `python examples/example_multi_agent.py`

### 🆕 Real Drone Deployment
- Deploy trained agents on actual drones!
- Supports: **Crazyflie 2.X**, **ArduPilot/PX4**, **Simulated**
- Safety features and coordinate transformations
- Example: `python examples/example_deployment.py`

**📖 Full Guide:** See [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) for detailed documentation, examples, and deployment instructions.

---

**Ready to train some awesome SAR drones!** 🚁🔥

For questions or issues, please open a GitHub issue.
