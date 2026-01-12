# 🚀 Advanced Features Guide

This guide covers the advanced features of the SAR Drone RL system: 3D environments, physics simulation, multi-agent coordination, and real drone deployment.

## Table of Contents

1. [3D Environment](#3d-environment)
2. [PyBullet Physics Simulation](#pybullet-physics-simulation)
3. [Multi-Agent Coordination](#multi-agent-coordination)
4. [Real Drone Deployment](#real-drone-deployment)

---

## 3D Environment

The 3D SAR environment extends the 2D grid to include altitude control, making it more realistic for actual UAV operations.

### Features

- **3D Grid Space**: Navigate in X, Y, and Z dimensions
- **Altitude Control**: 27 actions (3×3×3 movement grid)
- **Vertical Sensor Range**: Configurable altitude observation
- **Altitude-Dependent Detection**: Detection probability decreases with altitude
- **Battery Costs**: Extra battery consumption for climbing/descending

### Usage

```python
from envs import SAREnv3D, EnvConfig

# Create 3D environment
config = EnvConfig(grid_size=10, num_targets=2)
env = SAREnv3D(
    config,
    max_altitude=5,           # Maximum altitude level
    altitude_sensor_range=2   # Vertical sensor range
)

# Reset and run
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # 27 possible actions
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Observation Space

```python
{
    "position": (x, y, z),              # 3D position
    "battery": float,                    # Battery level
    "altitude": float,                   # Current altitude
    "local_occupancy": [7, 7, 5, 4],    # 3D local view
    "coverage_map": [N, N, Z],          # 3D coverage
    "belief_map": [N, N],               # 2D ground belief
    "targets_found": int
}
```

### Action Space

- **Discrete(27)**: All combinations of {-1, 0, +1} for (dx, dy, dz)
- Actions 0-8: Move at current altitude
- Actions 9-17: Climb and move
- Actions 18-26: Descend and move

### Example

```bash
python examples/example_3d_env.py
```

---

## PyBullet Physics Simulation

Integrate with Gym-Pybullet-Drones for realistic drone physics including aerodynamics, wind, and hardware validation.

### Installation

```bash
pip install gym-pybullet-drones pybullet
```

### Features

- **Realistic Physics**: Aerodynamics, thrust dynamics, wind effects
- **Multiple Drone Models**: Crazyflie 2.X, Crazyflie 2.P, Hummingbird
- **Physics Engines**: Multiple fidelity levels (PYB, DYN, etc.)
- **Battery Physics**: Thrust-based battery consumption
- **Waypoint Navigation**: High-level control abstraction

### Usage

```python
from envs import create_pybullet_sar_env, EnvConfig

# Create physics-based environment
config = EnvConfig(grid_size=10, num_targets=1)
env = create_pybullet_sar_env(
    config,
    drone_model="cf2x",    # Crazyflie 2.X
    physics="pyb",          # Physics engine
    gui=True                # Show GUI
)

# Run mission
obs, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()  # Waypoint actions
    obs, reward, terminated, truncated, info = env.step(action)
```

### Drone Models

- **cf2x**: Crazyflie 2.X (small, agile)
- **cf2p**: Crazyflie 2.P (with propeller guards)
- **hb**: Hummingbird (larger platform)

### Physics Engines

- **pyb**: Standard PyBullet physics
- **dyn**: Dynamics-based simulation
- **pyb_drag**: PyBullet with drag
- **pyb_dw**: PyBullet with downwash
- **pyb_gnd_drag_dw**: Full physics (ground effect, drag, downwash)

### Training with Physics

```bash
python training/train.py \
    --algorithm SAC \
    --config experiments/configs/sac_pybullet.yaml \
    --timesteps 1000000
```

---

## Multi-Agent Coordination

Coordinate multiple UAVs to search collaboratively with shared information and collision avoidance.

### Features

- **N Agents**: Configurable number of drones (default: 3)
- **Shared Information**: Common coverage and belief maps
- **Communication**: Agents know each other's positions and battery
- **Collision Penalties**: Discourage agents from occupying same cell
- **Team Rewards**: All agents rewarded for target discovery
- **Decentralized Control**: Each agent has its own policy

### Usage

```python
from envs import MultiAgentSAREnv, EnvConfig

# Create multi-agent environment
config = EnvConfig(grid_size=15, num_targets=3)
env = MultiAgentSAREnv(config, num_agents=3)

# Reset
observations, info = env.reset()  # List of obs per agent

# Step with multiple actions
actions = (action_agent0, action_agent1, action_agent2)
observations, rewards, terminated, truncated, info = env.step(actions)
```

### Observation Space (Per Agent)

Each agent receives:
```python
{
    "position": own_position,
    "battery": own_battery,
    "local_occupancy": what_agent_sees,
    "coverage_map": shared_coverage,      # All agents contribute
    "belief_map": shared_belief,           # Shared probability map
    "targets_found": team_total,
    "agent_positions": all_agent_positions, # Communication
    "agent_batteries": all_agent_batteries
}
```

### Training Multi-Agent Policies

```python
# Train with centralized training, decentralized execution (CTDE)
from stable_baselines3 import PPO

# Wrap environment for single-agent training
# (Advanced: use MAPPO or IPPO for true multi-agent RL)
```

### Example

```bash
python examples/example_multi_agent.py
```

---

## Real Drone Deployment

Deploy trained RL agents on actual drones. Supports Crazyflie, MAVLink (ArduPilot/PX4), and simulated testing.

### Supported Platforms

1. **Crazyflie 2.X** (via cflib)
2. **ArduPilot/PX4** (via DroneKit/MAVLink)
3. **Simulated** (for testing without hardware)

### Installation

**For Crazyflie:**
```bash
pip install cflib
```

**For MAVLink:**
```bash
pip install dronekit pymavlink
```

### Drone Interfaces

```python
from deployment import create_drone_interface

# Simulated (no hardware)
drone = create_drone_interface("simulated")

# Real Crazyflie
drone = create_drone_interface(
    "crazyflie",
    uri="radio://0/80/2M/E7E7E7E7E7"
)

# MAVLink drone
drone = create_drone_interface(
    "mavlink",
    connection_string="udp:127.0.0.1:14550"
)

# Use drone
drone.connect()
drone.takeoff(altitude=1.5)
drone.move_to(x=2.0, y=2.0, z=1.5)
drone.land()
drone.disconnect()
```

### Deploy Trained Agent

```bash
# Test with simulated drone
python deployment/deploy_sar_agent.py \
    --model experiments/runs/SAC_*/best_models/best_model_0.85.zip \
    --algorithm SAC \
    --drone-type simulated \
    --altitude 1.5 \
    --grid-size 10

# Deploy on Crazyflie
python deployment/deploy_sar_agent.py \
    --model path/to/model.zip \
    --algorithm SAC \
    --drone-type crazyflie \
    --connection "radio://0/80/2M" \
    --altitude 1.0 \
    --grid-scale 1.5  # meters per grid cell

# Deploy on MAVLink drone
python deployment/deploy_sar_agent.py \
    --model path/to/model.zip \
    --algorithm SAC \
    --drone-type mavlink \
    --connection "/dev/ttyUSB0" \
    --altitude 2.0 \
    --grid-scale 2.0
```

### Safety Considerations

⚠️ **IMPORTANT**: When deploying on real hardware:

1. **Test in Simulation First**: Always validate with simulated drone
2. **Start Small**: Begin with small grid sizes and low altitudes
3. **Battery Monitoring**: Set conservative battery thresholds (>20%)
4. **Emergency Stop**: Keep RC controller ready for manual override
5. **Indoor Testing**: Test indoors with safety net before outdoor deployment
6. **Check Regulations**: Comply with local drone laws (registration, airspace)
7. **GPS Requirements**: MAVLink drones need GPS lock for outdoor flights

### Coordinate Transformations

The deployment system handles grid-to-world coordinate conversion:

```python
# Grid: discrete cells (0, 1, 2, ...)
# World: continuous meters (0.0, 2.0, 4.0, ...)

grid_scale = 2.0  # meters per grid cell

# Grid (3, 5) -> World (6.0m, 10.0m)
world_x = grid_x * grid_scale
world_y = grid_y * grid_scale
```

### Example Deployment

```bash
python examples/example_deployment.py
```

---

## Complete Workflow

### 1. Train in 2D

```bash
python training/train.py --algorithm SAC --grid-size 10 --timesteps 500000
```

### 2. Scale to 3D

```bash
# Test 3D environment
python examples/example_3d_env.py

# Train in 3D (requires custom training script)
# See examples/train_3d.py
```

### 3. Add Physics

```bash
# Install PyBullet
pip install gym-pybullet-drones pybullet

# Train with physics
# (Requires custom config - see experiments/configs/sac_pybullet.yaml)
```

### 4. Multi-Agent Coordination

```bash
# Test multi-agent
python examples/example_multi_agent.py

# Train multi-agent (requires MAPPO/IPPO)
# (Advanced feature - see multi-agent RL literature)
```

### 5. Deploy on Hardware

```bash
# Test with simulation
python deployment/deploy_sar_agent.py \
    --model path/to/best_model.zip \
    --algorithm SAC \
    --drone-type simulated

# Deploy on real drone (Crazyflie example)
python deployment/deploy_sar_agent.py \
    --model path/to/best_model.zip \
    --algorithm SAC \
    --drone-type crazyflie \
    --connection "radio://0/80/2M" \
    --altitude 1.0
```

---

## Tips for Success

### 3D Training

- Start with low `max_altitude` (3-5 levels)
- Increase `initial_battery` (3D uses more energy)
- Use `reward_prob_gradient` to encourage smart altitude choices
- Render periodically to visualize 3D search patterns

### Physics Simulation

- Start with `physics="pyb"` (simplest)
- Add complexity gradually (drag, downwash)
- Increase training timesteps (2M+)
- Use slower learning rates (1e-4)

### Multi-Agent

- Use team rewards (all agents benefit from any target found)
- Add collision penalties (-100)
- Start with 2-3 agents, scale up
- Consider communication radius limits for realism

### Real Deployment

- Always test with `drone_type="simulated"` first
- Use conservative `grid_scale` (start small)
- Monitor battery throughout flight
- Have manual override ready
- Start indoors with safety barriers

---

## Troubleshooting

### 3D Environment

**Issue**: Agent always stays at one altitude
**Solution**: Increase altitude change rewards, ensure battery sufficient

**Issue**: Poor 3D visualization
**Solution**: Use `render_mode="human"` and rotate 3D plot interactively

### PyBullet

**Issue**: `ImportError: No module named 'gym_pybullet_drones'`
**Solution**: `pip install gym-pybullet-drones pybullet`

**Issue**: Simulation too slow
**Solution**: Reduce `freq` parameter, disable GUI during training

### Multi-Agent

**Issue**: Agents collide frequently
**Solution**: Increase collision penalty, add spacing in initial positions

**Issue**: Agents don't coordinate
**Solution**: Ensure shared coverage/belief maps, consider explicit communication

### Deployment

**Issue**: `cflib` not found
**Solution**: `pip install cflib` (for Crazyflie)

**Issue**: Drone doesn't move
**Solution**: Check connection string, ensure drone is powered and paired

**Issue**: Erratic movements
**Solution**: Adjust `grid_scale` (may be too large), reduce step frequency

---

## Next Steps

- Implement **curriculum learning** (start simple, increase complexity)
- Add **vision-based detection** (integrate camera for real target detection)
- Implement **SLAM** (simultaneous localization and mapping)
- Add **dynamic targets** (moving targets for more realistic scenarios)
- Implement **swarm intelligence** (emergent multi-agent behaviors)

---

**Congratulations!** You now have a complete SAR drone system that works from simulation to real hardware. 🚁✨
