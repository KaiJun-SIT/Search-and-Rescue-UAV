from envs import SAREnv, EnvConfig

config = EnvConfig(
    grid_size=10, 
    num_targets=1,
    render_mode="human"  # ← Add this!
)
env = SAREnv(config)

obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # ← Shows visualization
    if done or truncated:
        break

env.close()
