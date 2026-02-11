from envs import EnvConfig, SAREnv
import time

# Create config with render_mode enabled
config = EnvConfig(
    grid_size=10,
    sensor_radius=3,
    render_mode="human"  # This enables visualization!
)

env = SAREnv(config)

# Run simulation
obs, info = env.reset()
print("Starting visual simulation...")
print("Press Ctrl+C to stop")

try:
    for step in range(500):
        # Random action for demo (or use trained agent)
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        # Slow down so you can see what's happening
        time.sleep(0.1)
        
        if terminated or truncated:
            print(f"\nEpisode finished at step {step}")
            print(f"Success: {info.get('success', False)}")
            print(f"Targets found: {info.get('targets_found', 0)}")
            break
            
except KeyboardInterrupt:
    print("\nStopping simulation...")
    
finally:
    env.close()