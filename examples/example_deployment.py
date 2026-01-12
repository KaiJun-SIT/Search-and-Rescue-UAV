"""Example: Deploying trained agent on simulated drone.

Demonstrates how to load a trained model and deploy it on a simulated drone.
For real deployment, change drone_type to "crazyflie" or "mavlink".
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from envs import EnvConfig
from deployment.drone_interface import create_drone_interface


def demonstrate_simulated_flight():
    """Demonstrate basic flight with simulated drone."""
    print("\n" + "="*60)
    print("Simulated Drone Deployment Example")
    print("="*60)

    # Create simulated drone
    drone = create_drone_interface("simulated")

    # Connect
    print("\n1. Connecting to drone...")
    if not drone.connect():
        print("❌ Connection failed")
        return

    # Takeoff
    print("\n2. Taking off...")
    if not drone.takeoff(altitude=1.5):
        print("❌ Takeoff failed")
        drone.disconnect()
        return

    # Execute simple search pattern
    print("\n3. Executing search pattern...")
    waypoints = [
        (2.0, 0.0, 1.5),   # Move east
        (2.0, 2.0, 1.5),   # Move north
        (0.0, 2.0, 1.5),   # Move west
        (0.0, 0.0, 1.5),   # Return to start
    ]

    for i, (x, y, z) in enumerate(waypoints):
        print(f"  Waypoint {i+1}: Moving to ({x}m, {y}m, {z}m)")
        drone.move_to(x, y, z, speed=1.0)

        pos = drone.get_position()
        battery = drone.get_battery()
        print(f"    Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        print(f"    Battery: {battery:.1f}%")

    # Land
    print("\n4. Landing...")
    drone.land()

    # Disconnect
    print("\n5. Disconnecting...")
    drone.disconnect()

    print("\n✅ Mission completed successfully!")
    print("="*60)


def demonstrate_rl_agent_deployment():
    """Demonstrate RL agent deployment (requires trained model)."""
    print("\n" + "="*60)
    print("RL Agent Deployment Example")
    print("="*60)

    print("\nTo deploy a trained RL agent on a real drone:")
    print("\n1. Train a model first:")
    print("   python training/train.py --algorithm SAC --grid-size 10 --timesteps 500000")

    print("\n2. Deploy on simulated drone:")
    print("   python deployment/deploy_sar_agent.py \\")
    print("       --model experiments/runs/SAC_*/best_models/best_model_*.zip \\")
    print("       --algorithm SAC \\")
    print("       --drone-type simulated \\")
    print("       --altitude 1.5")

    print("\n3. Deploy on real Crazyflie:")
    print("   python deployment/deploy_sar_agent.py \\")
    print("       --model experiments/runs/SAC_*/best_models/best_model_*.zip \\")
    print("       --algorithm SAC \\")
    print("       --drone-type crazyflie \\")
    print("       --connection 'radio://0/80/2M' \\")
    print("       --altitude 1.0 \\")
    print("       --grid-scale 1.5")

    print("\n4. Deploy on MAVLink drone (ArduPilot/PX4):")
    print("   python deployment/deploy_sar_agent.py \\")
    print("       --model experiments/runs/SAC_*/best_models/best_model_*.zip \\")
    print("       --algorithm SAC \\")
    print("       --drone-type mavlink \\")
    print("       --connection 'udp:127.0.0.1:14550' \\")
    print("       --altitude 2.0")

    print("\n" + "="*60)


if __name__ == "__main__":
    # Run simulated flight demo
    demonstrate_simulated_flight()

    # Show RL agent deployment instructions
    print("\n")
    demonstrate_rl_agent_deployment()
