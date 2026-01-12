"""Real Drone Interface for SAR Deployment.

Provides interfaces to deploy trained RL agents on real drones.
Supports multiple platforms:
- DJI Drones (via DJI SDK)
- ArduPilot/PX4 (via MAVLink/DroneKit)
- Crazyflie (via cflib)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
import time


class DroneInterface(ABC):
    """Abstract base class for drone interfaces."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to drone."""
        pass

    @abstractmethod
    def takeoff(self, altitude: float) -> bool:
        """Take off to specified altitude."""
        pass

    @abstractmethod
    def land(self) -> bool:
        """Land the drone."""
        pass

    @abstractmethod
    def get_position(self) -> Tuple[float, float, float]:
        """Get current position (x, y, z) in meters."""
        pass

    @abstractmethod
    def get_battery(self) -> float:
        """Get battery level (0-100)."""
        pass

    @abstractmethod
    def move_to(self, x: float, y: float, z: float, speed: float = 1.0) -> bool:
        """Move to specified position."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from drone."""
        pass


class CrazyflieInterface(DroneInterface):
    """Interface for Crazyflie 2.X drones.

    Requires cflib: pip install cflib
    """

    def __init__(self, uri: str = "radio://0/80/2M/E7E7E7E7E7"):
        """Initialize Crazyflie interface.

        Args:
            uri: Crazyflie radio URI
        """
        self.uri = uri
        self.cf = None
        self.is_connected = False

        try:
            import cflib.crtp
            from cflib.crazyflie import Crazyflie
            from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

            self.cflib = cflib
            self.Crazyflie = Crazyflie
            self.SyncCrazyflie = SyncCrazyflie
            self.cflib_available = True
        except ImportError:
            self.cflib_available = False
            print("Warning: cflib not installed. Install with: pip install cflib")

    def connect(self) -> bool:
        """Connect to Crazyflie."""
        if not self.cflib_available:
            return False

        try:
            self.cflib.crtp.init_drivers()
            self.scf = self.SyncCrazyflie(self.uri, cf=self.Crazyflie(rw_cache="./cache"))
            self.scf.open_link()
            self.cf = self.scf.cf
            self.is_connected = True
            print(f"Connected to Crazyflie at {self.uri}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def takeoff(self, altitude: float) -> bool:
        """Take off to altitude."""
        if not self.is_connected:
            return False

        try:
            from cflib.positioning.motion_commander import MotionCommander

            mc = MotionCommander(self.scf)
            mc.take_off(altitude)
            self.mc = mc
            return True
        except Exception as e:
            print(f"Takeoff failed: {e}")
            return False

    def land(self) -> bool:
        """Land drone."""
        if hasattr(self, "mc"):
            self.mc.land()
            return True
        return False

    def get_position(self) -> Tuple[float, float, float]:
        """Get current position from state estimate."""
        if not self.is_connected:
            return (0.0, 0.0, 0.0)

        # Read from log TOC (if available)
        # This is a simplified version - real implementation would use LogConfig
        return (0.0, 0.0, 0.0)  # Placeholder

    def get_battery(self) -> float:
        """Get battery voltage and estimate percentage."""
        if not self.is_connected:
            return 0.0

        # Read from PM (power management)
        # This is simplified - real implementation would read actual battery
        return 100.0  # Placeholder

    def move_to(self, x: float, y: float, z: float, speed: float = 1.0) -> bool:
        """Move to position."""
        if not hasattr(self, "mc"):
            return False

        try:
            # Calculate movement
            current = self.get_position()
            dx = x - current[0]
            dy = y - current[1]
            dz = z - current[2]

            # Execute movement (simplified)
            self.mc.move_distance(dx, dy, dz, velocity=speed)
            return True
        except Exception as e:
            print(f"Movement failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from drone."""
        if hasattr(self, "mc"):
            self.mc.land()

        if self.is_connected:
            self.scf.close_link()
            self.is_connected = False


class MAVLinkInterface(DroneInterface):
    """Interface for ArduPilot/PX4 drones via MAVLink.

    Requires dronekit: pip install dronekit
    """

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550"):
        """Initialize MAVLink interface.

        Args:
            connection_string: Connection string (e.g., "udp:127.0.0.1:14550", "/dev/ttyUSB0")
        """
        self.connection_string = connection_string
        self.vehicle = None

        try:
            from dronekit import connect, VehicleMode
            from pymavlink import mavutil

            self.connect_fn = connect
            self.VehicleMode = VehicleMode
            self.mavutil = mavutil
            self.dronekit_available = True
        except ImportError:
            self.dronekit_available = False
            print("Warning: dronekit not installed. Install with: pip install dronekit")

    def connect(self) -> bool:
        """Connect to drone."""
        if not self.dronekit_available:
            return False

        try:
            self.vehicle = self.connect_fn(self.connection_string, wait_ready=True)
            print(f"Connected to vehicle at {self.connection_string}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def takeoff(self, altitude: float) -> bool:
        """Arm and takeoff."""
        if not self.vehicle:
            return False

        try:
            # Arm vehicle
            self.vehicle.mode = self.VehicleMode("GUIDED")
            self.vehicle.armed = True

            # Wait for arming
            while not self.vehicle.armed:
                time.sleep(1)

            # Takeoff
            self.vehicle.simple_takeoff(altitude)

            # Wait for altitude
            while True:
                current_alt = self.vehicle.location.global_relative_frame.alt
                if current_alt >= altitude * 0.95:
                    break
                time.sleep(1)

            return True
        except Exception as e:
            print(f"Takeoff failed: {e}")
            return False

    def land(self) -> bool:
        """Land drone."""
        if not self.vehicle:
            return False

        try:
            self.vehicle.mode = self.VehicleMode("LAND")
            return True
        except Exception as e:
            print(f"Landing failed: {e}")
            return False

    def get_position(self) -> Tuple[float, float, float]:
        """Get current position."""
        if not self.vehicle:
            return (0.0, 0.0, 0.0)

        loc = self.vehicle.location.global_relative_frame
        # Convert to local coordinates (simplified)
        # In real deployment, use proper coordinate transformation
        return (loc.lat, loc.lon, loc.alt)

    def get_battery(self) -> float:
        """Get battery level."""
        if not self.vehicle:
            return 0.0

        return self.vehicle.battery.level if self.vehicle.battery.level else 100.0

    def move_to(self, x: float, y: float, z: float, speed: float = 1.0) -> bool:
        """Move to position using goto."""
        if not self.vehicle:
            return False

        try:
            from dronekit import LocationGlobalRelative

            # Convert local to global (simplified)
            # In practice, use proper coordinate transformation
            point = LocationGlobalRelative(x, y, z)
            self.vehicle.simple_goto(point, groundspeed=speed)
            return True
        except Exception as e:
            print(f"Movement failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from vehicle."""
        if self.vehicle:
            self.vehicle.close()


class SimulatedDrone(DroneInterface):
    """Simulated drone for testing without hardware."""

    def __init__(self):
        """Initialize simulated drone."""
        self.position = np.array([0.0, 0.0, 0.0])
        self.battery = 100.0
        self.is_flying = False
        self.is_connected = False

    def connect(self) -> bool:
        """Simulate connection."""
        self.is_connected = True
        print("Simulated drone connected")
        return True

    def takeoff(self, altitude: float) -> bool:
        """Simulate takeoff."""
        if not self.is_connected:
            return False

        self.position[2] = altitude
        self.is_flying = True
        self.battery -= 5.0  # Takeoff costs battery
        print(f"Simulated takeoff to {altitude}m")
        return True

    def land(self) -> bool:
        """Simulate landing."""
        self.position[2] = 0.0
        self.is_flying = False
        print("Simulated landing")
        return True

    def get_position(self) -> Tuple[float, float, float]:
        """Get simulated position."""
        return tuple(self.position)

    def get_battery(self) -> float:
        """Get simulated battery."""
        return max(0.0, self.battery)

    def move_to(self, x: float, y: float, z: float, speed: float = 1.0) -> bool:
        """Simulate movement."""
        if not self.is_flying:
            return False

        # Calculate distance
        target = np.array([x, y, z])
        distance = np.linalg.norm(target - self.position)

        # Move to position
        self.position = target

        # Consume battery
        self.battery -= distance * 0.5

        print(f"Moved to ({x:.2f}, {y:.2f}, {z:.2f})")
        return True

    def disconnect(self) -> None:
        """Simulate disconnection."""
        if self.is_flying:
            self.land()
        self.is_connected = False
        print("Simulated drone disconnected")


def create_drone_interface(drone_type: str = "simulated", **kwargs) -> DroneInterface:
    """Factory function to create drone interface.

    Args:
        drone_type: Type of drone ("crazyflie", "mavlink", "simulated")
        **kwargs: Additional arguments for specific interface

    Returns:
        Drone interface instance

    Example:
        >>> # Simulated drone (no hardware needed)
        >>> drone = create_drone_interface("simulated")
        >>> drone.connect()
        >>> drone.takeoff(1.0)
        >>> drone.move_to(2.0, 2.0, 1.0)
        >>> drone.land()
        >>> drone.disconnect()

        >>> # Real Crazyflie
        >>> drone = create_drone_interface("crazyflie", uri="radio://0/80/2M")
        >>> drone.connect()
        >>> # ... fly mission ...
        >>> drone.disconnect()

        >>> # MAVLink drone
        >>> drone = create_drone_interface("mavlink", connection_string="udp:127.0.0.1:14550")
        >>> drone.connect()
        >>> # ... fly mission ...
        >>> drone.disconnect()
    """
    interfaces = {
        "crazyflie": CrazyflieInterface,
        "mavlink": MAVLinkInterface,
        "simulated": SimulatedDrone,
    }

    if drone_type not in interfaces:
        print(f"Unknown drone type: {drone_type}. Using simulated.")
        drone_type = "simulated"

    return interfaces[drone_type](**kwargs)
