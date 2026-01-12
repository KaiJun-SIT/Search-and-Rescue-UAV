"""SAR UAV Environment package."""

from envs.sar_env import SAREnv
from envs.config import EnvConfig
from envs.sar_env_3d import SAREnv3D
from envs.multi_agent_sar_env import MultiAgentSAREnv

# Optional: PyBullet environment (requires gym-pybullet-drones)
try:
    from envs.pybullet_sar_env import PyBulletSAREnv, create_pybullet_sar_env
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    PyBulletSAREnv = None
    create_pybullet_sar_env = None

__all__ = [
    "SAREnv",
    "SAREnv3D",
    "MultiAgentSAREnv",
    "EnvConfig",
]

if PYBULLET_AVAILABLE:
    __all__.extend(["PyBulletSAREnv", "create_pybullet_sar_env"])
