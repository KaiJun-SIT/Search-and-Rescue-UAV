"""RL agent wrapper and custom feature extractors for SAR environment.

This module provides Stable-Baselines3 integration with custom feature extractors
for the Dict observation space used in the SAR environment.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


class SARFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for SAR environment Dict observation space.

    This extracts and combines features from:
    - Position (2D)
    - Battery (1D)
    - Local occupancy (7x7x3 grid - what agent sees)
    - Coverage map (NxN - where agent has been)
    - Belief map (NxN - probability distribution)
    - Targets found (1D)

    Architecture:
    - Local occupancy: CNN (3 layers)
    - Coverage map: CNN (3 layers)
    - Belief map: CNN (3 layers)
    - Scalar features: Linear layers
    - All concatenated and passed through final MLP
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        """Initialize feature extractor.

        Args:
            observation_space: Dict observation space from environment
            features_dim: Dimension of final feature vector
        """
        super().__init__(observation_space, features_dim)

        # Extract dimensions
        self.grid_size = observation_space["coverage_map"].shape[0]
        self.obs_size = observation_space["local_occupancy"].shape[0]

        # CNN for local occupancy (7x7x3 -> features)
        self.local_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate local CNN output dim
        with torch.no_grad():
            sample = torch.zeros(1, 3, self.obs_size, self.obs_size)
            local_cnn_out = self.local_cnn(sample)
            local_out_dim = local_cnn_out.shape[1]

        # CNN for coverage map (NxN -> features)
        self.coverage_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate coverage CNN output dim
        with torch.no_grad():
            sample = torch.zeros(1, 1, self.grid_size, self.grid_size)
            coverage_out = self.coverage_cnn(sample)
            coverage_out_dim = coverage_out.shape[1]

        # CNN for belief map (NxN -> features)
        self.belief_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate belief CNN output dim
        with torch.no_grad():
            sample = torch.zeros(1, 1, self.grid_size, self.grid_size)
            belief_out = self.belief_cnn(sample)
            belief_out_dim = belief_out.shape[1]

        # MLP for scalar features (position, battery, targets_found)
        scalar_dim = 2 + 1 + 1  # position(2) + battery(1) + targets_found(1)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Final combination layer
        combined_dim = local_out_dim + coverage_out_dim + belief_out_dim + 32
        self.combine_mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations.

        Args:
            observations: Dict of observation tensors

        Returns:
            Feature vector of shape (batch_size, features_dim)
        """
        # Process local occupancy (batch, obs_size, obs_size, 3) -> (batch, 3, obs_size, obs_size)
        local_obs = observations["local_occupancy"].permute(0, 3, 1, 2)
        local_features = self.local_cnn(local_obs)

        # Process coverage map (batch, grid_size, grid_size) -> (batch, 1, grid_size, grid_size)
        coverage = observations["coverage_map"].unsqueeze(1)
        coverage_features = self.coverage_cnn(coverage)

        # Process belief map (batch, grid_size, grid_size) -> (batch, 1, grid_size, grid_size)
        belief = observations["belief_map"].unsqueeze(1)
        belief_features = self.belief_cnn(belief)

        # Process scalar features
        scalars = torch.cat(
            [
                observations["position"],
                observations["battery"],
                observations["targets_found"],
            ],
            dim=1,
        )
        scalar_features = self.scalar_mlp(scalars)

        # Combine all features
        combined = torch.cat(
            [local_features, coverage_features, belief_features, scalar_features], dim=1
        )
        features = self.combine_mlp(combined)

        return features


class CompactSARFeatureExtractor(BaseFeaturesExtractor):
    """Lightweight feature extractor for faster training.

    Uses smaller networks for quicker experiments. Good for 10x10 grids.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.grid_size = observation_space["coverage_map"].shape[0]
        self.obs_size = observation_space["local_occupancy"].shape[0]

        # Simple CNN for local occupancy
        self.local_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, 3, self.obs_size, self.obs_size)
            local_out_dim = self.local_cnn(sample).shape[1]

        # Simple CNN for maps
        self.map_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, 2, self.grid_size, self.grid_size)
            map_out_dim = self.map_cnn(sample).shape[1]

        # Combine everything
        scalar_dim = 4  # position(2) + battery(1) + targets_found(1)
        combined_dim = local_out_dim + map_out_dim + scalar_dim

        self.combine = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Local observation
        local_obs = observations["local_occupancy"].permute(0, 3, 1, 2)
        local_features = self.local_cnn(local_obs)

        # Coverage and belief combined
        maps = torch.stack(
            [observations["coverage_map"], observations["belief_map"]], dim=1
        )
        map_features = self.map_cnn(maps)

        # Scalars
        scalars = torch.cat(
            [
                observations["position"],
                observations["battery"],
                observations["targets_found"],
            ],
            dim=1,
        )

        # Combine
        combined = torch.cat([local_features, map_features, scalars], dim=1)
        return self.combine(combined)


def create_policy_kwargs(
    extractor_type: str = "standard", features_dim: int = 256
) -> Dict:
    """Create policy kwargs for SB3 algorithms.

    Args:
        extractor_type: "standard" or "compact"
        features_dim: Feature dimension

    Returns:
        Dict with policy_kwargs for SB3 algorithms
    """
    extractors = {
        "standard": SARFeatureExtractor,
        "compact": CompactSARFeatureExtractor,
    }

    if extractor_type not in extractors:
        raise ValueError(f"Unknown extractor: {extractor_type}")

    return {
        "features_extractor_class": extractors[extractor_type],
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": [256, 256],  # Additional layers after feature extraction
    }
