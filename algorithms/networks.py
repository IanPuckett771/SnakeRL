"""Shared CNN encoder and algorithm-specific network heads."""
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Shared CNN encoder for grid-based state representation.

    Takes (batch, 7, H, W) grid input and produces a 1024-dim feature vector.
    AdaptiveAvgPool2d handles variable board sizes (10-50) without resizing.
    """

    def __init__(self, in_channels=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.feature_dim = 64 * 4 * 4  # 1024

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class DQNCNNNetwork(nn.Module):
    """DQN with CNN encoder: encoder → 1024 → 256 → 4 (Q-values)."""

    def __init__(self, num_channels=7, action_size=4):
        super().__init__()
        self.encoder = CNNEncoder(num_channels)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class A2CCNNNetwork(nn.Module):
    """A2C actor-critic with CNN encoder: encoder → 1024 → 256 shared → actor/critic."""

    def __init__(self, num_channels=7, action_size=4):
        super().__init__()
        self.encoder = CNNEncoder(num_channels)
        self.shared = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 256),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        features = self.encoder(x)
        shared = self.shared(features)
        return self.actor(shared), self.critic(shared)


class PPOCNNNetwork(nn.Module):
    """PPO actor-critic with CNN encoder: encoder → 1024 → 256 shared → actor/critic."""

    def __init__(self, num_channels=7, action_size=4):
        super().__init__()
        self.encoder = CNNEncoder(num_channels)
        self.shared = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 256),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        features = self.encoder(x)
        shared = self.shared(features)
        return self.actor(shared), self.critic(shared)
