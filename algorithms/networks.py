"""Shared lightweight CNN encoder and algorithm-specific network heads.

The CNN encoder uses narrow channels (16→32) with a 1×1 channel reduction
before flattening, keeping total parameter count (~105K for DQN) on par with
the flat MLP while retaining the full 7-channel grid input.
"""
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Lightweight CNN encoder for grid-based state representation.

    Takes (batch, 7, H, W) grid input and produces a 256-dim feature vector.
    Uses narrow channels + 1×1 reduction to minimize parameters while
    preserving spatial information from the full game board.

    Architecture (~6.2K params):
        Conv2d(7→16, 3×3, stride=2)  — spatial features + downsample
        Conv2d(16→32, 3×3)           — deeper spatial features
        Conv2d(32→16, 1×1)           — channel reduction before flatten
        AdaptiveAvgPool2d(4)          — 16×4×4 = 256-dim output
    """

    def __init__(self, in_channels=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1),  # 1×1 channel reduction
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.feature_dim = 16 * 4 * 4  # 256

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class DQNCNNNetwork(nn.Module):
    """DQN with lightweight CNN encoder: encoder(256) → 256 → 128 → 4 (Q-values).

    Total ~105K params — same as the flat MLP, but sees the full board.
    """

    def __init__(self, num_channels=7, action_size=4):
        super().__init__()
        self.encoder = CNNEncoder(num_channels)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class A2CCNNNetwork(nn.Module):
    """A2C actor-critic with lightweight CNN encoder."""

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
    """PPO actor-critic with lightweight CNN encoder."""

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
