"""Neural network encoders for different observation types."""

from __future__ import annotations

import torch
import torch.nn as nn


class GridEncoder(nn.Module):
    """Encode 2D grid observations (Snake, Pong, Breakout, etc.)

    Input: (batch, channels, height, width)
    Output: (batch, feature_dim)
    """

    def __init__(
        self,
        input_channels: int,
        input_size: tuple[int, int],
        feature_dim: int = 256,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_size = input_size
        self.feature_dim = feature_dim

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate flattened size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            conv_out = self.conv_layers(dummy)
            self._flat_size = conv_out.view(1, -1).size(1)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self._flat_size, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode grid observation to feature vector.

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Feature tensor (batch, feature_dim)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        result: torch.Tensor = self.fc(x)
        return result


class VectorEncoder(nn.Module):
    """Encode flat vector observations.

    Input: (batch, input_dim)
    Output: (batch, feature_dim)
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode vector observation to feature vector."""
        result: torch.Tensor = self.layers(x)
        return result
