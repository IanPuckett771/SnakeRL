"""DQN Neural Network with Dueling architecture option."""

from __future__ import annotations

import torch
import torch.nn as nn

from networks.encoders import GridEncoder


class DQNNetwork(nn.Module):
    """Deep Q-Network with optional dueling architecture.

    Standard DQN: Q(s,a) directly
    Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A)
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_space_size: int,
        hidden_dim: int = 256,
        dueling: bool = True,
    ):
        super().__init__()

        self.dueling = dueling
        self.action_space_size = action_space_size

        # Encoder for grid observations
        input_channels = observation_shape[0]
        input_size = (observation_shape[1], observation_shape[2])
        self.encoder = GridEncoder(input_channels, input_size, hidden_dim)

        if dueling:
            # Value stream: V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            # Advantage stream: A(s, a)
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_space_size),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_space_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Args:
            x: Observation tensor (batch, channels, height, width)

        Returns:
            Q-values (batch, action_space_size)
        """
        features = self.encoder(x)

        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Q = V + (A - mean(A))
            q_values: torch.Tensor = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(features)

        return q_values
