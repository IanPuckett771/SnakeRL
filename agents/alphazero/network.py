"""AlphaZero Policy-Value Network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.encoders import GridEncoder


class AlphaZeroNetwork(nn.Module):
    """Neural network for AlphaZero.

    Single encoder shared between policy and value heads.

    Architecture:
        Input -> Encoder -> [Policy Head, Value Head]

    Policy Head: Outputs probability distribution over actions
    Value Head: Outputs scalar value estimate in [-1, 1]
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_space_size: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.observation_shape = observation_shape
        self.action_space_size = action_space_size

        # Encoder for grid observations
        input_channels = observation_shape[0]
        input_size = (observation_shape[1], observation_shape[2])
        self.encoder = GridEncoder(input_channels, input_size, hidden_dim)

        # Policy head: outputs action logits
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size),
        )

        # Value head: outputs scalar value in [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Observation tensor (batch, *observation_shape)

        Returns:
            policy_logits: Raw logits for policy (batch, action_space_size)
            value: Value estimate (batch, 1)
        """
        features = self.encoder(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value

    def predict(
        self,
        x: torch.Tensor,
        valid_actions_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy and value with optional action masking.

        Args:
            x: Observation tensor
            valid_actions_mask: Boolean mask of valid actions (batch, action_space_size)

        Returns:
            policy: Probability distribution (after softmax + masking)
            value: Value estimate
        """
        policy_logits, value = self.forward(x)

        if valid_actions_mask is not None:
            # Mask invalid actions with large negative value
            policy_logits = policy_logits.masked_fill(~valid_actions_mask, float("-inf"))

        policy = F.softmax(policy_logits, dim=-1)
        return policy, value.squeeze(-1)
