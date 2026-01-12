"""PPO Actor-Critic Network."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from networks.encoders import GridEncoder


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO.

    Shared encoder with separate policy (actor) and value (critic) heads.
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_space_size: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.action_space_size = action_space_size

        # Shared encoder
        input_channels = observation_shape[0]
        input_size = (observation_shape[1], observation_shape[2])
        self.encoder = GridEncoder(input_channels, input_size, hidden_dim)

        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size),
        )

        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Observation tensor

        Returns:
            action_logits: Logits for action distribution
            value: State value estimate
        """
        features = self.encoder(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value.

        Used during rollout collection and training.

        Args:
            x: Observation tensor
            action: Optional action to evaluate (for training)

        Returns:
            action: Sampled or provided action
            log_prob: Log probability of action
            entropy: Distribution entropy
            value: State value estimate
        """
        action_logits, value = self.forward(x)
        dist = Categorical(logits=action_logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        features = self.encoder(x)
        result: torch.Tensor = self.critic(features).squeeze(-1)
        return result
