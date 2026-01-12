"""Rollout buffer for PPO."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import torch


class RolloutBuffer:
    """Buffer for storing rollout trajectories for PPO training."""

    def __init__(
        self,
        buffer_size: int,
        observation_shape: tuple,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Initialize buffer.

        Args:
            buffer_size: Number of steps to store
            observation_shape: Shape of observations
            device: Torch device
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.reset()

    def reset(self) -> None:
        """Reset the buffer."""
        self.observations = np.zeros((self.buffer_size, *self.observation_shape), dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)

        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Add a step to the buffer."""
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float) -> None:
        """Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate of the last state
        """
        last_gae = 0.0

        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[step + 1]
                next_done = self.dones[step + 1]

            # TD error
            delta = (
                self.rewards[step] + self.gamma * next_value * (1 - next_done) - self.values[step]
            )

            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * last_gae
            self.advantages[step] = last_gae

        # Returns = advantages + values
        self.returns[: self.pos] = self.advantages[: self.pos] + self.values[: self.pos]

    def get(
        self,
        batch_size: int | None = None,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Generate batches for training.

        Args:
            batch_size: Size of mini-batches. If None, use full buffer.

        Yields:
            Dict with observations, actions, old_log_probs, advantages, returns
        """
        size = self.pos if not self.full else self.buffer_size

        if batch_size is None:
            batch_size = size

        # Normalize advantages
        advantages = self.advantages[:size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.random.permutation(size)

        for start in range(0, size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield {
                "observations": torch.FloatTensor(self.observations[batch_indices]).to(self.device),
                "actions": torch.LongTensor(self.actions[batch_indices]).to(self.device),
                "old_log_probs": torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                "advantages": torch.FloatTensor(advantages[batch_indices]).to(self.device),
                "returns": torch.FloatTensor(self.returns[batch_indices]).to(self.device),
            }
