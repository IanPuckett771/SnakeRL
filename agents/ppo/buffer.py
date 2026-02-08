"""Rollout buffer for PPO."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import torch

import config

# Try to import numba for JIT-compiled GAE
try:
    from numba import njit

    @njit
    def _compute_gae_numba(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        pos: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """JIT-compiled GAE computation for ~20x speedup over Python loop.

        Args:
            rewards: Reward array
            values: Value estimates
            dones: Done flags
            last_value: Value estimate of final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            pos: Number of valid entries in arrays

        Returns:
            Tuple of (advantages, returns) arrays
        """
        advantages = np.zeros(pos, dtype=np.float32)
        last_gae = 0.0

        for step in range(pos - 1, -1, -1):
            if step == pos - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = values[step + 1]
                next_done = dones[step + 1]

            delta = rewards[step] + gamma * next_value * (1 - next_done) - values[step]
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[step] = last_gae

        returns = advantages + values[:pos]
        return advantages, returns

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


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
        self.observations = np.zeros((self.buffer_size, *self.observation_shape), dtype=config.OBSERVATION_DTYPE)
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

        Uses numba JIT compilation when available for ~20x speedup.

        Args:
            last_value: Value estimate of the last state
        """
        if HAS_NUMBA:
            # Use JIT-compiled version
            adv, ret = _compute_gae_numba(
                self.rewards,
                self.values,
                self.dones,
                last_value,
                self.gamma,
                self.gae_lambda,
                self.pos,
            )
            self.advantages[: self.pos] = adv
            self.returns[: self.pos] = ret
        else:
            # Python fallback
            self._compute_gae_python(last_value)

    def _compute_gae_python(self, last_value: float) -> None:
        """Python fallback for GAE computation."""
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
                self.rewards[step]
                + self.gamma * next_value * (1 - next_done)
                - self.values[step]
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

            # Convert observations to float32 for training
            observations_f32 = self.observations[batch_indices].astype(np.float32)

            yield {
                "observations": torch.FloatTensor(observations_f32).to(self.device),
                "actions": torch.LongTensor(self.actions[batch_indices]).to(self.device),
                "old_log_probs": torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                "advantages": torch.FloatTensor(advantages[batch_indices]).to(self.device),
                "returns": torch.FloatTensor(self.returns[batch_indices]).to(self.device),
            }
