"""Experience replay buffer for DQN."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity: int = 100000):
        """Initialize buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample
            device: Torch device for tensors

        Returns:
            Dict with states, actions, rewards, next_states, dones
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            "states": torch.FloatTensor(np.array(states)).to(device),
            "actions": torch.LongTensor(actions).to(device),
            "rewards": torch.FloatTensor(rewards).to(device),
            "next_states": torch.FloatTensor(np.array(next_states)).to(device),
            "dones": torch.FloatTensor(dones).to(device),
        }

    def __len__(self) -> int:
        return len(self.buffer)
