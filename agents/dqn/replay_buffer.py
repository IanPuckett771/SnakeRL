"""Experience replay buffer for DQN with pre-allocated arrays."""

from __future__ import annotations

import numpy as np
import torch

import config


class ReplayBuffer:
    """Fixed-size buffer using pre-allocated numpy arrays with circular indexing.

    Much faster than deque[tuple] approach by avoiding:
    - Python object creation for each experience tuple
    - Triple array copies during sampling
    - Random.sample() overhead
    """

    def __init__(
        self,
        capacity: int = 100000,
        observation_shape: tuple[int, ...] = (3, 20, 20),
    ):
        """Initialize buffer with pre-allocated arrays.

        Args:
            capacity: Maximum number of experiences to store
            observation_shape: Shape of observation arrays
        """
        self.capacity = capacity
        self.observation_shape = observation_shape

        # Pre-allocate storage arrays
        self.states = np.zeros((capacity, *observation_shape), dtype=config.OBSERVATION_DTYPE)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *observation_shape), dtype=config.OBSERVATION_DTYPE)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Circular buffer state
        self.pos = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience to buffer using circular indexing."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Sample a batch of experiences.

        Uses numpy random indexing and torch.as_tensor for zero-copy
        tensor creation when possible. Converts observations to float32
        for neural network training.

        Args:
            batch_size: Number of experiences to sample
            device: Torch device for tensors

        Returns:
            Dict with states, actions, rewards, next_states, dones
        """
        # Sample random indices
        indices = np.random.randint(0, self.size, size=min(batch_size, self.size))

        # Use contiguous arrays for efficient tensor creation
        # Convert to float32 for training (from potentially lower precision storage)
        states = np.ascontiguousarray(self.states[indices]).astype(np.float32)
        next_states = np.ascontiguousarray(self.next_states[indices]).astype(np.float32)

        return {
            "states": torch.as_tensor(states, device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_states": torch.as_tensor(next_states, device=device),
            "dones": torch.as_tensor(self.dones[indices], device=device),
        }

    def __len__(self) -> int:
        return self.size
