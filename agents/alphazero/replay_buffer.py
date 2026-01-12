"""Game replay buffer for AlphaZero training."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GameRecord:
    """Record of a single self-play game."""

    observations: list[np.ndarray]  # State at each step
    policies: list[np.ndarray]  # MCTS policy at each step
    rewards: list[float]  # Reward at each step
    outcome: float  # Final normalized outcome


class GameReplayBuffer:
    """Buffer for storing complete games for AlphaZero training."""

    def __init__(self, max_games: int = 1000):
        """Initialize buffer.

        Args:
            max_games: Maximum number of games to store
        """
        self.games: deque[GameRecord] = deque(maxlen=max_games)
        self._positions: list[tuple] = []  # (game_idx, step_idx) for sampling

    def add_game(self, game: GameRecord) -> None:
        """Add a game to the buffer."""
        game_idx = len(self.games)
        self.games.append(game)

        # Add all positions from this game
        for step_idx in range(len(game.observations)):
            self._positions.append((game_idx, step_idx))

        # Clean up old positions if buffer wrapped
        if len(self.games) == self.games.maxlen:
            self._rebuild_positions()

    def _rebuild_positions(self) -> None:
        """Rebuild position index after buffer wraps."""
        self._positions = []
        for game_idx, game in enumerate(self.games):
            for step_idx in range(len(game.observations)):
                self._positions.append((game_idx, step_idx))

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Sample a batch of training examples.

        Args:
            batch_size: Number of positions to sample
            device: Torch device for tensors

        Returns:
            Dict with observations, target_policies, target_values
        """
        if len(self._positions) < batch_size:
            indices = list(range(len(self._positions)))
        else:
            indices = random.sample(range(len(self._positions)), batch_size)

        observations = []
        target_policies = []
        target_values = []

        for idx in indices:
            game_idx, step_idx = self._positions[idx]
            game = self.games[game_idx]

            observations.append(game.observations[step_idx])
            target_policies.append(game.policies[step_idx])
            target_values.append(game.outcome)

        return {
            "observations": torch.FloatTensor(np.array(observations)).to(device),
            "target_policies": torch.FloatTensor(np.array(target_policies)).to(device),
            "target_values": torch.FloatTensor(target_values).to(device),
        }

    def __len__(self) -> int:
        return len(self._positions)
