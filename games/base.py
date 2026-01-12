"""Base game environment interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GameMetadata:
    """Metadata about a game environment."""

    name: str
    action_space_size: int
    action_names: list[str]
    observation_shape: tuple[int, ...]
    max_episode_steps: int = 10000
    supports_mcts: bool = True


class BaseGameEnv(ABC):
    """Gymnasium-compatible base class for all games.

    All games implement this interface to ensure compatibility
    with all training algorithms (DQN, PPO, AlphaZero).
    """

    metadata: GameMetadata

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            observation: Initial observation as numpy array
            info: Additional information dict
        """
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Integer action index

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Episode ended naturally (win/lose)
            truncated: Episode ended due to step limit
            info: Additional info (score, etc.)
        """
        pass

    @abstractmethod
    def get_valid_actions(self) -> list[int]:
        """Return list of valid action indices for current state.

        Essential for MCTS to know which actions are legal.
        """
        pass

    @abstractmethod
    def clone(self) -> BaseGameEnv:
        """Create a deep copy of the environment.

        Essential for MCTS tree search simulation.
        """
        pass

    @abstractmethod
    def render_state(self) -> dict[str, Any]:
        """Return state dict for frontend rendering.

        Should match the format expected by the WebSocket protocol.
        """
        pass

    @property
    def action_space_size(self) -> int:
        """Number of possible actions."""
        return self.metadata.action_space_size

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of observation tensor."""
        return self.metadata.observation_shape

    def get_observation(self) -> np.ndarray:
        """Get current observation without stepping."""
        # Default implementation - subclasses may override
        raise NotImplementedError("Subclass must implement get_observation()")
