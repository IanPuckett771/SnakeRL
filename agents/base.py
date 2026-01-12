"""Base agent interface for all RL algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from games.base import BaseGameEnv


class BaseAgent(ABC):
    """Base class for all RL agents.

    Provides unified interface for training and inference
    across DQN, PPO, and AlphaZero algorithms.
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_space_size: int,
        device: torch.device,
    ):
        """Initialize agent.

        Args:
            observation_shape: Shape of observations
            action_space_size: Number of possible actions
            device: Torch device for computation
        """
        self.observation_shape = observation_shape
        self.action_space_size = action_space_size
        self.device = device
        self._training = True

    @abstractmethod
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Select action for given observation.

        Args:
            observation: Current state observation
            deterministic: If True, use greedy/deterministic selection

        Returns:
            Action index
        """
        pass

    @abstractmethod
    def get_action_with_info(
        self,
        observation: np.ndarray,
        valid_actions: list[int] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Select action and return additional info.

        Args:
            observation: Current state observation
            valid_actions: Optional list of valid action indices

        Returns:
            action: Selected action index
            info: Dict with policy probs, value estimate, etc.
        """
        pass

    @abstractmethod
    def train_step(self, batch: dict[str, torch.Tensor] | None = None) -> dict[str, float]:
        """Perform one training step.

        Args:
            batch: Optional training batch (format depends on algorithm).
                   If None, the agent will sample from its own buffer.

        Returns:
            Dict of loss values and metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        pass

    def train(self) -> None:
        """Set agent to training mode."""
        self._training = True

    def eval(self) -> None:
        """Set agent to evaluation mode."""
        self._training = False

    @property
    def training_mode(self) -> bool:
        """Whether agent is in training mode."""
        return self._training

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor on device."""
        return torch.FloatTensor(x).to(self.device)

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        result: np.ndarray = x.detach().cpu().numpy()
        return result

    # Algorithm-specific methods with default implementations
    # These are overridden by specific agents as needed

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer (DQN)."""
        raise NotImplementedError("store_transition not implemented for this agent")

    def get_value_and_log_prob(
        self,
        observation: np.ndarray,
    ) -> tuple[int, float, float]:
        """Get action, value, and log probability (PPO)."""
        raise NotImplementedError("get_value_and_log_prob not implemented for this agent")

    def collect_rollout(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Store a transition in the rollout buffer (PPO)."""
        raise NotImplementedError("collect_rollout not implemented for this agent")

    def create_self_play_worker(
        self,
        env_fn: Callable[[], BaseGameEnv],
    ) -> Any:
        """Create a self-play worker for data generation (AlphaZero)."""
        raise NotImplementedError("create_self_play_worker not implemented for this agent")

    def add_game(self, game: Any) -> None:
        """Add a completed game to the replay buffer (AlphaZero)."""
        raise NotImplementedError("add_game not implemented for this agent")
