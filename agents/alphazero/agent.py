"""AlphaZero Agent implementation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.alphazero.mcts import MCTS, MCTSConfig
from agents.alphazero.network import AlphaZeroNetwork
from agents.alphazero.replay_buffer import GameRecord, GameReplayBuffer
from agents.alphazero.self_play import SelfPlayWorker
from agents.base import BaseAgent

if TYPE_CHECKING:
    from games.base import BaseGameEnv


class AlphaZeroAgent(BaseAgent):
    """AlphaZero agent with MCTS and self-play training."""

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_space_size: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_dim: int = 256,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        buffer_size: int = 10000,
    ):
        super().__init__(observation_shape, action_space_size, device)

        # Network
        self.network = AlphaZeroNetwork(observation_shape, action_space_size, hidden_dim).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # MCTS config
        self.mcts_config = MCTSConfig(
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
        )

        # MCTS for inference
        self.mcts = MCTS(self.network, self.mcts_config, device)

        # Replay buffer
        self.buffer = GameReplayBuffer(max_games=buffer_size // 50)  # Approx positions

        # Training stats
        self.train_steps = 0

    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Select action using MCTS or direct network inference."""
        self.network.eval()

        if deterministic or not self._training:
            # Use network directly for fast inference
            with torch.no_grad():
                obs_tensor = self._to_tensor(observation).unsqueeze(0)
                policy, value = self.network.predict(obs_tensor)
                return int(policy.argmax(dim=-1).item())
        else:
            # This shouldn't typically be called during training
            # (self-play handles action selection)
            with torch.no_grad():
                obs_tensor = self._to_tensor(observation).unsqueeze(0)
                policy, value = self.network.predict(obs_tensor)
                probs = policy.squeeze(0).cpu().numpy()
                return int(np.random.choice(len(probs), p=probs))

    def get_action_with_info(
        self,
        observation: np.ndarray,
        valid_actions: list[int] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Select action and return policy/value info."""
        self.network.eval()

        with torch.no_grad():
            obs_tensor = self._to_tensor(observation).unsqueeze(0)

            # Create valid action mask if provided
            mask = None
            if valid_actions is not None:
                mask = torch.zeros(1, self.action_space_size, dtype=torch.bool, device=self.device)
                mask[0, valid_actions] = True

            policy, value = self.network.predict(obs_tensor, mask)
            policy_np = policy.squeeze(0).cpu().numpy()

            action = policy_np.argmax()

            return int(action), {
                "policy": policy_np,
                "value": value.item(),
            }

    def create_self_play_worker(
        self,
        env_fn: Callable[[], BaseGameEnv],
    ) -> SelfPlayWorker:
        """Create a self-play worker for data generation."""
        return SelfPlayWorker(env_fn, self.network, self.mcts_config, self.device)

    def add_game(self, game: GameRecord) -> None:
        """Add a completed game to the replay buffer."""
        self.buffer.add_game(game)

    def train_step(self, batch: dict[str, torch.Tensor] | None = None) -> dict[str, float]:
        """Perform one training step.

        Args:
            batch: Optional pre-sampled batch. If None, samples from buffer.

        Returns:
            Dict with policy_loss, value_loss, total_loss
        """
        if len(self.buffer) < 64:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        if batch is None:
            batch = self.buffer.sample(256, self.device)

        self.network.train()

        observations = batch["observations"]
        target_policies = batch["target_policies"]
        target_values = batch["target_values"]

        # Forward pass
        policy_logits, values = self.network(observations)
        values = values.squeeze(-1)

        # Policy loss: cross-entropy with MCTS policy
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(target_policies * log_probs).sum(dim=-1).mean()

        # Value loss: MSE with game outcome
        value_loss = F.mse_loss(values, target_values)

        # Total loss
        total_loss = policy_loss + value_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
                "mcts_config": {
                    "num_simulations": self.mcts_config.num_simulations,
                    "c_puct": self.mcts_config.c_puct,
                    "dirichlet_alpha": self.mcts_config.dirichlet_alpha,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint.get("train_steps", 0)
