"""DQN Agent implementation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.base import BaseAgent
from agents.dqn.network import DQNNetwork
from agents.dqn.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with experience replay and target network."""

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_space_size: int,
        device: torch.device,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        hidden_dim: int = 256,
        dueling: bool = True,
    ):
        super().__init__(observation_shape, action_space_size, device)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.policy_net = DQNNetwork(observation_shape, action_space_size, hidden_dim, dueling).to(
            device
        )
        self.target_net = DQNNetwork(observation_shape, action_space_size, hidden_dim, dueling).to(
            device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.steps = 0

    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Select action using epsilon-greedy policy."""
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_space_size)

        with torch.no_grad():
            state = self._to_tensor(observation).unsqueeze(0)
            q_values = self.policy_net(state)
            return int(q_values.argmax(dim=1).item())

    def get_action_with_info(
        self,
        observation: np.ndarray,
        valid_actions: list[int] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Select action and return Q-values."""
        with torch.no_grad():
            state = self._to_tensor(observation).unsqueeze(0)
            q_values = self.policy_net(state).squeeze(0)

            if valid_actions is not None:
                # Mask invalid actions
                mask = torch.full_like(q_values, float("-inf"))
                mask[valid_actions] = 0
                q_values = q_values + mask

            action = q_values.argmax().item()

            # Convert Q-values to pseudo-probabilities for visualization
            probs = F.softmax(q_values, dim=0)

            return action, {
                "q_values": self._to_numpy(q_values),
                "policy": self._to_numpy(probs),
                "value": q_values.max().item(),
            }

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch: dict[str, torch.Tensor] | None = None) -> dict[str, float]:
        """Perform one training step.

        Args:
            batch: Optional pre-sampled batch. If None, samples from buffer.

        Returns:
            Dict with loss and metrics
        """
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0}

        if batch is None:
            batch = self.buffer.sample(self.batch_size, self.device)

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN)
        with torch.no_grad():
            # Select actions with policy network
            next_actions = self.policy_net(next_states).argmax(dim=1)
            # Evaluate with target network
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "epsilon": self.epsilon,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.steps = checkpoint.get("steps", 0)
