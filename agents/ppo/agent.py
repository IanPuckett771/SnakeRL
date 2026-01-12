"""PPO Agent implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.base import BaseAgent
from agents.ppo.buffer import RolloutBuffer
from agents.ppo.network import PPONetwork


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent."""

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_space_size: int,
        device: torch.device,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: float | None = None,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_buffer_size: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__(observation_shape, action_space_size, device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Network
        self.network = PPONetwork(observation_shape, action_space_size, hidden_dim).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            rollout_buffer_size,
            observation_shape,
            device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Training stats
        self.train_steps = 0

    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Select action."""
        self.network.eval()

        with torch.no_grad():
            obs_tensor = self._to_tensor(observation).unsqueeze(0)
            action_logits, value = self.network(obs_tensor)

            if deterministic:
                return int(action_logits.argmax(dim=-1).item())
            else:
                from torch.distributions import Categorical

                dist = Categorical(logits=action_logits)
                return int(dist.sample().item())

    def get_action_with_info(
        self,
        observation: np.ndarray,
        valid_actions: list[int] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Select action and return policy/value info."""
        self.network.eval()

        with torch.no_grad():
            obs_tensor = self._to_tensor(observation).unsqueeze(0)
            action_logits, value = self.network(obs_tensor)

            # Apply mask if provided
            if valid_actions is not None:
                mask = torch.full_like(action_logits, float("-inf"))
                mask[0, valid_actions] = 0
                action_logits = action_logits + mask

            probs = F.softmax(action_logits, dim=-1)
            action = int(probs.argmax(dim=-1).item())

            return action, {
                "policy": self._to_numpy(probs.squeeze(0)),
                "value": value.squeeze().item(),
            }

    def collect_rollout(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Store a transition in the rollout buffer."""
        self.buffer.add(observation, action, reward, done, value, log_prob)

    def get_value_and_log_prob(
        self,
        observation: np.ndarray,
    ) -> tuple[int, float, float]:
        """Get action, value, and log probability for rollout collection."""
        self.network.eval()

        with torch.no_grad():
            obs_tensor = self._to_tensor(observation).unsqueeze(0)
            action, log_prob, entropy, value = self.network.get_action_and_value(obs_tensor)

            return int(action.item()), float(value.item()), float(log_prob.item())

    def train_step(self, batch: dict[str, torch.Tensor] | None = None) -> dict[str, float]:
        """Perform PPO training update.

        This should be called after a full rollout is collected.

        Returns:
            Dict with policy_loss, value_loss, entropy_loss, total_loss
        """
        if not self.buffer.full and self.buffer.pos == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
                "total_loss": 0.0,
            }

        self.network.train()

        # Compute returns and advantages
        with torch.no_grad():
            # Get last observation value
            last_obs = self.buffer.observations[self.buffer.pos - 1]
            last_value = self.network.get_value(self._to_tensor(last_obs).unsqueeze(0)).item()

        self.buffer.compute_returns_and_advantages(last_value)

        # Training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get(self.batch_size):
                observations = batch["observations"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Get current policy outputs
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    observations, actions
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.clip_range_vf is not None:
                    # Clipped value loss
                    values_clipped = batch.get("old_values", values) + torch.clamp(
                        values - batch.get("old_values", values),
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                    value_loss = (
                        0.5
                        * torch.max(
                            (values - returns) ** 2,
                            (values_clipped - returns) ** 2,
                        ).mean()
                    )
                else:
                    value_loss = 0.5 * F.mse_loss(values, returns)

                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1

        # Reset buffer for next rollout
        self.buffer.reset()

        self.train_steps += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy_loss": total_entropy_loss / n_updates,
            "total_loss": (total_policy_loss + total_value_loss + total_entropy_loss) / n_updates,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint.get("train_steps", 0)
