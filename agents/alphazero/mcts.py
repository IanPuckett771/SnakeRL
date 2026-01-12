"""Monte Carlo Tree Search for AlphaZero."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from agents.alphazero.network import AlphaZeroNetwork
    from games.base import BaseGameEnv


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""

    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_drop_step: int = 30


class MCTSNode:
    """A node in the MCTS tree."""

    def __init__(
        self,
        prior: float,
        parent: MCTSNode | None = None,
    ):
        self.prior = prior
        self.parent = parent
        self.children: dict[int, MCTSNode] = {}

        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        """Mean value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """Upper Confidence Bound score for selection."""
        if self.parent is None:
            return 0.0

        # UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count)
        exploration /= 1 + self.visit_count

        return self.value + exploration

    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0


class MCTS:
    """Monte Carlo Tree Search for AlphaZero.

    Integrates with neural network for policy/value estimation.
    """

    def __init__(
        self,
        network: AlphaZeroNetwork,
        config: MCTSConfig,
        device: torch.device,
    ):
        self.network = network
        self.config = config
        self.device = device

    def search(
        self,
        env: BaseGameEnv,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Run MCTS from current state.

        Args:
            env: Game environment (will be cloned for simulations)
            add_noise: Add Dirichlet noise to root for exploration

        Returns:
            policy: Improved policy distribution over actions
            value: Estimated value of current state
        """
        root = MCTSNode(prior=1.0)

        # Get initial policy/value from network
        obs = env.get_observation()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            policy_tensor, value = self.network.predict(obs_tensor)
            root_policy = policy_tensor.squeeze(0).cpu().numpy()
            root_value = float(value.item())

        # Add Dirichlet noise to root for exploration
        valid_actions = env.get_valid_actions()
        if add_noise and len(valid_actions) > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_actions))
            for i, action in enumerate(valid_actions):
                root_policy[action] = (1 - self.config.dirichlet_epsilon) * root_policy[
                    action
                ] + self.config.dirichlet_epsilon * noise[i]

        # Expand root node
        self._expand(root, valid_actions, root_policy)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            sim_env = env.clone()
            path = [node]

            # SELECT: traverse tree to leaf
            while node.is_expanded():
                action, node = self._select(node)
                path.append(node)

                obs, reward, terminated, truncated, info = sim_env.step(action)

                if terminated or truncated:
                    break

            # EVALUATE leaf
            if terminated or truncated:
                # Terminal state - use actual reward as value
                # Normalize to [-1, 1] range
                leaf_value = float(np.clip(reward / 10.0, -1.0, 1.0))
            else:
                # Non-terminal - use network estimate
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    leaf_policy_tensor, leaf_value_tensor = self.network.predict(obs_tensor)
                    leaf_value = float(leaf_value_tensor.item())

                # EXPAND if not terminal
                valid_actions = sim_env.get_valid_actions()
                leaf_policy_np = leaf_policy_tensor.squeeze(0).cpu().numpy()
                self._expand(node, valid_actions, leaf_policy_np)

            # BACKPROPAGATE
            self._backpropagate(path, leaf_value)

        # Build policy from visit counts
        action_visits = np.zeros(env.action_space_size)
        for action, child in root.children.items():
            action_visits[action] = child.visit_count

        # Apply temperature
        final_policy: np.ndarray
        if self.config.temperature == 0 or action_visits.sum() == 0:
            # Deterministic - choose max
            final_policy = np.zeros_like(action_visits)
            if action_visits.sum() > 0:
                final_policy[np.argmax(action_visits)] = 1.0
            else:
                # Fallback to uniform
                final_policy = np.ones_like(action_visits) / len(action_visits)
        else:
            # Softmax with temperature
            visits_temp = np.power(action_visits, 1.0 / self.config.temperature)
            final_policy = visits_temp / visits_temp.sum()

        return final_policy, root_value

    def _select(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Select child with highest UCB score."""
        best_score = float("-inf")
        best_action = -1
        best_child: MCTSNode | None = None

        for action, child in node.children.items():
            score = child.ucb_score(self.config.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None, "No children to select from"
        return best_action, best_child

    def _expand(
        self,
        node: MCTSNode,
        valid_actions: list[int],
        policy: np.ndarray,
    ) -> None:
        """Add children for all valid actions."""
        for action in valid_actions:
            if action not in node.children:
                node.children[action] = MCTSNode(prior=policy[action], parent=node)

    def _backpropagate(self, path: list[MCTSNode], value: float) -> None:
        """Update all nodes in path with the leaf value."""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            # Note: For single-player games like Snake, don't flip value
