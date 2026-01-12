"""Self-play data generation for AlphaZero."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from agents.alphazero.mcts import MCTS, MCTSConfig
from agents.alphazero.network import AlphaZeroNetwork
from agents.alphazero.replay_buffer import GameRecord

if TYPE_CHECKING:
    from games.base import BaseGameEnv


class SelfPlayWorker:
    """Generate self-play games for AlphaZero training."""

    def __init__(
        self,
        env_fn: Callable[[], BaseGameEnv],
        network: AlphaZeroNetwork,
        mcts_config: MCTSConfig,
        device: torch.device,
    ):
        self.env_fn = env_fn
        self.network = network
        self.mcts = MCTS(network, mcts_config, device)
        self.device = device

    def play_game(
        self,
        temperature_drop_step: int = 30,
    ) -> GameRecord:
        """Play one complete game with MCTS.

        Args:
            temperature_drop_step: Step at which to drop temperature to near-zero

        Returns:
            GameRecord with full game history
        """
        env = self.env_fn()
        obs, info = env.reset()

        observations = []
        policies = []
        rewards = []
        step = 0

        self.network.eval()

        while True:
            # Adjust temperature based on step
            if step < temperature_drop_step:
                self.mcts.config.temperature = 1.0
            else:
                self.mcts.config.temperature = 0.1

            # Run MCTS to get improved policy
            add_noise = step < temperature_drop_step
            policy, value = self.mcts.search(env, add_noise=add_noise)

            # Store for training
            observations.append(obs.copy())
            policies.append(policy)

            # Sample action from policy
            action = np.random.choice(len(policy), p=policy)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            step += 1

            if terminated or truncated:
                break

        # Compute final outcome (normalize final score)
        final_score = info.get("score", sum(rewards))
        # Normalize to [-1, 1] based on typical max score
        outcome = np.clip(final_score / 50.0, -1.0, 1.0)

        return GameRecord(
            observations=observations,
            policies=policies,
            rewards=rewards,
            outcome=outcome,
        )

    def generate_batch(self, num_games: int) -> list[GameRecord]:
        """Generate multiple self-play games.

        Args:
            num_games: Number of games to generate

        Returns:
            List of GameRecords
        """
        records = []
        for _ in range(num_games):
            record = self.play_game()
            records.append(record)
        return records
