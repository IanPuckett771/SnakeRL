"""Unified training loop with W&B integration."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from training.metrics import MetricsCollector

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from games.base import BaseGameEnv


class Trainer:
    """Unified trainer for all RL algorithms with W&B logging."""

    def __init__(
        self,
        agent: BaseAgent,
        env_fn: Callable[[], BaseGameEnv],
        config: dict[str, Any],
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        use_wandb: bool = True,
        wandb_project: str = "snakerl",
        wandb_entity: str | None = None,
    ):
        """Initialize trainer.

        Args:
            agent: RL agent to train
            env_fn: Factory function to create environments
            config: Training configuration dict
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
            use_wandb: Whether to use W&B logging
            wandb_project: W&B project name
            wandb_entity: W&B entity (team/username)
        """
        self.agent = agent
        self.env_fn = env_fn
        self.config = config

        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = MetricsCollector()

        # W&B setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = None

        if self.use_wandb:
            self.wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=config,
                name=config.get("run_name"),
                reinit=True,
            )

    def train(
        self,
        total_steps: int,
        eval_interval: int = 1000,
        checkpoint_interval: int = 5000,
        log_interval: int = 100,
    ) -> dict[str, Any]:
        """Run training loop.

        Args:
            total_steps: Total training steps
            eval_interval: Steps between evaluations
            checkpoint_interval: Steps between checkpoints
            log_interval: Steps between logging

        Returns:
            Summary of training metrics
        """
        algorithm = self.config.get("algorithm", "dqn")

        if algorithm == "dqn":
            return self._train_dqn(total_steps, eval_interval, checkpoint_interval, log_interval)
        elif algorithm == "ppo":
            return self._train_ppo(total_steps, eval_interval, checkpoint_interval, log_interval)
        elif algorithm == "alphazero":
            return self._train_alphazero(
                total_steps, eval_interval, checkpoint_interval, log_interval
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _train_dqn(
        self,
        total_steps: int,
        eval_interval: int,
        checkpoint_interval: int,
        log_interval: int,
    ) -> dict[str, Any]:
        """DQN training loop."""
        env = self.env_fn()
        obs, info = env.reset()

        episode_reward = 0.0
        episode_length = 0
        episode_count = 0
        start_time = time.time()

        for step in range(1, total_steps + 1):
            # Select action
            action = self.agent.get_action(obs)

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            self.agent.store_transition(obs, action, reward, next_obs, done)

            # Train
            train_metrics = self.agent.train_step()

            episode_reward += reward
            episode_length += 1

            if done:
                # Log episode
                episode_metrics = {
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "episode_score": info.get("score", 0),
                }
                self.metrics.add(episode_metrics, step)

                if self.use_wandb:
                    wandb.log(episode_metrics, step=step)

                # Reset
                obs, info = env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode_count += 1
            else:
                obs = next_obs

            # Logging
            if step % log_interval == 0:
                self._log_step(step, train_metrics, start_time)

            # Evaluation
            if step % eval_interval == 0:
                eval_metrics = self._evaluate()
                self.metrics.add({f"eval_{k}": v for k, v in eval_metrics.items()}, step)

                if self.use_wandb:
                    wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)

                print(f"Step {step}: eval_score={eval_metrics['mean_score']:.1f}")

            # Checkpoint
            if step % checkpoint_interval == 0:
                self._save_checkpoint(step)

        # Final save
        self._save_checkpoint(total_steps)
        self.metrics.save(str(self.log_dir / "metrics.json"))

        if self.use_wandb:
            wandb.finish()

        return self.metrics.get_summary()

    def _train_ppo(
        self,
        total_steps: int,
        eval_interval: int,
        checkpoint_interval: int,
        log_interval: int,
    ) -> dict[str, Any]:
        """PPO training loop."""
        env = self.env_fn()
        obs, info = env.reset()

        rollout_size = self.config.get("rollout_size", 2048)
        episode_reward = 0.0
        episode_length = 0
        step = 0

        while step < total_steps:
            # Collect rollout
            for _ in range(rollout_size):
                action, value, log_prob = self.agent.get_value_and_log_prob(obs)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.agent.collect_rollout(obs, action, reward, done, value, log_prob)

                episode_reward += reward
                episode_length += 1
                step += 1

                if done:
                    ep_metrics = {
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                        "episode_score": info.get("score", 0),
                    }
                    self.metrics.add(ep_metrics, step)

                    if self.use_wandb:
                        wandb.log(ep_metrics, step=step)

                    obs, info = env.reset()
                    episode_reward = 0.0
                    episode_length = 0
                else:
                    obs = next_obs

                if step >= total_steps:
                    break

            # Train on rollout
            train_metrics = self.agent.train_step()
            self.metrics.add(train_metrics, step)

            if self.use_wandb:
                wandb.log({f"train/{k}": v for k, v in train_metrics.items()}, step=step)

            # Evaluation
            if step % eval_interval == 0:
                eval_metrics = self._evaluate()
                self.metrics.add({f"eval_{k}": v for k, v in eval_metrics.items()}, step)

                if self.use_wandb:
                    wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)

                print(f"Step {step}: eval_score={eval_metrics['mean_score']:.1f}")

            # Checkpoint
            if step % checkpoint_interval == 0:
                self._save_checkpoint(step)

        self._save_checkpoint(total_steps)
        self.metrics.save(str(self.log_dir / "metrics.json"))

        if self.use_wandb:
            wandb.finish()

        return self.metrics.get_summary()

    def _train_alphazero(
        self,
        total_steps: int,
        eval_interval: int,
        checkpoint_interval: int,
        log_interval: int,
    ) -> dict[str, Any]:
        """AlphaZero training loop with self-play."""
        games_per_iter = self.config.get("games_per_iteration", 25)
        train_steps_per_iter = self.config.get("train_steps_per_iteration", 500)

        worker = self.agent.create_self_play_worker(self.env_fn)

        step = 0
        iteration = 0
        start_time = time.time()

        while step < total_steps:
            iteration += 1

            # Self-play phase
            self.agent.eval()
            games = worker.generate_batch(games_per_iter)

            game_scores = []
            game_lengths = []
            for game in games:
                self.agent.add_game(game)
                game_scores.append(sum(game.rewards))
                game_lengths.append(len(game.observations))

            # Log self-play stats
            sp_metrics = {
                "self_play/mean_score": np.mean(game_scores),
                "self_play/max_score": np.max(game_scores),
                "self_play/mean_length": np.mean(game_lengths),
            }

            if self.use_wandb:
                wandb.log(sp_metrics, step=step)

            # Training phase
            self.agent.train()
            for _ in range(train_steps_per_iter):
                train_metrics = self.agent.train_step()
                step += 1

                if step % log_interval == 0:
                    self._log_step(step, train_metrics, start_time)

                if step >= total_steps:
                    break

            # Evaluation
            if step % eval_interval == 0 or step >= total_steps:
                eval_metrics = self._evaluate()
                self.metrics.add({f"eval_{k}": v for k, v in eval_metrics.items()}, step)

                if self.use_wandb:
                    wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)

                print(f"Iter {iteration}, Step {step}: eval_score={eval_metrics['mean_score']:.1f}")

            # Checkpoint
            if step % checkpoint_interval == 0:
                self._save_checkpoint(step)

        self._save_checkpoint(total_steps)
        self.metrics.save(str(self.log_dir / "metrics.json"))

        if self.use_wandb:
            wandb.finish()

        return self.metrics.get_summary()

    def _evaluate(self, num_episodes: int = 10) -> dict[str, float]:
        """Evaluate agent performance."""
        self.agent.eval()

        scores = []
        lengths = []

        for _ in range(num_episodes):
            env = self.env_fn()
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0

            while True:
                action = self.agent.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            scores.append(info.get("score", total_reward))
            lengths.append(steps)

        return {
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
            "mean_length": float(np.mean(lengths)),
        }

    def _log_step(
        self,
        step: int,
        metrics: dict[str, float],
        start_time: float,
    ) -> None:
        """Log training progress."""
        elapsed = time.time() - start_time
        steps_per_sec = step / elapsed if elapsed > 0 else 0

        self.metrics.add(metrics, step)

        if self.use_wandb:
            wandb.log(
                {
                    **{f"train/{k}": v for k, v in metrics.items()},
                    "train/steps_per_sec": steps_per_sec,
                },
                step=step,
            )

    def _save_checkpoint(self, step: int) -> str:
        """Save model checkpoint."""
        game = self.config.get("game", "snake")
        algo = self.config.get("algorithm", "unknown")

        checkpoint_path = self.checkpoint_dir / game / algo
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        filename = f"checkpoint_{step}.pt"
        path = checkpoint_path / filename

        self.agent.save(str(path))

        # Save latest symlink
        latest = checkpoint_path / "latest.pt"
        if latest.is_symlink():
            latest.unlink()
        elif latest.exists():
            latest.unlink()
        latest.symlink_to(filename)

        if self.use_wandb:
            wandb.save(str(path))

        return str(path)
