#!/usr/bin/env python3
"""Training script for SnakeRL agents."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")

import yaml  # type: ignore[import-untyped]  # noqa: E402

from networks.utils import get_device, get_device_info  # noqa: E402
from training.trainer import Trainer  # noqa: E402


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def create_agent(config: dict, env):
    """Create agent based on configuration."""
    algorithm = config.get("algorithm", "dqn")
    device = get_device()

    obs_shape = env.observation_shape
    action_size = env.action_space_size

    if algorithm == "dqn":
        from agents.dqn import DQNAgent

        return DQNAgent(
            observation_shape=obs_shape,
            action_space_size=action_size,
            device=device,
            **config.get("agent", {}),
        )
    elif algorithm == "ppo":
        from agents.ppo import PPOAgent

        return PPOAgent(
            observation_shape=obs_shape,
            action_space_size=action_size,
            device=device,
            **config.get("agent", {}),
        )
    elif algorithm == "alphazero":
        from agents.alphazero import AlphaZeroAgent

        return AlphaZeroAgent(
            observation_shape=obs_shape,
            action_space_size=action_size,
            device=device,
            **config.get("agent", {}),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_env_fn(config: dict):
    """Create environment factory function."""
    game = config.get("game", "snake")
    env_config = config.get("environment", {})

    # Import games to register them (side effect: registers with GameRegistry)
    import games.snake.env  # noqa: F401

    try:
        import games.snake.enhanced  # noqa: F401
    except ImportError:
        pass

    from games.registry import GameRegistry

    def env_fn():
        return GameRegistry.create(game, env_config)

    return env_fn


def main():
    parser = argparse.ArgumentParser(description="Train RL agents on Snake games")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="snake",
        choices=["snake", "enhanced_snake"],
        help="Game to train on",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dqn",
        choices=["dqn", "ppo", "alphazero"],
        help="Algorithm to use",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Total training steps",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "snakerl"),
        help="W&B project name (default: $WANDB_PROJECT or 'snakerl')",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity/team (default: $WANDB_ENTITY)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name for this training run",
    )

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "game": args.game,
            "algorithm": args.algorithm,
            "environment": {},
            "agent": {},
        }

    # Override with command line args
    if args.game:
        config["game"] = args.game
    if args.algorithm:
        config["algorithm"] = args.algorithm
    if args.run_name:
        config["run_name"] = args.run_name
    else:
        config["run_name"] = f"{config['game']}_{config['algorithm']}"

    # Print device info
    print("Device info:", get_device_info())
    print(f"Using device: {get_device()}")
    print(f"Training {config['algorithm']} on {config['game']} for {args.steps} steps")

    # Create environment and agent
    env_fn = create_env_fn(config)
    env = env_fn()  # Create one to get shapes
    agent = create_agent(config, env)

    # Create trainer
    trainer = Trainer(
        agent=agent,
        env_fn=env_fn,
        config=config,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    # Train
    results = trainer.train(
        total_steps=args.steps,
        eval_interval=1000,
        checkpoint_interval=5000,
        log_interval=100,
    )

    print("\nTraining complete!")
    print("Results:", results)


if __name__ == "__main__":
    main()
