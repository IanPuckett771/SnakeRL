from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
import torch

from game.state import GameState
from game.tank_state import TankState
from game.tron_state import TronState

if TYPE_CHECKING:
    from agents.alphazero import AlphaZeroAgent
    from agents.dqn import DQNAgent
    from agents.ppo import PPOAgent

# Type alias for any game state
AnyGameState = Union[GameState, TronState, TankState]

# Game-specific configurations
GAME_CONFIGS: dict[str, dict[str, Any]] = {
    "snake": {
        "obs_shape": (3, 20, 20),  # body, head, food
        "action_size": 4,
        "action_map": {0: "up", 1: "down", 2: "left", 3: "right"},
    },
    "tron": {
        "obs_shape": (4, 20, 20),  # p1_trail, p2_trail, p1_head, p2_head
        "action_size": 4,
        "action_map": {0: 0, 1: 1, 2: 2, 3: 3},  # up, down, left, right (numeric)
    },
    "tank": {
        "obs_shape": (5, 20, 20),  # walls, player, enemies, bullets, collectibles
        "action_size": 5,
        "action_map": {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
        },  # forward, backward, turn_left, turn_right, shoot
    },
}


class AgentInterface:
    """Interface for RL agent to play multiple games.

    Loads trained DQN/PPO/AlphaZero checkpoints and uses them for inference.
    Supports Snake, Tron, and Tank Battle games.
    """

    ACTIONS = ["up", "down", "left", "right"]
    ACTION_MAP = {0: "up", 1: "down", 2: "left", 3: "right"}
    CHECKPOINTS_DIR = "checkpoints"

    def __init__(self, game: str = "snake"):
        """Initialize the agent interface.

        Args:
            game: Game type ("snake", "tron", or "tank")
        """
        if game not in GAME_CONFIGS:
            raise ValueError(f"Unknown game: {game}. Available: {list(GAME_CONFIGS.keys())}")

        self.game = game
        self.config: dict[str, Any] = GAME_CONFIGS[game]
        self.checkpoint_path: str | None = None
        self.agent: DQNAgent | PPOAgent | AlphaZeroAgent | None = None
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_checkpoint(self, path: str) -> bool:
        """Load a model checkpoint.

        Args:
            path: Path to the checkpoint file

        Returns:
            True if checkpoint was loaded successfully, False otherwise
        """
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return False

        try:
            self.checkpoint_path = path

            # Load checkpoint to inspect its structure
            checkpoint = torch.load(path, map_location=self.device)

            # Determine agent type from checkpoint keys
            if "policy_net" in checkpoint:
                # DQN checkpoint
                self._load_dqn(checkpoint)
            elif "network" in checkpoint and "mcts_config" in checkpoint:
                # AlphaZero checkpoint
                self._load_alphazero(checkpoint)
            elif "network" in checkpoint:
                # PPO checkpoint
                self._load_ppo(checkpoint)
            else:
                print(f"Unknown checkpoint format: {list(checkpoint.keys())}")
                return False

            print(f"Loaded checkpoint: {path}")
            return True

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def _load_dqn(self, checkpoint: dict[str, Any]) -> None:
        """Load a DQN agent from checkpoint."""
        from agents.dqn import DQNAgent

        # Use game-specific observation shape and action size
        obs_shape: tuple[int, ...] = cast(tuple[int, ...], self.config["obs_shape"])
        action_size: int = cast(int, self.config["action_size"])

        agent = DQNAgent(
            observation_shape=obs_shape,
            action_space_size=action_size,
            device=self.device,
        )
        agent.policy_net.load_state_dict(checkpoint["policy_net"])
        agent.policy_net.eval()
        agent.epsilon = 0.0  # No exploration during inference
        self.agent = agent

    def _load_ppo(self, checkpoint: dict[str, Any]) -> None:
        """Load a PPO agent from checkpoint."""
        from agents.ppo import PPOAgent

        obs_shape: tuple[int, ...] = cast(tuple[int, ...], self.config["obs_shape"])
        action_size: int = cast(int, self.config["action_size"])

        agent = PPOAgent(
            observation_shape=obs_shape,
            action_space_size=action_size,
            device=self.device,
        )
        agent.network.load_state_dict(checkpoint["network"])
        agent.network.eval()
        self.agent = agent

    def _load_alphazero(self, checkpoint: dict[str, Any]) -> None:
        """Load an AlphaZero agent from checkpoint."""
        from agents.alphazero import AlphaZeroAgent

        obs_shape: tuple[int, ...] = cast(tuple[int, ...], self.config["obs_shape"])
        action_size: int = cast(int, self.config["action_size"])

        agent = AlphaZeroAgent(
            observation_shape=obs_shape,
            action_space_size=action_size,
            device=self.device,
        )
        agent.network.load_state_dict(checkpoint["network"])
        agent.network.eval()
        self.agent = agent

    def _state_to_observation(self, state: AnyGameState) -> np.ndarray:
        """Convert game state to neural network observation.

        Handles different game types:
        - Snake: 3 channels (body, head, food)
        - Tron: 4 channels (p1_trail, p2_trail, p1_head, p2_head)
        - Tank: 5 channels (walls, player, enemies, bullets, collectibles)

        Args:
            state: Game state (GameState, TronState, or TankState)

        Returns:
            Numpy array observation suitable for neural network
        """
        if self.game == "snake":
            return self._snake_state_to_observation(cast(GameState, state))
        elif self.game == "tron":
            return self._tron_state_to_observation(cast(TronState, state))
        elif self.game == "tank":
            return self._tank_state_to_observation(cast(TankState, state))
        else:
            raise ValueError(f"Unknown game type: {self.game}")

    def _snake_state_to_observation(self, state: GameState) -> np.ndarray:
        """Convert Snake GameState to neural network observation.

        Returns 3-channel grid:
        - Channel 0: Snake body
        - Channel 1: Snake head
        - Channel 2: Food
        """
        obs = np.zeros((3, state.height, state.width), dtype=np.float32)

        # Snake body (excluding head)
        for x, y in state.snake[1:]:
            if 0 <= x < state.width and 0 <= y < state.height:
                obs[0, y, x] = 1.0

        # Snake head
        if state.snake:
            hx, hy = state.snake[0]
            if 0 <= hx < state.width and 0 <= hy < state.height:
                obs[1, hy, hx] = 1.0

        # Food
        fx, fy = state.food
        if 0 <= fx < state.width and 0 <= fy < state.height:
            obs[2, fy, fx] = 1.0

        return obs

    def _tron_state_to_observation(self, state: TronState, player: int = 1) -> np.ndarray:
        """Convert TronState to neural network observation.

        Returns 4-channel grid from the perspective of the specified player:
        - Channel 0: Own trail
        - Channel 1: Opponent trail
        - Channel 2: Own head
        - Channel 3: Opponent head

        Args:
            state: TronState
            player: Which player's perspective (1 or 2)
        """
        obs = np.zeros((4, state.height, state.width), dtype=np.float32)

        # Get player references based on perspective
        if player == 1:
            own_player = state.player1
            opp_player = state.player2
        else:
            own_player = state.player2
            opp_player = state.player1

        # Own trail (channel 0)
        for x, y in own_player.trail:
            if 0 <= x < state.width and 0 <= y < state.height:
                obs[0, y, x] = 1.0

        # Opponent trail (channel 1)
        for x, y in opp_player.trail:
            if 0 <= x < state.width and 0 <= y < state.height:
                obs[1, y, x] = 1.0

        # Own head (channel 2)
        if own_player.alive:
            if 0 <= own_player.x < state.width and 0 <= own_player.y < state.height:
                obs[2, own_player.y, own_player.x] = 1.0

        # Opponent head (channel 3)
        if opp_player.alive:
            if 0 <= opp_player.x < state.width and 0 <= opp_player.y < state.height:
                obs[3, opp_player.y, opp_player.x] = 1.0

        return obs

    def _tank_state_to_observation(self, state: TankState) -> np.ndarray:
        """Convert TankState to neural network observation.

        Converts pixel coordinates to grid coordinates for observation.
        Uses 20 pixels per grid cell (matching TankEnv.CELL_SIZE).

        Returns 5-channel grid:
        - Channel 0: Walls
        - Channel 1: Player tank
        - Channel 2: Enemy tanks
        - Channel 3: Bullets
        - Channel 4: Collectibles
        """
        cell_size = 20  # Pixels per grid cell
        grid_width = int(state.width / cell_size)
        grid_height = int(state.height / cell_size)

        obs = np.zeros((5, grid_height, grid_width), dtype=np.float32)

        def pixel_to_grid(px: float, py: float) -> tuple[int, int]:
            gx = int(px / cell_size)
            gy = int(py / cell_size)
            return (
                max(0, min(grid_width - 1, gx)),
                max(0, min(grid_height - 1, gy)),
            )

        # Walls (Wall objects with x, y, width, height in pixels)
        for wall in state.walls:
            x1 = int(wall.x / cell_size)
            y1 = int(wall.y / cell_size)
            x2 = int((wall.x + wall.width) / cell_size)
            y2 = int((wall.y + wall.height) / cell_size)
            for gx in range(max(0, x1), min(grid_width, x2 + 1)):
                for gy in range(max(0, y1), min(grid_height, y2 + 1)):
                    obs[0, gy, gx] = 1.0

        # Player tank
        if state.player.alive:
            gx, gy = pixel_to_grid(state.player.x, state.player.y)
            obs[1, gy, gx] = 1.0

        # Enemy tanks
        for enemy in state.enemies:
            if enemy.alive:
                gx, gy = pixel_to_grid(enemy.x, enemy.y)
                obs[2, gy, gx] = 1.0

        # Bullets
        for bullet in state.bullets:
            gx, gy = pixel_to_grid(bullet.x, bullet.y)
            obs[3, gy, gx] = 1.0

        # Collectibles
        for coll in state.collectibles:
            gx, gy = pixel_to_grid(coll.x, coll.y)
            obs[4, gy, gx] = 1.0

        return obs

    def get_action(self, state: AnyGameState, player: int = 1) -> str | int:
        """Get the next action for the given state.

        Args:
            state: Current game state (GameState, TronState, or TankState)
            player: For Tron, which player to get action for (1 or 2)

        Returns:
            Action - string for Snake, int for Tron/Tank
        """
        if self.agent is None:
            # No model loaded, return random action
            import random

            if self.game == "snake":
                return random.choice(self.ACTIONS)
            else:
                action_size = cast(int, self.config["action_size"])
                return random.randint(0, action_size - 1)

        # Convert state to observation (with player perspective for Tron)
        if self.game == "tron":
            obs = self._tron_state_to_observation(cast(TronState, state), player=player)
        else:
            obs = self._state_to_observation(state)

        # Get action from agent
        action_idx = self.agent.get_action(obs, deterministic=True)

        # Map action based on game type
        action_map = cast(dict[int, str | int], self.config["action_map"])
        return action_map[action_idx]

    @classmethod
    def list_checkpoints(cls, checkpoints_dir: str | None = None) -> list[str]:
        """Scan checkpoints directory and return list of checkpoint files.

        Scans nested structure: checkpoints/{game}/{algorithm}/*.pt

        Args:
            checkpoints_dir: Optional custom checkpoints directory path

        Returns:
            List of checkpoint paths relative to checkpoints dir
        """
        directory = Path(checkpoints_dir or cls.CHECKPOINTS_DIR)
        checkpoints = []

        if directory.exists():
            # Scan for .pt and .pth files recursively
            for pt_file in directory.rglob("*.pt"):
                # Skip symlinks (like latest.pt)
                if not pt_file.is_symlink():
                    rel_path = pt_file.relative_to(directory)
                    checkpoints.append(str(rel_path))

            for pth_file in directory.rglob("*.pth"):
                if not pth_file.is_symlink():
                    rel_path = pth_file.relative_to(directory)
                    checkpoints.append(str(rel_path))

        return sorted(checkpoints)
