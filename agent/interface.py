import os
import random
import numpy as np
from pathlib import Path
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from game.state import GameState


if TORCH_AVAILABLE:
    # Legacy FC network for old checkpoints (matches DQNNetwork in algorithms/dqn.py)
    class SimpleDQN(nn.Module):
        """Simple Deep Q-Network for Snake (legacy flat vector)."""

        def __init__(self, state_size=24, action_size=4, hidden_size=256):
            super(SimpleDQN, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 128)
            self.fc4 = nn.Linear(128, action_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            return self.fc4(x)
else:
    class SimpleDQN:
        pass


class AgentInterface:
    """Interface for RL agent to play Snake."""

    ACTIONS = ["up", "down", "left", "right"]
    CHECKPOINTS_DIR = "checkpoints"

    def __init__(self):
        """Initialize the agent interface."""
        self.checkpoint_path: Optional[str] = None
        self.model = None
        self.model_type = None  # 'cnn' or 'legacy'
        self.device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None

    def load_checkpoint(self, path: str) -> bool:
        """Load a model checkpoint (CNN or legacy FC).

        Args:
            path: Path to the checkpoint file

        Returns:
            True if checkpoint was found, False otherwise
        """
        if not TORCH_AVAILABLE:
            return False

        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)

                if isinstance(checkpoint, dict) and checkpoint.get('model_type') == 'cnn':
                    # CNN checkpoint — detect architecture from state_dict keys
                    from algorithms.networks import DQNCNNNetwork, A2CCNNNetwork, PPOCNNNetwork
                    state_dict = checkpoint['model_state_dict']
                    has_head = any(k.startswith('head.') for k in state_dict)
                    if has_head:
                        # DQN: encoder + head
                        self.model = DQNCNNNetwork(num_channels=7, action_size=4).to(self.device)
                    else:
                        # A2C/PPO: encoder + shared + actor + critic (same arch)
                        self.model = A2CCNNNetwork(num_channels=7, action_size=4).to(self.device)
                    self.model.load_state_dict(state_dict)
                    self.model_type = 'cnn'
                else:
                    # Legacy flat checkpoint
                    self.model = SimpleDQN(state_size=24, action_size=4).to(self.device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model = checkpoint
                    self.model_type = 'legacy'

                self.model.eval()
                self.checkpoint_path = path
                return True
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return False
        return False

    def _encode_state_legacy(self, state: GameState) -> np.ndarray:
        """Encode game state into a flat feature vector (legacy)."""
        head_x, head_y = state.snake[0]
        food_x, food_y = state.food

        width, height = state.width, state.height

        direction_map = {"up": 0, "down": 1, "left": 2, "right": 3}
        direction_idx = direction_map.get(state.direction, 0)
        direction_onehot = [0.0] * 4
        direction_onehot[direction_idx] = 1.0

        dangers = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for dx, dy in directions:
            next_x, next_y = head_x + dx, head_y + dy
            is_danger = (
                next_x < 0 or next_x >= width or
                next_y < 0 or next_y >= height or
                (next_x, next_y) in state.walls or
                (next_x, next_y) in state.snake[:-1]
            )
            dangers.append(1.0 if is_danger else 0.0)

        food_dx = (food_x - head_x) / width if width > 0 else 0
        food_dy = (food_y - head_y) / height if height > 0 else 0

        features = [
            head_x / width if width > 0 else 0,
            head_y / height if height > 0 else 0,
            food_dx,
            food_dy,
            *direction_onehot,
            *dangers,
        ]

        return np.array(features, dtype=np.float32)

    def get_action(self, state: GameState) -> str:
        """Get the next action for the given state.

        Args:
            state: Current game state

        Returns:
            Action string ("up", "down", "left", "right")
        """
        if state.game_over:
            return random.choice(self.ACTIONS)

        # Use trained model if available
        if self.model is not None and TORCH_AVAILABLE:
            try:
                if self.model_type == 'cnn':
                    from algorithms.base import encode_state_grid
                    grid = encode_state_grid(state)
                    state_tensor = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
                else:
                    state_encoded = self._encode_state_legacy(state)
                    state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(state_tensor)
                    # Handle both DQN (returns Q-values) and actor-critic (returns probs, value)
                    if isinstance(output, tuple):
                        action_probs = output[0]
                        action_idx = action_probs.cpu().data.numpy().argmax()
                    else:
                        action_idx = output.cpu().data.numpy().argmax()
                    return self.ACTIONS[action_idx]
            except Exception as e:
                print(f"Model inference error: {e}, using heuristic")

        # Simple heuristic fallback
        head_x, head_y = state.snake[0]
        food_x, food_y = state.food

        dx = food_x - head_x
        dy = food_y - head_y

        valid_actions = []
        opposite = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left"
        }

        for action in self.ACTIONS:
            if state.direction and action == opposite.get(state.direction):
                continue
            valid_actions.append(action)

        if not valid_actions:
            return random.choice(self.ACTIONS)

        preferred_actions = []
        if abs(dx) > abs(dy):
            if dx > 0:
                preferred_actions.append("right")
            elif dx < 0:
                preferred_actions.append("left")
        else:
            if dy > 0:
                preferred_actions.append("down")
            elif dy < 0:
                preferred_actions.append("up")

        preferred_valid = [a for a in preferred_actions if a in valid_actions]

        safe_actions = []
        for action in (preferred_valid if preferred_valid else valid_actions):
            direction_map = {
                "up": (0, -1),
                "down": (0, 1),
                "left": (-1, 0),
                "right": (1, 0)
            }
            dx_move, dy_move = direction_map[action]
            next_pos = (head_x + dx_move, head_y + dy_move)

            if (next_pos not in state.walls and
                next_pos not in state.snake[:-1] and
                0 <= next_pos[0] < state.width and
                0 <= next_pos[1] < state.height):
                safe_actions.append(action)

        if safe_actions:
            if preferred_valid and random.random() < 0.8:
                preferred_safe = [a for a in preferred_valid if a in safe_actions]
                if preferred_safe:
                    return random.choice(preferred_safe)
            return random.choice(safe_actions)

        return random.choice(valid_actions)

    @classmethod
    def list_checkpoints(cls, checkpoints_dir: Optional[str] = None) -> List[str]:
        """Scan checkpoints directory and return list of checkpoint files.

        Args:
            checkpoints_dir: Optional custom checkpoints directory path

        Returns:
            List of checkpoint filenames (.pt or .pth files)
        """
        directory = checkpoints_dir or cls.CHECKPOINTS_DIR
        checkpoints = []

        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith((".pt", ".pth")):
                    checkpoints.append(filename)

        return sorted(checkpoints)
