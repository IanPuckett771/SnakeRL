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
    class SimpleDQN(nn.Module):
        """Simple Deep Q-Network for Snake."""
        
        def __init__(self, state_size=12, action_size=4, hidden_size=128):
            super(SimpleDQN, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, action_size)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
else:
    # Dummy class if torch not available
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
        self.device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None

    def load_checkpoint(self, path: str) -> bool:
        """Load a model checkpoint.

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
                self.model = SimpleDQN(state_size=12, action_size=4).to(self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Old format - direct model save
                    self.model = checkpoint
                self.model.eval()  # Set to evaluation mode
                self.checkpoint_path = path
                return True
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return False
        return False
    
    def _encode_state(self, state: GameState) -> np.ndarray:
        """Encode game state into a feature vector for the neural network."""
        head_x, head_y = state.snake[0]
        food_x, food_y = state.food
        
        # Normalize positions to [0, 1]
        width, height = state.width, state.height
        
        # Direction one-hot encoding
        direction_map = {"up": 0, "down": 1, "left": 2, "right": 3}
        direction_idx = direction_map.get(state.direction, 0)
        direction_onehot = [0.0] * 4
        direction_onehot[direction_idx] = 1.0
        
        # Calculate danger in 4 directions (up, down, left, right)
        dangers = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        
        for dx, dy in directions:
            next_x, next_y = head_x + dx, head_y + dy
            # Check if next position is dangerous
            is_danger = (
                next_x < 0 or next_x >= width or
                next_y < 0 or next_y >= height or
                (next_x, next_y) in state.walls or
                (next_x, next_y) in state.snake[:-1]
            )
            dangers.append(1.0 if is_danger else 0.0)
        
        # Distance to food (normalized)
        food_dx = (food_x - head_x) / width if width > 0 else 0
        food_dy = (food_y - head_y) / height if height > 0 else 0
        
        # Combine all features
        features = [
            head_x / width if width > 0 else 0,  # Normalized head x
            head_y / height if height > 0 else 0,  # Normalized head y
            food_dx,  # Normalized food dx
            food_dy,  # Normalized food dy
            *direction_onehot,  # Direction encoding
            *dangers,  # Danger in 4 directions
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
                state_encoded = self._encode_state(state)
                state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                    action_idx = q_values.cpu().data.numpy().argmax()
                    return self.ACTIONS[action_idx]
            except Exception as e:
                # Fall back to heuristic if model fails
                print(f"Model inference error: {e}, using heuristic")
                pass
        
        # Simple heuristic: try to move towards food, avoid walls and self
        head_x, head_y = state.snake[0]
        food_x, food_y = state.food
        
        # Calculate direction to food
        dx = food_x - head_x
        dy = food_y - head_y
        
        # Get valid actions (avoid reversing direction)
        valid_actions = []
        opposite = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left"
        }
        
        for action in self.ACTIONS:
            # Don't reverse direction
            if state.direction and action == opposite.get(state.direction):
                continue
            valid_actions.append(action)
        
        if not valid_actions:
            return random.choice(self.ACTIONS)
        
        # Prefer actions that move towards food
        preferred_actions = []
        if abs(dx) > abs(dy):
            # Prefer horizontal movement
            if dx > 0:
                preferred_actions.append("right")
            elif dx < 0:
                preferred_actions.append("left")
        else:
            # Prefer vertical movement
            if dy > 0:
                preferred_actions.append("down")
            elif dy < 0:
                preferred_actions.append("up")
        
        # Filter preferred actions to only valid ones
        preferred_valid = [a for a in preferred_actions if a in valid_actions]
        
        # Check if preferred action would cause collision
        safe_actions = []
        for action in (preferred_valid if preferred_valid else valid_actions):
            # Calculate next position
            direction_map = {
                "up": (0, -1),
                "down": (0, 1),
                "left": (-1, 0),
                "right": (1, 0)
            }
            dx_move, dy_move = direction_map[action]
            next_pos = (head_x + dx_move, head_y + dy_move)
            
            # Check if next position is safe (not a wall, not snake body, within bounds)
            if (next_pos not in state.walls and 
                next_pos not in state.snake[:-1] and
                0 <= next_pos[0] < state.width and
                0 <= next_pos[1] < state.height):
                safe_actions.append(action)
        
        # Choose from safe actions, or fall back to random
        if safe_actions:
            # 80% chance to choose preferred safe action, 20% random safe action
            if preferred_valid and random.random() < 0.8:
                preferred_safe = [a for a in preferred_valid if a in safe_actions]
                if preferred_safe:
                    return random.choice(preferred_safe)
            return random.choice(safe_actions)
        
        # If no safe actions, return random (will likely cause game over)
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
