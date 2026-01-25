"""Base classes and utilities for RL algorithms."""
import numpy as np
from typing import Tuple
from game.state import GameState


def encode_state(state: GameState) -> np.ndarray:
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
    
    # Combine all features (12 features total)
    features = [
        head_x / width if width > 0 else 0,
        head_y / height if height > 0 else 0,
        food_dx,
        food_dy,
        *direction_onehot,  # 4 features
        *dangers,  # 4 features
    ]
    
    return np.array(features, dtype=np.float32)


class BaseAgent:
    """Base class for RL agents."""
    
    ACTIONS = ["up", "down", "left", "right"]
    STATE_SIZE = 12
    ACTION_SIZE = 4
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.episode = 0
        self.total_steps = 0
        
    def get_action(self, state: GameState, training: bool = True) -> str:
        """Get action from agent. Must be implemented by subclasses."""
        raise NotImplementedError
        
    def update(self, *args, **kwargs):
        """Update agent parameters. Must be implemented by subclasses."""
        raise NotImplementedError
        
    def save_checkpoint(self, path: str):
        """Save agent checkpoint. Must be implemented by subclasses."""
        raise NotImplementedError
