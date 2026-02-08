"""Base classes and utilities for RL algorithms."""
import numpy as np
from collections import deque
from typing import Tuple
from game.state import GameState


# Pre-allocated direction tuples for speed
_DIRECTIONS_4 = ((0, -1), (0, 1), (-1, 0), (1, 0))  # up, down, left, right
_DIR_MAP = {"up": 0, "down": 1, "left": 2, "right": 3}
_DIR_VECTORS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _flood_fill_fast(start_x, start_y, width, height, blocked_set, max_count=100):
    """Fast flood fill using BFS. Cap at 100 cells for better space awareness at long lengths."""
    if (start_x < 0 or start_x >= width or start_y < 0 or start_y >= height or
            (start_x, start_y) in blocked_set):
        return 0
    visited = {(start_x, start_y)}
    queue = deque(((start_x, start_y),))
    count = 0
    while queue and count < max_count:
        x, y = queue.popleft()
        count += 1
        for ddx, ddy in _DIRECTIONS_4:
            nx, ny = x + ddx, y + ddy
            if (0 <= nx < width and 0 <= ny < height and
                    (nx, ny) not in visited and (nx, ny) not in blocked_set):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return count


def encode_state(state: GameState) -> np.ndarray:
    """Encode game state into a rich feature vector for the neural network.
    
    Optimized: uses fast flood fill (cap=20), pre-built sets, minimal allocations.
    
    Features (24 total):
    - Head position (2): normalized x, y
    - Food direction (2): normalized dx, dy to food
    - Direction one-hot (4): current movement direction
    - Immediate danger (4): blocked 1 cell ahead in each direction
    - 2-step danger (4): blocked 2 cells ahead in each direction
    - Reachable space per direction (4): flood fill from each neighbor (normalized)
    - Snake length (1): normalized by board area
    - Tail direction (2): normalized dx, dy from head to tail
    - Body density ahead (1): fraction of body segments in front of snake
    """
    snake = state.snake
    head_x, head_y = snake[0]
    food_x, food_y = state.food
    tail_x, tail_y = snake[-1]
    
    width, height = state.width, state.height
    
    # Build blocked set once (body excluding tail for collision, full body for flood fill)
    snake_body_set = set(snake[:-1])
    wall_set = set(state.walls) if state.walls else set()
    blocked_collision = snake_body_set | wall_set
    blocked_flood = set(snake) | wall_set  # Conservative for flood fill
    
    # Direction one-hot encoding (avoid list allocation)
    direction_idx = _DIR_MAP.get(state.direction, 0)
    
    # Pre-allocate output array
    features = np.zeros(24, dtype=np.float32)
    
    # Head position (normalized)
    features[0] = head_x / width
    features[1] = head_y / height
    
    # Food direction (normalized)
    features[2] = (food_x - head_x) / width
    features[3] = (food_y - head_y) / height
    
    # Direction one-hot
    features[4 + direction_idx] = 1.0
    
    # Danger + reachable space per direction
    for i, (dx, dy) in enumerate(_DIRECTIONS_4):
        nx1, ny1 = head_x + dx, head_y + dy
        
        # 1-step danger
        blocked_1 = (nx1 < 0 or nx1 >= width or ny1 < 0 or ny1 >= height or
                     (nx1, ny1) in blocked_collision)
        if blocked_1:
            features[8 + i] = 1.0   # danger_1
            features[12 + i] = 1.0  # danger_2 (also blocked)
            features[16 + i] = 0.0  # reachable = 0
        else:
            # 2-step danger
            nx2, ny2 = head_x + 2*dx, head_y + 2*dy
            if (nx2 < 0 or nx2 >= width or ny2 < 0 or ny2 >= height or
                    (nx2, ny2) in blocked_collision):
                features[12 + i] = 1.0
            
            # Reachable space (fast flood fill, cap=100 for long snake awareness)
            count = _flood_fill_fast(nx1, ny1, width, height, blocked_flood, max_count=100)
            features[16 + i] = count / 100.0
    
    # Snake length normalized
    snake_len = len(snake)
    features[20] = snake_len / (width * height)
    
    # Tail direction (normalized)
    features[21] = (tail_x - head_x) / width
    features[22] = (tail_y - head_y) / height
    
    # Body density ahead
    move_dx, move_dy = _DIR_VECTORS.get(state.direction, (1, 0))
    body_ahead = 0
    for j in range(1, snake_len):
        bx, by = snake[j]
        if (bx - head_x) * move_dx + (by - head_y) * move_dy > 0:
            body_ahead += 1
    features[23] = body_ahead / max(snake_len - 1, 1)
    
    return features


class BaseAgent:
    """Base class for RL agents."""
    
    ACTIONS = ["up", "down", "left", "right"]
    STATE_SIZE = 24  # Expanded from 12 to 24 features
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
