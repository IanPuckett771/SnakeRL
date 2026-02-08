"""Base classes and utilities for RL algorithms."""
import numpy as np
from collections import deque
from typing import Tuple
from game.state import GameState


def _is_blocked(x, y, width, height, snake_set, wall_set):
    """Check if a position is blocked (wall, body, or out of bounds)."""
    return (x < 0 or x >= width or y < 0 or y >= height or
            (x, y) in snake_set or (x, y) in wall_set)


def _flood_fill_count(start_x, start_y, width, height, snake_set, wall_set, max_count=50):
    """Count reachable cells from a position using BFS (capped for performance)."""
    if _is_blocked(start_x, start_y, width, height, snake_set, wall_set):
        return 0
    visited = set()
    visited.add((start_x, start_y))
    queue = deque([(start_x, start_y)])
    count = 0
    while queue and count < max_count:
        x, y = queue.popleft()
        count += 1
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and not _is_blocked(nx, ny, width, height, snake_set, wall_set):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return count


def encode_state(state: GameState) -> np.ndarray:
    """Encode game state into a rich feature vector for the neural network.
    
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
    head_x, head_y = state.snake[0]
    food_x, food_y = state.food
    tail_x, tail_y = state.snake[-1]
    
    width, height = state.width, state.height
    board_area = width * height
    
    snake_set = set(state.snake[:-1])  # Exclude tail (it will move)
    wall_set = set(state.walls) if state.walls else set()
    full_blocked = snake_set | wall_set  # For flood fill, include all body
    
    # Direction one-hot encoding
    direction_map = {"up": 0, "down": 1, "left": 2, "right": 3}
    direction_idx = direction_map.get(state.direction, 0)
    direction_onehot = [0.0] * 4
    direction_onehot[direction_idx] = 1.0
    
    # Immediate danger (1 cell ahead) in 4 directions
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
    dangers_1 = []
    dangers_2 = []
    reachable_space = []
    
    for dx, dy in directions:
        nx1, ny1 = head_x + dx, head_y + dy
        nx2, ny2 = head_x + 2*dx, head_y + 2*dy
        
        # 1-step danger
        is_danger_1 = _is_blocked(nx1, ny1, width, height, snake_set, wall_set)
        dangers_1.append(1.0 if is_danger_1 else 0.0)
        
        # 2-step danger
        is_danger_2 = _is_blocked(nx2, ny2, width, height, snake_set, wall_set)
        dangers_2.append(1.0 if is_danger_2 else 0.0)
        
        # Reachable space from this neighbor (how much room if we go this way)
        if is_danger_1:
            reachable_space.append(0.0)
        else:
            # Use full snake set for flood fill (conservative)
            full_snake_set = set(state.snake)
            count = _flood_fill_count(nx1, ny1, width, height, full_snake_set, wall_set, max_count=50)
            reachable_space.append(count / 50.0)  # Normalize to [0, 1]
    
    # Food direction (normalized)
    food_dx = (food_x - head_x) / width if width > 0 else 0
    food_dy = (food_y - head_y) / height if height > 0 else 0
    
    # Tail direction (normalized) - helps snake follow its tail
    tail_dx = (tail_x - head_x) / width if width > 0 else 0
    tail_dy = (tail_y - head_y) / height if height > 0 else 0
    
    # Snake length normalized
    snake_length_norm = len(state.snake) / board_area
    
    # Body density ahead of snake (in the direction we're moving)
    dir_vectors = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    move_dx, move_dy = dir_vectors.get(state.direction, (1, 0))
    body_ahead = 0
    for bx, by in state.snake[1:]:
        # Check if body segment is "ahead" (in the direction of movement)
        rel_x = bx - head_x
        rel_y = by - head_y
        # Dot product with movement direction
        if rel_x * move_dx + rel_y * move_dy > 0:
            body_ahead += 1
    body_density_ahead = body_ahead / max(len(state.snake) - 1, 1)
    
    # Combine all features (24 features total)
    features = [
        head_x / width if width > 0 else 0,       # 1: head x
        head_y / height if height > 0 else 0,      # 2: head y
        food_dx,                                     # 3: food direction x
        food_dy,                                     # 4: food direction y
        *direction_onehot,                           # 5-8: current direction
        *dangers_1,                                  # 9-12: immediate danger
        *dangers_2,                                  # 13-16: 2-step danger
        *reachable_space,                            # 17-20: reachable space per direction
        snake_length_norm,                           # 21: snake length
        tail_dx,                                     # 22: tail direction x
        tail_dy,                                     # 23: tail direction y
        body_density_ahead,                          # 24: body density ahead
    ]
    
    return np.array(features, dtype=np.float32)


def encode_state_grid(state: GameState) -> np.ndarray:
    """Encode game state as a 7-channel grid for CNN input.

    Channels:
        0: Snake head — 1.0 at head position
        1: Snake body — decaying gradient from 1.0 (neck) to 0.0 (tail)
        2: Snake tail — 1.0 at tail position
        3: Food — food_points / 20.0 at food cell
        4: Walls — 1.0 at each wall cell
        5: Direction — gradient in movement direction (full-board context)
        6: Reachability — BFS from head: 1.0 = reachable, 0.0 = blocked

    Returns:
        np.ndarray of shape (7, height, width), dtype float32
    """
    width, height = state.width, state.height
    grid = np.zeros((7, height, width), dtype=np.float32)

    head_x, head_y = state.snake[0]
    snake_len = len(state.snake)

    # Channel 0: Head
    grid[0, head_y, head_x] = 1.0

    # Channel 1: Body gradient (1.0 at neck → 0.0 at tail)
    if snake_len > 2:
        for i, (bx, by) in enumerate(state.snake[1:-1], start=1):
            grid[1, by, bx] = 1.0 - (i / (snake_len - 1))
    elif snake_len == 2:
        # Only head + tail, no body segments
        pass

    # Channel 2: Tail
    tail_x, tail_y = state.snake[-1]
    grid[2, tail_y, tail_x] = 1.0

    # Channel 3: Food (scaled by points)
    food_x, food_y = state.food
    grid[3, food_y, food_x] = state.food_points / 20.0

    # Channel 4: Walls
    if state.walls:
        for wx, wy in state.walls:
            grid[4, wy, wx] = 1.0

    # Channel 5: Direction gradient
    dir_vectors = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    dx, dy = dir_vectors.get(state.direction, (1, 0))
    if dx != 0:
        # Horizontal gradient
        for col in range(width):
            val = (col - head_x) * dx
            grid[5, :, col] = val / max(width - 1, 1)
    else:
        # Vertical gradient
        for row in range(height):
            val = (row - head_y) * dy
            grid[5, row, :] = val / max(height - 1, 1)

    # Channel 6: Reachability via BFS from head
    snake_set = set(state.snake[1:])  # body blocks movement
    wall_set = set(state.walls) if state.walls else set()
    blocked = snake_set | wall_set

    visited = np.zeros((height, width), dtype=np.float32)
    visited[head_y, head_x] = 1.0
    queue = deque([(head_x, head_y)])
    while queue:
        cx, cy = queue.popleft()
        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + ddx, cy + ddy
            if (0 <= nx < width and 0 <= ny < height
                    and visited[ny, nx] == 0.0
                    and (nx, ny) not in blocked):
                visited[ny, nx] = 1.0
                queue.append((nx, ny))
    grid[6] = visited

    return grid


class BaseAgent:
    """Base class for RL agents."""

    ACTIONS = ["up", "down", "left", "right"]
    STATE_SIZE = 24  # Legacy flat feature vector size
    NUM_CHANNELS = 7  # CNN grid channels
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
