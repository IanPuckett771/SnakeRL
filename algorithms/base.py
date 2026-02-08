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
