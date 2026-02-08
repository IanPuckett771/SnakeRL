"""Deep Q-Network (DQN) implementation with CNN encoder."""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional

from game.state import GameState
from .base import BaseAgent, encode_state, encode_state_grid
from .networks import DQNCNNNetwork


# Keep legacy FC network for checkpoint backward compat
class DQNNetwork(nn.Module):
    """Legacy DQN Neural Network (flat feature vector)."""

    def __init__(self, state_size=24, action_size=4, hidden_size=256):
        super(DQNNetwork, self).__init__()
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


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent with CNN encoder."""

    def __init__(self, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.9995, memory_size=10000, batch_size=64,
                 board_width=20, board_height=20):
        super().__init__("DQN")

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.board_width = board_width
        self.board_height = board_height

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNCNNNetwork(self.NUM_CHANNELS, self.ACTION_SIZE).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, state: GameState, training: bool = True) -> str:
        """Get action using epsilon-greedy policy with guided exploration."""
        grid = encode_state_grid(state)
        state_tensor = torch.FloatTensor(grid).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor)
            best_action_idx = q_values.cpu().data.numpy().argmax()

        # Epsilon-greedy with guided exploration
        if training and np.random.random() <= self.epsilon:
            head_x, head_y = state.snake[0]
            food_x, food_y = state.food
            dx = food_x - head_x
            dy = food_y - head_y

            # Get safe actions (not reversing, not colliding)
            safe_actions = []
            opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
            direction_map = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

            for action in self.ACTIONS:
                if state.direction and action == opposite.get(state.direction):
                    continue
                dx_move, dy_move = direction_map[action]
                next_pos = (head_x + dx_move, head_y + dy_move)
                if (next_pos not in state.walls and
                    next_pos not in state.snake[:-1] and
                    0 <= next_pos[0] < state.width and
                    0 <= next_pos[1] < state.height):
                    safe_actions.append(action)

            if safe_actions:
                preferred = []
                if abs(dx) > abs(dy):
                    preferred.append("right" if dx > 0 else "left")
                else:
                    preferred.append("down" if dy > 0 else "up")

                preferred_safe = [a for a in preferred if a in safe_actions]
                if preferred_safe and np.random.random() < 0.6:
                    return random.choice(preferred_safe)
                return random.choice(safe_actions)
            else:
                return self.ACTIONS[best_action_idx]

        return self.ACTIONS[best_action_idx]

    def remember(self, state: GameState, action: str, reward: float,
                 next_state: GameState, done: bool):
        """Store experience in replay buffer as float16 grids."""
        action_idx = self.ACTIONS.index(action)
        grid = encode_state_grid(state).astype(np.float16)
        next_grid = encode_state_grid(next_state).astype(np.float16)
        self.memory.append((grid, action_idx, reward, next_grid, done))

    def update(self):
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch], dtype=np.float32)).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch], dtype=np.float32)).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save_checkpoint(self, path: str):
        """Save checkpoint with model_type marker."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes': self.episode,
            'model_type': 'cnn',
        }, path)

    def load_checkpoint(self, path: str) -> bool:
        """Load checkpoint from file. Returns True if successful."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if checkpoint.get('model_type') == 'cnn':
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                # Legacy flat checkpoint — incompatible architecture, start fresh CNN
                print("  Legacy flat checkpoint detected, starting fresh CNN training")
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.episode = checkpoint.get('episodes', 0)
            print(f"  Loaded checkpoint: epsilon={self.epsilon:.4f}, episodes={self.episode}")
            return True
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}")
            return False
