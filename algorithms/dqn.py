"""Deep Q-Network (DQN) implementation with target network and optimizations.

Provides two agent variants:
  - DQNAgent: flat 24-feature vector input (fast, lightweight)
  - DQNCNNAgent: 7-channel grid input (sees entire board, better spatial reasoning)
"""
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional

from game.state import GameState
from .base import BaseAgent, encode_state, encode_state_grid
from .networks import DQNCNNNetwork


class DQNNetwork(nn.Module):
    """DQN Neural Network - deeper architecture for richer state."""
    
    def __init__(self, state_size=44, action_size=4, hidden_size=256):
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
    """Deep Q-Network Agent with target network and optimized training."""
    
    def __init__(self, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.05, 
                 epsilon_decay=0.9995, memory_size=10000, batch_size=64,
                 update_every=4, target_update_every=1000):
        super().__init__("DQN")
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.update_every = update_every          # Only train every N steps
        self.target_update_every = target_update_every  # Sync target network every N updates
        self.step_count = 0
        self.update_count = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)
        self.target_model = DQNNetwork(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Target network is never trained directly
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Cache for avoiding redundant encode_state calls
        self._last_encoded_state = None
        self._last_state_id = None
        
    def get_action(self, state: GameState, training: bool = True) -> str:
        """Get action using epsilon-greedy policy with guided exploration."""
        # Cache the encoded state so remember() can reuse it
        state_encoded = encode_state(state)
        self._last_encoded_state = state_encoded
        self._last_state_id = id(state)
        
        state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            best_action_idx = q_values.cpu().data.numpy().argmax()
        
        # Epsilon-greedy with guided exploration
        if training and np.random.random() <= self.epsilon:
            head_x, head_y = state.snake[0]
            food_x, food_y = state.food
            dx = food_x - head_x
            dy = food_y - head_y
            
            safe_actions = []
            opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
            direction_map = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
            
            # Build blocked set once
            snake_set = set(state.snake[:-1])
            wall_set = set(state.walls) if state.walls else set()
            
            for action in self.ACTIONS:
                if state.direction and action == opposite.get(state.direction):
                    continue
                dx_move, dy_move = direction_map[action]
                next_pos = (head_x + dx_move, head_y + dy_move)
                if (next_pos not in wall_set and 
                    next_pos not in snake_set and
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
        """Store experience in replay buffer. Uses cached encodings when possible."""
        action_idx = self.ACTIONS.index(action)
        
        # Reuse cached encoding for 'state' if it's the same object
        if self._last_state_id == id(state) and self._last_encoded_state is not None:
            state_encoded = self._last_encoded_state
        else:
            state_encoded = encode_state(state)
        
        next_state_encoded = encode_state(next_state)
        self.memory.append((state_encoded, action_idx, reward, next_state_encoded, done))
        self.step_count += 1
    
    def update(self):
        """Train the model on a batch of experiences.
        
        Only updates every `update_every` steps for efficiency.
        Uses target network for stable Q-value estimates.
        Applies gradient clipping for training stability.
        """
        # Only update every N steps
        if self.step_count % self.update_every != 0:
            return 0.0
            
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Use TARGET network for next Q values (Double DQN style stability)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)  # Huber loss for stability
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.update_count += 1
        
        # Periodically sync target network
        if self.update_count % self.target_update_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_checkpoint(self, path: str):
        """Save checkpoint including target network."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes': self.episode,
            'step_count': self.step_count,
            'update_count': self.update_count,
        }, path)
    
    def load_checkpoint(self, path: str) -> bool:
        """Load checkpoint from file. Returns True if successful."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # Load target network if available, otherwise copy from model
            if 'target_model_state_dict' in checkpoint:
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            else:
                self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Override optimizer LR to current setting (fine-tuning may need lower LR)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.optimizer.defaults.get('lr', param_group['lr'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.episode = checkpoint.get('episodes', 0)
            self.step_count = checkpoint.get('step_count', 0)
            self.update_count = checkpoint.get('update_count', 0)
            print(f"  Loaded checkpoint: epsilon={self.epsilon:.4f}, episodes={self.episode}, "
                  f"steps={self.step_count}, lr={self.optimizer.param_groups[0]['lr']}")
            return True
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}")
            return False


class DQNCNNAgent(BaseAgent):
    """DQN Agent with CNN encoder — sees the entire board as a 7-channel grid.
    
    This agent processes the full game board (head, body gradient, tail, food, 
    walls, direction gradient, reachability) through convolutional layers,
    giving it true spatial reasoning about the entire board state.
    
    Includes all training optimizations: target network, Huber loss, 
    gradient clipping, guided exploration, and experience replay.
    """
    
    def __init__(self, lr=0.0005, gamma=0.99, epsilon=1.0, epsilon_min=0.05, 
                 epsilon_decay=0.9995, memory_size=50000, batch_size=64,
                 update_every=4, target_update_every=1000,
                 board_width=20, board_height=20):
        super().__init__("DQN-CNN")
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.update_every = update_every
        self.target_update_every = target_update_every
        self.step_count = 0
        self.update_count = 0
        self.board_width = board_width
        self.board_height = board_height
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CNN model + target network
        self.model = DQNCNNNetwork(num_channels=self.NUM_CHANNELS, action_size=self.ACTION_SIZE).to(self.device)
        self.target_model = DQNCNNNetwork(num_channels=self.NUM_CHANNELS, action_size=self.ACTION_SIZE).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Cache
        self._last_encoded_grid = None
        self._last_state_id = None
        
    def get_action(self, state: GameState, training: bool = True) -> str:
        """Get action using epsilon-greedy with guided exploration."""
        # Encode as grid and cache
        grid = encode_state_grid(state)
        self._last_encoded_grid = grid
        self._last_state_id = id(state)
        
        grid_tensor = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(grid_tensor)
            best_action_idx = q_values.cpu().data.numpy().argmax()
        
        # Guided exploration (same logic as flat DQN)
        if training and np.random.random() <= self.epsilon:
            head_x, head_y = state.snake[0]
            food_x, food_y = state.food
            dx = food_x - head_x
            dy = food_y - head_y
            
            safe_actions = []
            opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
            direction_map = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
            
            snake_set = set(state.snake[:-1])
            wall_set = set(state.walls) if state.walls else set()
            
            for action in self.ACTIONS:
                if state.direction and action == opposite.get(state.direction):
                    continue
                dx_move, dy_move = direction_map[action]
                next_pos = (head_x + dx_move, head_y + dy_move)
                if (next_pos not in wall_set and 
                    next_pos not in snake_set and
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
        """Store experience as float16 grids to save memory."""
        action_idx = self.ACTIONS.index(action)
        
        # Reuse cached grid for current state
        if self._last_state_id == id(state) and self._last_encoded_grid is not None:
            grid = self._last_encoded_grid
        else:
            grid = encode_state_grid(state)
        
        next_grid = encode_state_grid(next_state)
        # Store as float16 to reduce memory (~50% savings for large grids)
        self.memory.append((grid.astype(np.float16), action_idx, reward, 
                           next_grid.astype(np.float16), done))
        self.step_count += 1
    
    def update(self):
        """Train with target network, Huber loss, and gradient clipping."""
        if self.step_count % self.update_every != 0:
            return 0.0
            
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        # Convert float16 back to float32 for training
        states = torch.FloatTensor(np.array([e[0] for e in batch], dtype=np.float32)).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch], dtype=np.float32)).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Target network for stable Q estimates
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)  # Huber loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.update_count += 1
        
        # Sync target network periodically
        if self.update_count % self.target_update_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_checkpoint(self, path: str):
        """Save checkpoint with CNN model type marker."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes': self.episode,
            'step_count': self.step_count,
            'update_count': self.update_count,
            'model_type': 'cnn',  # Marker so interface.py knows to use CNN
        }, path)
    
    def load_checkpoint(self, path: str) -> bool:
        """Load CNN checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'target_model_state_dict' in checkpoint:
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            else:
                self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.optimizer.defaults.get('lr', param_group['lr'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.episode = checkpoint.get('episodes', 0)
            self.step_count = checkpoint.get('step_count', 0)
            self.update_count = checkpoint.get('update_count', 0)
            print(f"  Loaded CNN checkpoint: epsilon={self.epsilon:.4f}, episodes={self.episode}, "
                  f"steps={self.step_count}, lr={self.optimizer.param_groups[0]['lr']}")
            return True
        except Exception as e:
            print(f"  Failed to load CNN checkpoint: {e}")
            return False
