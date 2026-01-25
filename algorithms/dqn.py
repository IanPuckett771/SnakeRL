"""Deep Q-Network (DQN) implementation."""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional

from game.state import GameState
from .base import BaseAgent, encode_state


class DQNNetwork(nn.Module):
    """DQN Neural Network."""
    
    def __init__(self, state_size=12, action_size=4, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent."""
    
    def __init__(self, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.05, 
                 epsilon_decay=0.9995, memory_size=10000, batch_size=64):
        super().__init__("DQN")
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min  # Keep some exploration even late in training
        self.epsilon_decay = epsilon_decay  # Slower decay for longer training
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def get_action(self, state: GameState, training: bool = True) -> str:
        """Get action using epsilon-greedy policy with guided exploration."""
        state_encoded = encode_state(state)
        state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            best_action_idx = q_values.cpu().data.numpy().argmax()
        
        # Epsilon-greedy with guided exploration
        if training and np.random.random() <= self.epsilon:
            # During exploration, prefer actions that:
            # 1. Don't reverse direction
            # 2. Don't immediately cause collision
            # 3. Move toward food if safe
            
            head_x, head_y = state.snake[0]
            food_x, food_y = state.food
            dx = food_x - head_x
            dy = food_y - head_y
            
            # Get safe actions (not reversing, not colliding)
            safe_actions = []
            opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
            direction_map = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
            
            for action in self.ACTIONS:
                # Don't reverse
                if state.direction and action == opposite.get(state.direction):
                    continue
                
                # Check if safe
                dx_move, dy_move = direction_map[action]
                next_pos = (head_x + dx_move, head_y + dy_move)
                if (next_pos not in state.walls and 
                    next_pos not in state.snake[:-1] and
                    0 <= next_pos[0] < state.width and
                    0 <= next_pos[1] < state.height):
                    safe_actions.append(action)
            
            if safe_actions:
                # Prefer actions toward food during exploration
                preferred = []
                if abs(dx) > abs(dy):
                    preferred.append("right" if dx > 0 else "left")
                else:
                    preferred.append("down" if dy > 0 else "up")
                
                preferred_safe = [a for a in preferred if a in safe_actions]
                if preferred_safe and np.random.random() < 0.6:  # 60% chance to use preferred
                    return random.choice(preferred_safe)
                return random.choice(safe_actions)
            else:
                # No safe actions, use best from model
                return self.ACTIONS[best_action_idx]
        
        return self.ACTIONS[best_action_idx]
    
    def remember(self, state: GameState, action: str, reward: float, 
                 next_state: GameState, done: bool):
        """Store experience in replay buffer."""
        action_idx = self.ACTIONS.index(action)
        state_encoded = encode_state(state)
        next_state_encoded = encode_state(next_state)
        self.memory.append((state_encoded, action_idx, reward, next_state_encoded, done))
    
    def update(self):
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
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
        """Save checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes': self.episode,
        }, path)
