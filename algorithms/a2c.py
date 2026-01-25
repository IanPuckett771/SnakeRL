"""Advantage Actor-Critic (A2C) implementation."""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List

from game.state import GameState
from .base import BaseAgent, encode_state


class A2CNetwork(nn.Module):
    """A2C Actor-Critic Network."""
    
    def __init__(self, state_size=12, action_size=4, hidden_size=128):
        super(A2CNetwork, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)
        )
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_probs, value


class A2CAgent(BaseAgent):
    """Advantage Actor-Critic Agent."""
    
    def __init__(self, lr=0.0007, gamma=0.99, n_steps=5, entropy_coef=0.01):
        super().__init__("A2C")
        
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = A2CNetwork(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Episode storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def get_action(self, state: GameState, training: bool = True) -> str:
        """Get action from policy."""
        state_encoded = encode_state(state)
        state_tensor = torch.FloatTensor(state_encoded).unsqueeze(0).to(self.device)
        
        action_probs, value = self.model(state_tensor)
        dist = Categorical(action_probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        if training:
            self.states.append(state_encoded)
            self.actions.append(action_idx.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
        
        return self.ACTIONS[action_idx.item()]
    
    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self):
        """Update policy using A2C."""
        if len(self.states) < self.n_steps and not any(self.dones):
            return 0.0
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        # Compute returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + (self.gamma * G * (1 - done))
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Get new action probabilities and values
        action_probs, values = self.model(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        values = values.squeeze()
        
        # Compute advantages
        advantages = returns - old_values
        
        # Policy loss
        policy_loss = -(new_log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        return loss.item()
    
    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episodes': self.episode,
        }, path)
