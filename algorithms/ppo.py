"""Proximal Policy Optimization (PPO) implementation."""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple

from game.state import GameState
from .base import BaseAgent, encode_state


class PPONetwork(nn.Module):
    """PPO Actor-Critic Network."""
    
    def __init__(self, state_size=12, action_size=4, hidden_size=128):
        super(PPONetwork, self).__init__()
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


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent."""
    
    def __init__(self, lr=0.0003, gamma=0.99, eps_clip=0.2, 
                 update_epochs=4, batch_size=64):
        super().__init__("PPO")
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPONetwork(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Episode storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
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
        
        return self.ACTIONS[action_idx.item()]
    
    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self):
        """Update policy using PPO."""
        if len(self.states) == 0:
            return 0.0
        
        # Ensure all arrays have the same length
        min_len = min(len(self.states), len(self.actions), len(self.log_probs), len(self.rewards), len(self.dones))
        if min_len == 0:
            # Clear buffers and return
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
            self.dones = []
            return 0.0
        
        # Truncate lists to same length before converting to tensors
        states_list = self.states[:min_len]
        actions_list = self.actions[:min_len]
        log_probs_list = self.log_probs[:min_len]
        rewards_list = self.rewards[:min_len]
        dones_list = self.dones[:min_len]
        
        # Convert to tensors
        try:
            states = torch.FloatTensor(states_list).to(self.device)
            actions = torch.LongTensor(actions_list).to(self.device)
            old_log_probs = torch.FloatTensor(log_probs_list).to(self.device)
            rewards = np.array(rewards_list, dtype=np.float32)
            dones = np.array(dones_list, dtype=bool)
        except Exception as e:
            print(f"Error converting to tensors: {e}")
            # Clear buffers
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
            self.dones = []
            return 0.0
        
        # Compute discounted returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + (self.gamma * G * (1 - float(done)))
            returns.insert(0, G)
        
        try:
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Normalize returns (only if more than 1 sample and std > 0)
            if len(returns) > 1:
                returns_std = returns.std()
                if returns_std > 1e-8:
                    returns = (returns - returns.mean()) / returns_std
            
            # Get old values
            with torch.no_grad():
                _, old_values = self.model(states)
            old_values = old_values.squeeze()
            if old_values.dim() == 0:
                old_values = old_values.unsqueeze(0)
            
            # Ensure old_values matches returns length
            if len(old_values) != len(returns):
                min_val = min(len(old_values), len(returns))
                old_values = old_values[:min_val]
                returns = returns[:min_val]
                states = states[:min_val]
                actions = actions[:min_val]
                old_log_probs = old_log_probs[:min_val]
            
            # Compute advantages
            advantages = returns - old_values
            if len(advantages) > 1:
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / adv_std
        except Exception as e:
            print(f"Error computing returns/advantages: {e}")
            # Clear buffers
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
            self.dones = []
            return 0.0
        
        total_loss = 0.0
        
        # Update for multiple epochs
        try:
            for epoch in range(self.update_epochs):
                # Get new action probabilities and values
                action_probs, values = self.model(states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                values = values.squeeze()
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                
                # Ensure values match returns length
                if len(values) != len(returns):
                    min_val = min(len(values), len(returns))
                    values = values[:min_val]
                    returns_epoch = returns[:min_val]
                    advantages_epoch = advantages[:min_val]
                    new_log_probs = new_log_probs[:min_val]
                    old_log_probs_epoch = old_log_probs[:min_val]
                    actions_epoch = actions[:min_val]
                else:
                    returns_epoch = returns
                    advantages_epoch = advantages
                    old_log_probs_epoch = old_log_probs
                    actions_epoch = actions
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - old_log_probs_epoch)
                
                # Compute policy loss
                surr1 = ratio * advantages_epoch
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_epoch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.MSELoss()(values, returns_epoch)
                
                # Compute entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Check for NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected at epoch {epoch}, skipping update")
                    break
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        except Exception as e:
            print(f"Error during PPO update: {e}")
            import traceback
            traceback.print_exc()
            # Clear buffers
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
            self.dones = []
            return 0.0
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        
        return total_loss / self.update_epochs
    
    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episodes': self.episode,
        }, path)
