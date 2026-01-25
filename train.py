"""
Simple DQN training script for SnakeRL.
Trains an agent for approximately 1 minute and saves a checkpoint.
"""
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path

from game.engine import SnakeGame
from game.state import GameState


class DQN(nn.Module):
    """Simple Deep Q-Network for Snake."""
    
    def __init__(self, state_size=12, action_size=4, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


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
    food_dx = (food_x - head_x) / width
    food_dy = (food_y - head_y) / height
    
    # Combine all features
    features = [
        head_x / width,  # Normalized head x
        head_y / height,  # Normalized head y
        food_dx,  # Normalized food dx
        food_dy,  # Normalized food dy
        *direction_onehot,  # Direction encoding
        *dangers,  # Danger in 4 directions
    ]
    
    return np.array(features, dtype=np.float32)


class DQNAgent:
    """DQN Agent for training."""
    
    def __init__(self, state_size=12, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.batch_size = 64
        self.learning_rate = 0.001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.action_map = ["up", "down", "left", "right"]
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state_tensor)
        return q_values.cpu().data.numpy().argmax()
    
    def replay(self):
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
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


def train_agent(duration_seconds=60):
    """Train the agent for a specified duration."""
    import json
    
    print(f"Starting training for {duration_seconds} seconds...")
    
    # Create training lock file
    training_lock_file = Path(".training_lock")
    start_time = time.time()
    
    with open(training_lock_file, 'w') as f:
        json.dump({
            "start_time": start_time,
            "duration": duration_seconds,
            "episodes": 0,
            "avg_score": 0,
        }, f)
    
    game = SnakeGame(width=20, height=20)
    agent = DQNAgent(state_size=12, action_size=4)
    
    episode = 0
    scores = []
    
    try:
        # Training loop
        while time.time() - start_time < duration_seconds:
        episode += 1
        state = game.reset()
        state_encoded = encode_state(state)
        total_reward = 0
        steps = 0
        
        while not game.game_over:
            # Get action from agent
            action_idx = agent.act(state_encoded, training=True)
            action = agent.action_map[action_idx]
            
            # Take step
            next_state, reward, done = game.step(action)
            next_state_encoded = encode_state(next_state)
            
            # Store experience
            agent.remember(state_encoded, action_idx, reward, next_state_encoded, done)
            
            state_encoded = next_state_encoded
            total_reward += reward
            steps += 1
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            # Prevent infinite loops
            if steps > 1000:
                break
        
        scores.append(game.score)
        
        # Print progress every 10 episodes
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:]) if scores else 0
            elapsed = time.time() - start_time
            remaining = duration_seconds - elapsed
            
            # Update training lock file
            try:
                with open(training_lock_file, 'w') as f:
                    json.dump({
                        "start_time": start_time,
                        "duration": duration_seconds,
                        "episodes": episode,
                        "avg_score": avg_score,
                    }, f)
            except:
                pass
            
            print(f"Episode {episode:4d} | Avg Score: {avg_score:6.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | Time left: {remaining:.1f}s")
        
        # Save checkpoint
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / "trained_agent.pt"
        
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'episodes': episode,
            'avg_score': np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores) if scores else 0,
        }, checkpoint_path)
        
        print(f"\nTraining complete!")
        print(f"Episodes: {episode}")
        print(f"Final average score: {np.mean(scores[-50:]):.2f}" if len(scores) >= 50 else f"Final average score: {np.mean(scores):.2f}" if scores else "No scores")
        print(f"Checkpoint saved to: {checkpoint_path}")
        
        return agent, checkpoint_path
    finally:
        # Always remove training lock file
        if training_lock_file.exists():
            try:
                training_lock_file.unlink()
            except:
                pass


if __name__ == "__main__":
    # Train for 60 seconds (1 minute)
    train_agent(duration_seconds=60)
