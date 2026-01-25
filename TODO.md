# SnakeRL TODO & Roadmap

> **Last Updated:** January 25, 2026

---

## 🔴 Critical: Training Issues

The trained model is not performing well. These issues need to be investigated and fixed.

### Observed Problems

| Issue | Possible Cause | Suggested Fix |
|-------|----------------|---------------|
| **Goes in circles** | State encoding lacks directional history; agent exploits reward shaping by moving back and forth | Add movement history to state; implement anti-loop detection penalty |
| **Sometimes doesn't move** | May be a direction validation bug, or model outputting invalid actions | Debug action selection; verify direction change logic |
| **Doesn't go for treats** | Distance reward (+0.1) may be too weak relative to death penalty (-10) | Increase distance reward magnitude; add curriculum learning |
| **Doesn't seem to be learning** | Missing target network; epsilon decays too fast; learning rate may be too high | Implement Double DQN; tune hyperparameters; add gradient clipping |

### Current Reward Function (for reference)

From `game/engine.py` lines 174-254:

```
Positive Rewards:
  ├── Food eaten: +10 × food_points (10-200 based on rarity)
  ├── Quick treat bonus: +2.0 if eaten within 50 steps
  └── Moving toward food: +0.1 × (distance_delta / board_size)

Negative Rewards:
  ├── Death (collision): -10.0
  ├── Moving away from food: -0.05 × (distance_delta / board_size)
  └── Starvation penalty: -0.1 per step after 100 steps without food
```

### Action Items

- [ ] **Add Target Network (Double DQN)** — Critical for stable learning
  - 📖 [Deep RL Bootcamp: Double DQN](https://www.youtube.com/watch?v=fevMOp5TDQs)
  - 📖 [Original Double DQN Paper](https://arxiv.org/abs/1509.06461)
  
- [ ] **Enhance State Encoding** — Current 12 features may be insufficient
  - Add: last 3-4 directions taken (prevents loops)
  - Add: snake length as a feature
  - Add: wall positions relative to head
  - Consider: CNN with visual board representation
  - 📖 [State Representation in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

- [ ] **Tune Hyperparameters**
  - `epsilon_decay`: Currently 0.995 (fast). Try 0.9995-0.9999 for longer exploration
  - `epsilon_min`: Currently 0.01. Try 0.05 to maintain some exploration
  - `gamma`: Currently 0.95. Snake games benefit from higher (0.99) to value future food
  - `learning_rate`: Currently 0.001. Try 0.0005 or 0.0001 for stability
  - 📖 [Hyperparameter Tuning Guide](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

- [ ] **Add Gradient Clipping** — Prevents exploding gradients
  ```python
  torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
  ```

- [ ] **Implement Prioritized Experience Replay**
  - Learn more from rare events (eating food, dying)
  - 📖 [PER Paper](https://arxiv.org/abs/1511.05952)
  - 📖 [Implementation Guide](https://github.com/Curt-Park/rainbow-is-all-you-need)

---

## 🟡 High Priority: Tracking & Visualization

### Training Analytics

- [ ] **Average snake length chart per training session**
  - Track in W&B or create local matplotlib charts
  - Store `max_length` per episode alongside score
  - 📖 [W&B Custom Charts](https://docs.wandb.ai/guides/app/features/panels/code)

- [ ] **Checkpoint list improvements**
  - [ ] Show best/worst performing stage per training session
  - [ ] Make checkpoint list collapsible (stages grouped by session)
  - [ ] Add thumbnail previews or performance badges
  - Consider: Store metadata JSON alongside each .pt file

- [ ] **Q-Value Visualization**
  - Overlay Q-values on game board during agent play
  - Color-code cells by action preference
  - 📖 [Visualizing DQN](https://keon.github.io/deep-q-learning/)

### Suggested Metrics to Track

```
Per Episode:
  ├── Score (already tracked)
  ├── Max snake length
  ├── Steps to first food
  ├── Steps survived
  ├── Foods eaten
  └── Cause of death (wall/self/wall_obstacle)

Per Training Session:
  ├── Learning rate over time
  ├── Loss curve
  ├── Epsilon decay curve
  ├── Q-value distribution
  └── Action distribution
```

---

## 🟢 Medium Priority: Advanced Features

### Agent Improvements

- [ ] **Implement PPO (Proximal Policy Optimization)**
  - Already have `algorithms/ppo.py` — verify it works
  - Generally more stable than DQN for continuous training
  - 📖 [PPO Explained](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
  - 📖 [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

- [ ] **Implement A2C (Advantage Actor-Critic)**
  - Already have `algorithms/a2c.py` — verify it works
  - 📖 [A2C/A3C Explained](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

- [ ] **Curriculum Learning**
  - Start with smaller board, increase size as agent improves
  - Start without walls, add them gradually
  - 📖 [Curriculum Learning in RL](https://lilianweng.github.io/posts/2020-01-29-curriculum/)

- [ ] **Self-Play / Tournament Mode**
  - Run multiple agents simultaneously
  - Compare performance across algorithms
  - 📖 [Multi-Agent RL](https://github.com/PettingZoo-Team/PettingZoo)

### Gameplay Features

- [ ] **Playback Speed Control** — Slider for agent mode (0.5x to 5x)
- [ ] **Game Recording & Replay** — Save games as JSON, replay in UI
- [ ] **Multiple Food Items** — Spawn 2-3 foods at once
- [ ] **Different Game Modes** — Classic (no walls), Maze, Battle Royale

---

## 🔵 Low Priority: Quality of Life

### UI/UX Improvements

- [ ] **Pause/Resume** (P key or button)
- [ ] **Sound Effects**
  - Food eaten: positive chime
  - Death: game over sound
  - Rare treat: special sound
  - 📖 [Howler.js](https://howlerjs.com/) — Lightweight audio library

- [ ] **Mobile Touch Controls**
  - Swipe gestures for direction
  - On-screen D-pad overlay
  - 📖 [Hammer.js](https://hammerjs.github.io/) — Touch gesture library

- [ ] **Dark/Light Theme Toggle**
  - Store preference in localStorage
  - CSS variables already in use — easy to implement

- [ ] **Keyboard Shortcuts Help Modal** (? key to show)

---

## 📚 Resources & Learning

### RL Fundamentals

| Resource | Description |
|----------|-------------|
| [Spinning Up in Deep RL](https://spinningup.openai.com/) | OpenAI's RL educational resource |
| [Deep RL Course (Hugging Face)](https://huggingface.co/learn/deep-rl-course) | Free hands-on course |
| [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/) | Production-ready RL implementations |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | Single-file RL implementations |

### Snake-Specific RL Examples

| Resource | Description |
|----------|-------------|
| [Snake RL Tutorial](https://www.youtube.com/watch?v=PJl4iabBEz0) | Python Engineer YouTube series |
| [Pygame RL Snake](https://github.com/patrickloeber/snake-ai-pytorch) | Popular GitHub implementation |
| [Gymnasium Snake Env](https://gymnasium.farama.org/) | Standard RL environment interface |

### Tools You're Already Using

| Tool | Docs | Purpose |
|------|------|---------|
| PyTorch | [pytorch.org/docs](https://pytorch.org/docs/stable/index.html) | Neural network framework |
| Weights & Biases | [docs.wandb.ai](https://docs.wandb.ai/) | Experiment tracking |
| FastAPI | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) | Backend API |

---

## 🧪 Investigation Notes

### Questions to Answer

1. **What is currently giving reward to the model?**
   - ✅ Food eaten (+10 to +200 depending on rarity)
   - ✅ Moving toward food (+0.1 scaled by distance)
   - ✅ Quick eating bonus (+2.0)
   
2. **What is currently penalizing the model?**
   - ✅ Death by collision (-10)
   - ✅ Moving away from food (-0.05 scaled by distance)
   - ✅ Starvation after 100 steps (-0.1 per additional step)

3. **Shape of reward function?**
   - Sparse: Large rewards/penalties only on terminal events
   - Dense: Small distance-based shaping every step
   - ⚠️ Problem: Death penalty (-10) may be causing risk-averse behavior (going in circles to avoid walls)

### Experiments to Run

- [ ] Train with **no survival penalty** — see if agent becomes more aggressive
- [ ] Train with **higher distance reward** (+0.5 instead of +0.1)
- [ ] Train with **curriculum learning** — smaller board first
- [ ] Compare **DQN vs PPO vs A2C** on same seed/episodes
- [ ] Test with **visual CNN input** instead of feature vector

---

## ✅ Completed

- [x] Base game layout with configurable board sizes
- [x] Adjustable game speed
- [x] Snake color customization
- [x] Web-based frontend with real-time updates
- [x] Leaderboard system
- [x] RL Training Pipeline
  - [x] State encoding for neural network input
  - [x] Reward shaping (distance to food, survival bonus)
  - [x] Training loop with experience replay
  - [x] DQN agent implementation
- [x] Treat variety system (different colors/points)
- [x] Obstacle walls that respawn
- [x] W&B integration for experiment tracking

---

## 💡 Pro Tips

1. **Before long training runs**: Test with 100-200 episodes to verify learning is happening
2. **Save frequently**: The 10-stage checkpoint system is good — also save on best score
3. **Use seeds**: Set `torch.manual_seed()` and `random.seed()` for reproducibility
4. **Monitor GPU usage**: `nvidia-smi` or Task Manager to ensure CUDA is being used
5. **Log everything**: Better to have too much data in W&B than too little

---

*For questions or contributions, see [CONTRIBUTING](https://github.com/IanPuckett771/SnakeRL#contributing) in README.*
