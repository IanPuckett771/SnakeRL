# SnakeRL

A classic Snake game with a web-based frontend, designed for training and visualizing Reinforcement Learning agents.

## Features

- **Play Mode** - Control the snake with arrow keys or WASD
- **Watch Agent Mode** - Load trained model checkpoints and watch the AI play
- **Customizable Settings**
  - Board size (10x10 to 50x50)
  - Snake color picker
  - Game speed (Slow, Normal, Fast, Insane)
- **Leaderboard** - Compete for high scores, persisted across sessions
- **Real-time WebSocket** - Smooth, responsive gameplay

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

**Linux/macOS:**
```bash
# Clone the repository
git clone https://github.com/IanPuckett771/SnakeRL.git
cd SnakeRL

# Install dependencies
make install
# or manually:
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
# Clone the repository
git clone https://github.com/IanPuckett771/SnakeRL.git
cd SnakeRL

# Install dependencies (choose one method)
.\Makefile.ps1 install
# or use the standalone script:
.\install.ps1
# or manually:
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

### Running the Game

**Linux/macOS:**
```bash
# Development mode (with auto-reload)
make dev

# Production mode
make run
```

**Windows (PowerShell):**
```powershell
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Development mode (with auto-reload)
.\Makefile.ps1 dev

# Production mode
.\Makefile.ps1 run
```

Then open http://localhost:8000 in your browser.

## How to Play

1. Select **Play** mode (default)
2. Adjust board size, snake color, and speed as desired
3. Click **Start Game**
4. Use **Arrow Keys** or **WASD** to control the snake
5. Eat the red food to grow and score points
6. Avoid hitting walls or yourself
7. Submit your score to the leaderboard!

## Project Structure

```
SnakeRL/
├── main.py              # FastAPI server & WebSocket handler
├── requirements.txt     # Python dependencies
├── Makefile             # Build/run commands
├── leaderboard.txt      # High scores (CSV format)
├── game/
│   ├── engine.py        # Core game logic
│   └── state.py         # Game state representation
├── agent/
│   └── interface.py     # RL agent interface (stub)
├── checkpoints/         # Trained model storage
└── static/
    ├── index.html       # Web UI
    ├── css/style.css    # Styling
    └── js/
        ├── game.js      # Game rendering & controls
        └── websocket.js # Server communication
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Frontend (Browser)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Canvas    │  │   Controls  │  │ Leaderboard │  │
│  │  (Game UI)  │  │ (WASD/Arrow)│  │  (Top 10)   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└────────────────────────┬────────────────────────────┘
                         │ WebSocket
┌────────────────────────┴────────────────────────────┐
│                 Backend (Python/FastAPI)            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Game Engine │  │   Agent     │  │ Leaderboard │  │
│  │   (Logic)   │  │  Interface  │  │   (File)    │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Game Mechanics

- **Rewards**: +10 for eating food, -10 for death
- **Collision**: Game ends on wall or self-collision
- **Growth**: Snake grows by 1 segment per food eaten
- **Direction**: Cannot reverse direction (no 180° turns)

## Training an RL Agent

Train a DQN (Deep Q-Network) agent to play Snake using the provided training script.

### Prerequisites

1. Install dependencies (includes PyTorch and Weights & Biases):
   ```powershell
   # Windows
   .\venv\Scripts\pip install -r requirements.txt
   
   # Linux/macOS
   pip install -r requirements.txt
   ```

2. Set up Weights & Biases (optional but recommended):
   ```powershell
   # Windows (activate venv first)
   .\venv\Scripts\Activate.ps1
   python -m wandb login
   
   # Linux/macOS
   wandb login
   ```
   You'll need an API key from https://wandb.ai/authorize
   Or run with `--no-wandb` to disable logging.

### Basic Training

```bash
# Train for 1000 episodes with default settings
python train.py

# Train with custom parameters
python train.py --episodes 2000 --board-size 20 --lr 0.001

# Train without W&B logging
python train.py --no-wandb
```

### Training Options

```bash
python train.py \
    --episodes 1000 \          # Number of training episodes
    --board-size 20 \          # Board size (20x20)
    --lr 0.001 \               # Learning rate
    --gamma 0.99 \             # Discount factor
    --batch-size 64 \          # Training batch size
    --save-interval 100 \      # Save checkpoint every N episodes
    --project snakerl \        # W&B project name
    --name my-run \            # W&B run name
    --resume                   # Resume from latest checkpoint
```

### Monitoring Training

- **Weights & Biases**: Training metrics are automatically logged. View at https://wandb.ai
- **Checkpoints**: Saved to `checkpoints/` directory
  - `dqn_latest.pt` - Latest checkpoint (updated periodically)
  - `dqn_final.pt` - Final model after training
  - `dqn_episode_N.pt` - Checkpoints at specific episodes

### Using Trained Models

Trained models can be loaded in the web interface:
1. Start the server: `python main.py` or `.\Makefile.ps1 dev`
2. Select "Watch Agent" mode
3. Choose a checkpoint from the dropdown
4. Click "Start Game" to watch the agent play

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve game UI |
| `/checkpoints` | GET | List available model checkpoints |
| `/leaderboard` | GET | Get top scores |
| `/leaderboard` | POST | Submit a new score |
| `/ws/game` | WebSocket | Real-time game communication |

## Makefile Commands

**Linux/macOS:**
```bash
make install  # Create venv and install dependencies
make dev      # Run with auto-reload (development)
make run      # Run production server
make clean    # Remove cache files
```

**Windows (PowerShell):**
```powershell
.\Makefile.ps1 install  # Create venv and install dependencies
.\Makefile.ps1 dev      # Run with auto-reload (development)
.\Makefile.ps1 run      # Run production server
.\Makefile.ps1 clean    # Remove cache files
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## TODO

### Completed
- [x] Base game layout with configurable board sizes
- [x] Adjustable game speed
- [x] Snake color customization
- [x] Web-based frontend with real-time updates
- [x] Leaderboard system

### In Progress
- [x] **RL Training Pipeline**
  - [x] Implement state encoding for neural network input
  - [x] Set up reward shaping (distance to food, survival bonus)
  - [x] Create training loop with experience replay
  - [x] Add DQN agent implementation

### Planned
- [ ] **Agent Visualization**
  - [ ] Save/load model checkpoints during training
  - [ ] Visualize Q-values or policy probabilities on the board
  - [ ] Training progress graphs (score over episodes)

- [ ] **Advanced Features**
  - [ ] Playback speed control for agent mode
  - [ ] Game recording and replay system
  - [ ] Multiple food items / obstacles
  - [ ] Tournament mode (agent vs agent)

- [ ] **Quality of Life**
  - [ ] Pause/resume functionality
  - [ ] Sound effects
  - [ ] Mobile touch controls
  - [ ] Dark/light theme toggle
