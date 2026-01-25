import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from game.engine import SnakeGame
from agent.interface import AgentInterface

app = FastAPI(title="SnakeRL")

# Paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
LEADERBOARD_FILE = BASE_DIR / "leaderboard.txt"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"


class LeaderboardEntry(BaseModel):
    """Model for leaderboard entry."""

    name: str
    score: int
    board_size: str  # e.g., "20x20"


@app.get("/")
async def serve_index():
    """Serve the main index.html file."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"error": "index.html not found"}


@app.get("/checkpoints")
async def get_checkpoints():
    """Return list of available checkpoints."""
    checkpoints = AgentInterface.list_checkpoints(str(CHECKPOINTS_DIR))
    return {"checkpoints": checkpoints}


@app.get("/checkpoint-info/{checkpoint_name}")
async def get_checkpoint_info(checkpoint_name: str):
    """Return metadata for a specific checkpoint."""
    checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
    
    if not checkpoint_path.exists():
        return {"error": "Checkpoint not found"}, 404
    
    try:
        import torch
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        
        # Extract metadata if available
        metadata = {}
        if isinstance(checkpoint, dict):
            if "episodes" in checkpoint:
                metadata["episodes"] = checkpoint["episodes"]
            if "avg_score" in checkpoint:
                metadata["avg_score"] = float(checkpoint["avg_score"])
            if "epsilon" in checkpoint:
                metadata["epsilon"] = float(checkpoint["epsilon"])
        
        return metadata
    except Exception as e:
        return {"error": str(e)}, 500


@app.get("/training-status")
async def get_training_status():
    """Check if training is currently running by looking for training lock file."""
    import time
    import json
    
    training_lock_file = BASE_DIR / ".training_lock"
    
    if training_lock_file.exists():
        try:
            with open(training_lock_file, 'r') as f:
                data = json.load(f)
            return {
                "training": True,
                "start_time": data.get("start_time", 0),
                "duration": data.get("duration", 60),
                "elapsed": time.time() - data.get("start_time", time.time()),
                "episodes": data.get("episodes", 0),
                "avg_score": data.get("avg_score", 0),
            }
        except:
            pass
    
    return {"training": False}


@app.get("/leaderboard")
async def get_leaderboard():
    """Read leaderboard.txt (CSV format) and return JSON array sorted by score descending."""
    entries = []

    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE, "r") as f:
            lines = f.readlines()
            # Skip header line if present
            for line in lines:
                line = line.strip()
                if not line or line.startswith("name,"):
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        entry = {
                            "name": parts[0],
                            "score": int(parts[1]),
                            "board_size": parts[2],
                            "date": parts[3] if len(parts) > 3 else ""
                        }
                        entries.append(entry)
                    except (ValueError, IndexError):
                        continue

    # Sort by score descending
    entries.sort(key=lambda x: x.get("score", 0), reverse=True)

    return {"leaderboard": entries}


@app.post("/leaderboard")
async def add_leaderboard_entry(entry: LeaderboardEntry):
    """Add entry to leaderboard.txt in CSV format."""
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_line = f"{entry.name},{entry.score},{entry.board_size},{date_str}\n"

    # Append to file
    with open(LEADERBOARD_FILE, "a") as f:
        f.write(csv_line)

    return {"status": "success", "entry": entry.model_dump()}


class GameSession:
    """Manages a single game session."""
    def __init__(self):
        self.game: Optional[SnakeGame] = None
        self.mode: str = "play"
        self.snake_color: str = "#00ff00"
        self.speed: float = 0.15
        self.pending_action: Optional[str] = None
        self.game_task: Optional[asyncio.Task] = None
        self.running: bool = False


async def run_play_loop(websocket: WebSocket, session: GameSession, tick_rate: float = 0.15):
    """Run the game loop for play mode - snake moves automatically."""
    session.running = True
    try:
        while session.running and session.game and not session.game.game_over:
            # Use pending action if set, otherwise continue in current direction
            action = session.pending_action
            session.pending_action = None  # Clear after use

            state, reward, done = session.game.step(action)

            await websocket.send_json({
                "type": "state_update",
                "state": state.to_dict(),
                "reward": reward,
                "snake_color": session.snake_color,
            })

            if done:
                await websocket.send_json({
                    "type": "game_over",
                    "state": state.to_dict(),
                    "final_score": state.score,
                })
                break

            await asyncio.sleep(tick_rate)
    except Exception as e:
        print(f"Game loop error: {e}")
    finally:
        session.running = False


@app.websocket("/ws/game")
async def websocket_game(websocket: WebSocket):
    """WebSocket endpoint for real-time game communication."""
    await websocket.accept()

    session = GameSession()

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "start_game":
                # Stop any existing game loop
                session.running = False
                if session.game_task and not session.game_task.done():
                    session.game_task.cancel()
                    try:
                        await session.game_task
                    except asyncio.CancelledError:
                        pass

                # Initialize game with provided settings
                width = message.get("width", 20)
                height = message.get("height", 20)
                session.mode = message.get("mode", "play")
                session.snake_color = message.get("snake_color", "#00ff00")
                session.speed = message.get("speed", 0.15)
                checkpoint = message.get("checkpoint")

                session.game = SnakeGame(width=width, height=height)
                state = session.game.reset()
                session.pending_action = None

                # Send initial state
                await websocket.send_json({
                    "type": "state_update",
                    "state": state.to_dict(),
                    "snake_color": session.snake_color,
                })

                # Start the appropriate game loop
                if session.mode == "agent":
                    agent = AgentInterface()
                    if checkpoint:
                        checkpoint_path = CHECKPOINTS_DIR / checkpoint
                        agent.load_checkpoint(str(checkpoint_path))
                    session.game_task = asyncio.create_task(
                        run_agent_loop(websocket, session.game, agent, session.snake_color)
                    )
                else:
                    # Play mode - start auto-move loop
                    session.game_task = asyncio.create_task(
                        run_play_loop(websocket, session, session.speed)
                    )

            elif msg_type == "action" and session.game and session.mode == "play":
                # Queue the action for the next tick
                action = message.get("action")
                if action in ["up", "down", "left", "right"]:
                    session.pending_action = action

            elif msg_type == "reset" and session.game:
                # Stop current game loop
                session.running = False
                if session.game_task and not session.game_task.done():
                    session.game_task.cancel()

                # Reset the game
                state = session.game.reset()
                session.pending_action = None

                await websocket.send_json({
                    "type": "state_update",
                    "state": state.to_dict(),
                    "snake_color": session.snake_color,
                })

                # Restart game loop
                if session.mode == "play":
                    session.game_task = asyncio.create_task(
                        run_play_loop(websocket, session, session.speed)
                    )

    except WebSocketDisconnect:
        session.running = False
    except Exception as e:
        session.running = False
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except:
            pass


async def run_agent_loop(
    websocket: WebSocket,
    game: SnakeGame,
    agent: AgentInterface,
    snake_color: str,
    delay: float = 0.2,  # Increased from 0.1 to 0.2 for better viewing
):
    """Run the agent loop, sending state updates with delay.

    Args:
        websocket: WebSocket connection
        game: SnakeGame instance
        agent: AgentInterface instance
        snake_color: Color for the snake
        delay: Delay between moves in seconds
    """
    try:
        while not game.game_over:
            # Get agent's action
            state = game.get_state()
            action = agent.get_action(state)

            # Execute action
            state, reward, done = game.step(action)

            # Send state update
            await websocket.send_json({
                "type": "state_update",
                "state": state.to_dict(),
                "reward": reward,
                "action": action,
                "snake_color": snake_color,
            })

            if done:
                await websocket.send_json({
                    "type": "game_over",
                    "state": state.to_dict(),
                    "final_score": state.score,
                })
                break

            # Delay between moves
            await asyncio.sleep(delay)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except:
            pass


# Mount static files (after all routes)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port.

    Args:
        start_port: Port number to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        An available port number

    Raises:
        RuntimeError: If no available port is found
    """
    import socket

    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue

    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts - 1}")


if __name__ == "__main__":
    import uvicorn

    default_port = 8000
    port = int(os.environ.get("PORT", 0)) or find_available_port(default_port)

    if port != default_port:
        print(f"Port {default_port} is in use, using port {port} instead")

    print(f"Starting server at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
