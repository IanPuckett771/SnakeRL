import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.interface import AgentInterface
from game.engine import SnakeGame
from game.tank_state import TankState
from game.tron_state import TronState

app = FastAPI(title="SnakeRL")

# Game registry with metadata
AVAILABLE_GAMES = {
    "snake": {
        "id": "snake",
        "name": "Snake",
        "description": "Classic snake game",
    },
    "tron": {
        "id": "tron",
        "name": "Tron",
        "description": "Light cycle battle",
    },
    "tank": {
        "id": "tank",
        "name": "Tank Battle",
        "description": "Tank arena combat",
    },
}

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


@app.get("/games")
async def get_games():
    """Return list of available games with metadata."""
    return {"games": list(AVAILABLE_GAMES.values())}


@app.get("/leaderboard")
async def get_leaderboard():
    """Read leaderboard.txt (CSV format) and return JSON array sorted by score descending."""
    entries = []

    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE) as f:
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
                            "date": parts[3] if len(parts) > 3 else "",
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

    def __init__(self, game_type: str = "snake"):
        self.game_type: str = game_type
        self.game: SnakeGame | None = None
        self.tron_state: TronState | None = None
        self.tank_state: TankState | None = None
        self.mode: str = "play"
        self.snake_color: str = "#00ff00"
        self.player1_color: str = "#00ff00"
        self.player2_color: str = "#ff0000"
        self.speed: float = 0.15
        self.pending_action: str | int | None = None
        self.pending_action_p2: int | None = None  # For Tron player 2
        self.game_task: asyncio.Task | None = None
        self.running: bool = False
        self.opponent_agent: AgentInterface | None = None  # AI opponent for Tron/Tank
        # Tank combined input state (for smooth simultaneous controls)
        self.tank_input: dict[str, bool] = {
            "forward": False,
            "backward": False,
            "turn_left": False,
            "turn_right": False,
            "shoot": False,
        }

    def get_current_state(self) -> Any:
        """Get the current state based on game type."""
        if self.game_type == "snake" and self.game:
            return self.game.get_state()
        elif self.game_type == "tron" and self.tron_state:
            return self.tron_state
        elif self.game_type == "tank" and self.tank_state:
            return self.tank_state
        return None

    def is_game_over(self) -> bool:
        """Check if the current game is over."""
        if self.game_type == "snake" and self.game:
            return self.game.game_over
        elif self.game_type == "tron" and self.tron_state:
            return self.tron_state.game_over
        elif self.game_type == "tank" and self.tank_state:
            return self.tank_state.game_over
        return True


async def run_snake_play_loop(websocket: WebSocket, session: GameSession, tick_rate: float = 0.15):
    """Run the game loop for Snake play mode - snake moves automatically."""
    session.running = True
    try:
        while session.running and session.game and not session.game.game_over:
            # Use pending action if set, otherwise continue in current direction
            action: str | None = (
                str(session.pending_action) if isinstance(session.pending_action, str) else None
            )
            session.pending_action = None  # Clear after use

            state, reward, done = session.game.step(action)

            await websocket.send_json(
                {
                    "type": "state_update",
                    "game": "snake",
                    "state": state.to_dict(),
                    "reward": reward,
                    "snake_color": session.snake_color,
                }
            )

            if done:
                await websocket.send_json(
                    {
                        "type": "game_over",
                        "game": "snake",
                        "state": state.to_dict(),
                        "final_score": state.score,
                    }
                )
                break

            await asyncio.sleep(tick_rate)
    except Exception as e:
        print(f"Snake game loop error: {e}")
    finally:
        session.running = False


# Backward compatibility alias
run_play_loop = run_snake_play_loop


def get_simple_tron_ai_action(state: TronState, player_idx: int) -> int:
    """Simple AI for Tron player 2.

    Tries to avoid immediate collisions by checking each direction.

    Args:
        state: Current Tron game state
        player_idx: 1 or 2 for which player to control

    Returns:
        Action (0=up, 1=down, 2=left, 3=right)
    """
    from game.tron_state import DIRECTION_VECTORS, Direction

    player = state.player1 if player_idx == 1 else state.player2

    # Check each direction for safety
    safe_directions: list[int] = []
    for direction in Direction:
        dx, dy = DIRECTION_VECTORS[direction]
        new_x = player.x + dx
        new_y = player.y + dy

        # Check bounds
        if new_x < 0 or new_x >= state.width or new_y < 0 or new_y >= state.height:
            continue

        # Check trail collision
        if (new_x, new_y) in state.player1.trail or (new_x, new_y) in state.player2.trail:
            continue

        safe_directions.append(int(direction))

    if safe_directions:
        # Prefer continuing in current direction if safe
        current_dir = int(player.direction)
        if current_dir in safe_directions:
            return current_dir
        # Otherwise pick a random safe direction
        import random

        return random.choice(safe_directions)

    # No safe direction - just continue forward
    return int(player.direction)


async def run_tron_play_loop(websocket: WebSocket, session: GameSession, tick_rate: float = 0.15):
    """Run the game loop for Tron play mode - both players move automatically."""
    session.running = True
    try:
        while session.running and session.tron_state and not session.tron_state.game_over:
            # Player 1 action from pending, player 2 from AI
            action1 = (
                session.pending_action
                if session.pending_action is not None
                else int(session.tron_state.player1.direction)
            )
            session.pending_action = None

            # Player 2: use trained agent if available, otherwise simple AI
            if session.opponent_agent and session.opponent_agent.agent:
                action2_raw = session.opponent_agent.get_action(session.tron_state, player=2)
                action2 = int(action2_raw) if isinstance(action2_raw, int) else 0
            else:
                action2 = get_simple_tron_ai_action(session.tron_state, 2)

            # Execute step
            session.tron_state = session.tron_state.step(int(action1), action2)

            await websocket.send_json(
                {
                    "type": "state_update",
                    "game": "tron",
                    "state": session.tron_state.to_dict(),
                    "player1_color": session.player1_color,
                    "player2_color": session.player2_color,
                }
            )

            if session.tron_state.game_over:
                await websocket.send_json(
                    {
                        "type": "game_over",
                        "game": "tron",
                        "state": session.tron_state.to_dict(),
                        "winner": session.tron_state.winner,
                    }
                )
                break

            await asyncio.sleep(tick_rate)
    except Exception as e:
        print(f"Tron game loop error: {e}")
    finally:
        session.running = False


async def run_tank_play_loop(websocket: WebSocket, session: GameSession, tick_rate: float = 0.033):
    """Run the game loop for Tank Battle play mode.

    Uses combined input for smooth simultaneous controls (move + turn + shoot).
    Default tick rate of ~30fps for smooth gameplay.
    """
    session.running = True
    try:
        while session.running and session.tank_state and not session.tank_state.game_over:
            # Use combined input state for smooth controls
            inp = session.tank_input
            session.tank_state, reward, done = session.tank_state.step_multi(
                forward=inp["forward"],
                backward=inp["backward"],
                turn_left=inp["turn_left"],
                turn_right=inp["turn_right"],
                shoot=inp["shoot"],
            )

            await websocket.send_json(
                {
                    "type": "state_update",
                    "game": "tank",
                    "state": session.tank_state.to_dict(),
                    "reward": reward,
                }
            )

            if done:
                await websocket.send_json(
                    {
                        "type": "game_over",
                        "game": "tank",
                        "state": session.tank_state.to_dict(),
                        "final_score": session.tank_state.score,
                    }
                )
                break

            await asyncio.sleep(tick_rate)
    except Exception as e:
        print(f"Tank game loop error: {e}")
    finally:
        session.running = False


@app.websocket("/ws/game")
async def websocket_game(
    websocket: WebSocket,
    game: str = Query(default="snake", description="Game type: snake, tron, or tank"),
):
    """WebSocket endpoint for real-time game communication.

    Supports multiple games via the `game` query parameter:
    - /ws/game?game=snake (default)
    - /ws/game?game=tron
    - /ws/game?game=tank
    """
    await websocket.accept()

    # Validate game type
    if game not in AVAILABLE_GAMES:
        await websocket.send_json(
            {
                "type": "error",
                "message": f"Unknown game: {game}. Available: {list(AVAILABLE_GAMES.keys())}",
            }
        )
        await websocket.close()
        return

    session = GameSession(game_type=game)

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

                # Common settings
                width = message.get("width", 20)
                height = message.get("height", 20)
                session.mode = message.get("mode", "play")
                session.speed = message.get("speed", 0.15)
                checkpoint = message.get("checkpoint")

                # Initialize game based on type
                if session.game_type == "snake":
                    session.snake_color = message.get("snake_color", "#00ff00")
                    session.game = SnakeGame(width=width, height=height)
                    state = session.game.reset()
                    session.pending_action = None

                    await websocket.send_json(
                        {
                            "type": "state_update",
                            "game": "snake",
                            "state": state.to_dict(),
                            "snake_color": session.snake_color,
                        }
                    )

                    if session.mode == "agent":
                        agent = AgentInterface(game="snake")
                        if checkpoint:
                            checkpoint_path = CHECKPOINTS_DIR / checkpoint
                            agent.load_checkpoint(str(checkpoint_path))
                        session.game_task = asyncio.create_task(
                            run_agent_loop(websocket, session, agent)
                        )
                    else:
                        session.game_task = asyncio.create_task(
                            run_snake_play_loop(websocket, session, session.speed)
                        )

                elif session.game_type == "tron":
                    session.player1_color = message.get("player1_color", "#00ff00")
                    session.player2_color = message.get("player2_color", "#ff0000")
                    session.tron_state = TronState.create(width=width, height=height)
                    session.pending_action = None

                    # Load opponent AI if checkpoint provided
                    opponent_checkpoint = message.get("opponent_checkpoint")
                    if opponent_checkpoint:
                        session.opponent_agent = AgentInterface(game="tron")
                        opp_path = CHECKPOINTS_DIR / opponent_checkpoint
                        session.opponent_agent.load_checkpoint(str(opp_path))
                    else:
                        session.opponent_agent = None

                    await websocket.send_json(
                        {
                            "type": "state_update",
                            "game": "tron",
                            "state": session.tron_state.to_dict(),
                            "player1_color": session.player1_color,
                            "player2_color": session.player2_color,
                        }
                    )

                    if session.mode == "agent":
                        agent = AgentInterface(game="tron")
                        if checkpoint:
                            checkpoint_path = CHECKPOINTS_DIR / checkpoint
                            agent.load_checkpoint(str(checkpoint_path))
                        session.game_task = asyncio.create_task(
                            run_tron_agent_loop(websocket, session, agent)
                        )
                    else:
                        session.game_task = asyncio.create_task(
                            run_tron_play_loop(websocket, session, session.speed)
                        )

                elif session.game_type == "tank":
                    num_enemies = message.get("num_enemies", 3)
                    session.tank_state = TankState.create(
                        width=width, height=height, num_enemies=num_enemies
                    )
                    session.pending_action = None
                    # Reset combined input state
                    session.tank_input = {
                        "forward": False,
                        "backward": False,
                        "turn_left": False,
                        "turn_right": False,
                        "shoot": False,
                    }

                    # Load opponent AI if checkpoint provided (for smarter enemies)
                    opponent_checkpoint = message.get("opponent_checkpoint")
                    if opponent_checkpoint:
                        session.opponent_agent = AgentInterface(game="tank")
                        opp_path = CHECKPOINTS_DIR / opponent_checkpoint
                        session.opponent_agent.load_checkpoint(str(opp_path))
                    else:
                        session.opponent_agent = None

                    await websocket.send_json(
                        {
                            "type": "state_update",
                            "game": "tank",
                            "state": session.tank_state.to_dict(),
                        }
                    )

                    if session.mode == "agent":
                        agent = AgentInterface(game="tank")
                        if checkpoint:
                            checkpoint_path = CHECKPOINTS_DIR / checkpoint
                            agent.load_checkpoint(str(checkpoint_path))
                        session.game_task = asyncio.create_task(
                            run_tank_agent_loop(websocket, session, agent)
                        )
                    else:
                        session.game_task = asyncio.create_task(
                            run_tank_play_loop(websocket, session, session.speed)
                        )

            elif msg_type == "action" and session.mode == "play":
                # Queue the action for the next tick
                action = message.get("action")

                if session.game_type == "snake":
                    if action in ["up", "down", "left", "right"]:
                        session.pending_action = action

                elif session.game_type == "tron":
                    # Tron uses numeric actions: 0=up, 1=down, 2=left, 3=right
                    direction_map = {"up": 0, "down": 1, "left": 2, "right": 3}
                    if action in direction_map:
                        session.pending_action = direction_map[action]
                    elif isinstance(action, int) and 0 <= action <= 3:
                        session.pending_action = action

                elif session.game_type == "tank":
                    # Tank uses: 0=forward, 1=backward, 2=turn_left, 3=turn_right, 4=shoot
                    action_map = {
                        "forward": 0,
                        "backward": 1,
                        "turn_left": 2,
                        "turn_right": 3,
                        "shoot": 4,
                    }
                    if action in action_map:
                        session.pending_action = action_map[action]
                    elif isinstance(action, int) and 0 <= action <= 4:
                        session.pending_action = action

            elif msg_type == "tank_input" and session.game_type == "tank":
                # Combined tank input for smooth controls (multiple keys held simultaneously)
                session.tank_input["forward"] = message.get("forward", False)
                session.tank_input["backward"] = message.get("backward", False)
                session.tank_input["turn_left"] = message.get("turn_left", False)
                session.tank_input["turn_right"] = message.get("turn_right", False)
                session.tank_input["shoot"] = message.get("shoot", False)

            elif msg_type == "reset":
                # Stop current game loop
                session.running = False
                if session.game_task and not session.game_task.done():
                    session.game_task.cancel()
                    try:
                        await session.game_task
                    except asyncio.CancelledError:
                        pass

                session.pending_action = None

                # Reset based on game type
                if session.game_type == "snake" and session.game:
                    state = session.game.reset()
                    await websocket.send_json(
                        {
                            "type": "state_update",
                            "game": "snake",
                            "state": state.to_dict(),
                            "snake_color": session.snake_color,
                        }
                    )
                    if session.mode == "play":
                        session.game_task = asyncio.create_task(
                            run_snake_play_loop(websocket, session, session.speed)
                        )

                elif session.game_type == "tron" and session.tron_state:
                    session.tron_state = session.tron_state.reset()
                    await websocket.send_json(
                        {
                            "type": "state_update",
                            "game": "tron",
                            "state": session.tron_state.to_dict(),
                            "player1_color": session.player1_color,
                            "player2_color": session.player2_color,
                        }
                    )
                    if session.mode == "play":
                        session.game_task = asyncio.create_task(
                            run_tron_play_loop(websocket, session, session.speed)
                        )

                elif session.game_type == "tank" and session.tank_state:
                    session.tank_state = session.tank_state.reset()
                    # Reset combined input state
                    session.tank_input = {
                        "forward": False,
                        "backward": False,
                        "turn_left": False,
                        "turn_right": False,
                        "shoot": False,
                    }
                    await websocket.send_json(
                        {
                            "type": "state_update",
                            "game": "tank",
                            "state": session.tank_state.to_dict(),
                        }
                    )
                    if session.mode == "play":
                        session.game_task = asyncio.create_task(
                            run_tank_play_loop(websocket, session, session.speed)
                        )

    except WebSocketDisconnect:
        session.running = False
    except Exception as e:
        session.running = False
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                }
            )
        except Exception:
            pass


async def run_agent_loop(
    websocket: WebSocket,
    session: GameSession,
    agent: AgentInterface,
    delay: float = 0.1,
):
    """Run the Snake agent loop, sending state updates with delay.

    Args:
        websocket: WebSocket connection
        session: GameSession instance
        agent: AgentInterface instance
        delay: Delay between moves in seconds
    """
    try:
        while session.game and not session.game.game_over:
            # Get agent's action
            state = session.game.get_state()
            action_raw = agent.get_action(state)
            action: str | None = str(action_raw) if isinstance(action_raw, str) else None

            # Execute action
            state, reward, done = session.game.step(action)

            # Send state update
            await websocket.send_json(
                {
                    "type": "state_update",
                    "game": "snake",
                    "state": state.to_dict(),
                    "reward": reward,
                    "action": action,
                    "snake_color": session.snake_color,
                }
            )

            if done:
                await websocket.send_json(
                    {
                        "type": "game_over",
                        "game": "snake",
                        "state": state.to_dict(),
                        "final_score": state.score,
                    }
                )
                break

            # Delay between moves
            await asyncio.sleep(delay)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                }
            )
        except Exception:
            pass


async def run_tron_agent_loop(
    websocket: WebSocket,
    session: GameSession,
    agent: AgentInterface,
    delay: float = 0.1,
):
    """Run the Tron agent loop.

    Agent controls player 1, simple AI controls player 2.

    Args:
        websocket: WebSocket connection
        session: GameSession instance
        agent: AgentInterface instance for player 1
        delay: Delay between moves in seconds
    """
    try:
        while session.tron_state and not session.tron_state.game_over:
            # Get agent's action for player 1
            action1_raw = agent.get_action(session.tron_state)
            action1 = int(action1_raw) if isinstance(action1_raw, int) else 0

            # Simple AI for player 2
            action2 = get_simple_tron_ai_action(session.tron_state, 2)

            # Execute step
            session.tron_state = session.tron_state.step(action1, action2)

            # Send state update
            await websocket.send_json(
                {
                    "type": "state_update",
                    "game": "tron",
                    "state": session.tron_state.to_dict(),
                    "action": action1,
                    "player1_color": session.player1_color,
                    "player2_color": session.player2_color,
                }
            )

            if session.tron_state.game_over:
                await websocket.send_json(
                    {
                        "type": "game_over",
                        "game": "tron",
                        "state": session.tron_state.to_dict(),
                        "winner": session.tron_state.winner,
                    }
                )
                break

            await asyncio.sleep(delay)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                }
            )
        except Exception:
            pass


async def run_tank_agent_loop(
    websocket: WebSocket,
    session: GameSession,
    agent: AgentInterface,
    delay: float = 0.1,
):
    """Run the Tank Battle agent loop.

    Args:
        websocket: WebSocket connection
        session: GameSession instance
        agent: AgentInterface instance
        delay: Delay between moves in seconds
    """
    try:
        while session.tank_state and not session.tank_state.game_over:
            # Get agent's action
            action_raw = agent.get_action(session.tank_state)
            action = int(action_raw) if isinstance(action_raw, int) else 0

            # Execute step
            session.tank_state, reward, done = session.tank_state.step(action)

            # Send state update
            await websocket.send_json(
                {
                    "type": "state_update",
                    "game": "tank",
                    "state": session.tank_state.to_dict(),
                    "reward": reward,
                    "action": action,
                }
            )

            if done:
                await websocket.send_json(
                    {
                        "type": "game_over",
                        "game": "tank",
                        "state": session.tank_state.to_dict(),
                        "final_score": session.tank_state.score,
                    }
                )
                break

            await asyncio.sleep(delay)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                }
            )
        except Exception:
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

    end_port = start_port + max_attempts - 1
    raise RuntimeError(f"No available port found in range {start_port}-{end_port}")


if __name__ == "__main__":
    import uvicorn

    default_port = 8000
    port = int(os.environ.get("PORT", 0)) or find_available_port(default_port)

    if port != default_port:
        print(f"Port {default_port} is in use, using port {port} instead")

    print(f"Starting server at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
