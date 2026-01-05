import os
import random
from pathlib import Path
from typing import List, Optional

from game.state import GameState


class AgentInterface:
    """Interface for RL agent to play Snake.

    This is a stub implementation that returns random actions.
    Will be replaced with actual RL agent later.
    """

    ACTIONS = ["up", "down", "left", "right"]
    CHECKPOINTS_DIR = "checkpoints"

    def __init__(self):
        """Initialize the agent interface."""
        self.checkpoint_path: Optional[str] = None
        self.model = None  # Placeholder for actual model

    def load_checkpoint(self, path: str) -> bool:
        """Load a model checkpoint.

        Args:
            path: Path to the checkpoint file

        Returns:
            True if checkpoint was found, False otherwise
        """
        if os.path.exists(path):
            self.checkpoint_path = path
            # Placeholder: actual model loading would happen here
            # self.model = torch.load(path)
            return True
        return False

    def get_action(self, state: GameState) -> str:
        """Get the next action for the given state.

        Args:
            state: Current game state

        Returns:
            Action string ("up", "down", "left", "right")
        """
        # Placeholder: returns random action
        # In real implementation, this would use the loaded model
        # to predict the best action based on state
        return random.choice(self.ACTIONS)

    @classmethod
    def list_checkpoints(cls, checkpoints_dir: Optional[str] = None) -> List[str]:
        """Scan checkpoints directory and return list of checkpoint files.

        Args:
            checkpoints_dir: Optional custom checkpoints directory path

        Returns:
            List of checkpoint filenames (.pt or .pth files)
        """
        directory = checkpoints_dir or cls.CHECKPOINTS_DIR
        checkpoints = []

        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith((".pt", ".pth")):
                    checkpoints.append(filename)

        return sorted(checkpoints)
