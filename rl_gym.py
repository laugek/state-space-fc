"""
Key Design Questions (gymnasium create custom evn)[https://gymnasium.farama.org/introduction/create_custom_env/]
🎯 What skill should the agent learn?
Buy and sell players to maximize accumulated value of players over a tournament

👀 What information does the agent need?
- Price of all players
- Model of expected player value development 
- Tournament schedule 
- Team rankings

🎮 What actions can the agent take?
In each round of tournament, the agent can:
- Buy players
- Sell players
- Choose a captain (get double value from this player for the round)

🏆 How do we measure success?
- Total value of team roster at end of tournament
- Or some greedy component of each round?

⏰ When should episodes end?
- After tournament is finished


Version 1: as simple as possible 
- one player per team
- no captain
- two players to choose from
- same teams play each other each round
"""


from models import Player, Position
from typing import List, Optional
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
class FantasySoccerEnvV1(gym.Env):

    def __init__(self):

        # Init state
        self.roster: List[Player] = []
        self.team_value: float = self.calculate_team_value()
        self.cash: float = 50_000_000 
        self.total_value: float = self.team_value + self.cash

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict({
            "roster": gym.spaces.MultiDiscrete([1, 1]),
            "cash": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Dict({
            "foward": MultiDiscrete(np.array([1, 1])) # would be +100 considering all players
            # "midfielder": MultiDiscrete(np.array([1, 1])),
            # ... etc
        })

        # Map action numbers to actual actions
        # This makes the code more readable than using raw numbers
        self._action_to_player = {
            0: Player(
                name="Rasmus Hojlund", 
                position=Position.FORWARD, 
                value=6_000_000,
                xG=0.89
            ),
            1: Player(
                name="Kylian Mbappé",
                position=Position.FORWARD,
                value=8_000_000,
                xG=1.2
            )
            # 2: Player (...) I wonder how I'll scale this.
        }

    def calculate_team_value(self) -> float:
        """Calculate total value of current roster."""
        return sum(player.value for player in self.roster)

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "roster": [player.name for player in self.roster],
            "team_value": self.team_value,
            "cash": self.cash,
            "total_value": self.total_value,
        }

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {
            "roster": [player.name for player in self.roster],
            "cash": self.cash,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused now)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self.roster: List[Player] = []
        self.team_value: float = self.calculate_team_value()
        self.cash: float = 50_000_000 
        self.total_value: float = self.team_value + self.cash

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        roster = some_function(action)

        # Need to set stopping after x rounds of play
        terminated = False
        truncated = False

        reward = self.total_value
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
