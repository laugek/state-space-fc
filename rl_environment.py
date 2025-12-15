#%%
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
- cash limit high enough to buy any players
"""


#%%
from models import Player, Position
from typing import List, Optional
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np


class FantasySoccerEnvV1(gym.Env):

    def __init__(self):

        # Init state
        self.roster: List[Player] = []
        self.team_value: float = self.calculate_roster_value()
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
            "forward": Discrete(2),
            # "midfielder": Discrete(2),
            # ... etc
        })

        # Keep track of index mapping to players and their value
        self._all_players_list = {
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

    def _action_to_player(self, action: dict) -> Player:
        """Map action number to Player object by their index."""
        return self._all_players_list[action["forward"]]

    def calculate_roster_value(self) -> float:
        """Calculate total value of current roster."""
        return sum(player.value for player in self.roster)

    def calculate_total_value(self) -> float:
        """Calculate total value of roster + cash."""
        return self.calculate_roster_value() + self.cash

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
            "all_players_value": {idx: player.value for idx, player in self._all_players_list.items()},
        }

    def _get_obs(self) -> dict:
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {
            "roster": [player.name for player in self.roster],
            "cash": self.cash,
        }

    def reset(self, seed=None, options=None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused now)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # A reset shouldn't have a random component, but always start from empty roster
        self.cash: float = 50_000_000 
        self.roster: List[Player] = []
        self.team_value: float = self.calculate_roster_value()
        self.total_value: float = self.calculate_total_value()

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

        # Update roster based on action taken
        self.roster = [self._action_to_player(action)]
        self.cash -= self.roster[0].value

        # Update team value and total value after games played
        # Now I'll just add fixed percentage increase for simplicity
        for _, player in self._all_players_list.items():
            player.value *= 1.05 

        # Update team value based on new player values
        self.team_value = self.calculate_roster_value()
        self.total_value = self.calculate_total_value()

        # Need to set stopping after x rounds of play
        terminated = False
        truncated = False

        reward = self.total_value
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

#%%
# Register the environment so we can create it with gym.make()
# gym.register(
#     id="gymnasium_env/FantasySoccerEnv-v1",
#     entry_point=FantasySoccerEnvV1,
# )
# env = gym.make("gymnasium_env/FantasySoccerEnv-v1")
#%%

# Init environment
env = FantasySoccerEnvV1()
from gymnasium.utils.env_checker import check_env
check_env(env)
#%%
observation, info = env.reset()
print(f"Starting observation: {observation}")

# Sample random action
sample_action = env.action_space.sample()
print(f"Sampled action: {sample_action}")
print(f"Action maps to player: {env._action_to_player(sample_action)}")

# Take the action and see what happens in env
observation, reward, terminated, truncated, info = env.step(sample_action)
print(f"New observation: {observation}")
print(f"Reward received: {reward}")
print(f"Terminated: {terminated}, Truncated: {truncated}")
print(f"Info: {info}")
# %%
env = FantasySoccerEnvV1()
env = gym.wrappers.FlattenObservation(gym.wrappers.RecordEpisodeStatistics(env))
observation, info = env.reset()
print(f"Starting observation: {observation}")
# %%
# Sample random action
sample_action = env.action_space.sample()
print(f"Sampled action: {sample_action}")
print(f"Action maps to player: {env._action_to_player(sample_action)}")

# Take the action and see what happens in env
observation, reward, terminated, truncated, info = env.step(sample_action)
print(f"New observation: {observation}")
print(f"Reward received: {reward}")
print(f"Terminated: {terminated}, Truncated: {truncated}")
print(f"Info: {info}")

# %%
# Simulate a couple of actions in the environment 
env = FantasySoccerEnvV1()
env = gym.wrappers.FlattenObservation(gym.wrappers.RecordEpisodeStatistics(env))
episode_over = False
total_reward = 0
for _ in range(5):  # Simulate 5 steps
    action = env.action_space.sample()

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()

# %% %%
