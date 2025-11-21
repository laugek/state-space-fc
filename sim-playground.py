#%%
import gym
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import pandas as pd

import math

def odds_to_probability(odds):
    log_odds = math.log(odds)
    probability = 1 / (1 + math.exp(-log_odds))
    return probability

def xG_to_probability(xG, max_goals=10):
    probabilities = {}
    for k in range(max_goals + 1):
        prob_k = (math.exp(-xG) * xG**k) / math.factorial(k)
        probabilities[k] = prob_k
    return probabilities

#%%
class Position(Enum):
    GOALKEEPER = 1
    DEFENDER = 2
    MIDFIELDER = 3
    FORWARD = 4

@dataclass
class Player:
    name: str
    position: Position 
    value: float
    xG: float

    def __repr__(self) -> str:
        return f"{self.name} ({self.position.name})"

    def __hash__(self):
        return hash(self.name)

kasper_scmeichel = Player(
    name="Kasper Schmeichel", 
    position=Position.GOALKEEPER, 
    value=5_000_000,
    xG=0.01
)
rasmus_hojlund = Player(
    name="Rasmus Hojlund", 
    position=Position.FORWARD, 
    value=6_000_000,
    xG=0.89
)

mbappe = Player(
    name="Kylian Mbappé", 
    position=Position.GOALKEEPER, 
    value=8_000_000,
    xG=1.2
)
griez = Player(
    name="Antoine Griezmann", 
    position=Position.FORWARD, 
    value=7_000_000,
    xG=0.56
)


rasmus_hojlund

#%%
@dataclass
class Team: 
    name: str
    roster: List[Player]
    xG: float
    xGa: float
    clean_sheet_pct: float

    def __post_init__(self):
        self.xGD = self.xG - self.xGa
        self.proba_goals_scored = xG_to_probability(self.xG)
        self.proba_goals_conceeded = xG_to_probability(self.xGa)

    def __repr__(self) -> str:
        return f"{self.name}"

    def __hash__(self):
        return hash(self.name)

france = Team(
    name="France", 
    xG=2.6, 
    xGa=0.73,
    clean_sheet_pct=0.25,
    roster=[mbappe, griez], 
)

denmark = Team(
    name="Denmark", 
    xG=1.9, 
    xGa=0.76,
    clean_sheet_pct=0.3,
    roster=[kasper_scmeichel, rasmus_hojlund], 
)

mix = Team(
    name="Mix", 
    xG=2.1, 
    xGa=0.99,
    clean_sheet_pct=0.15,
    roster=[mbappe, rasmus_hojlund], 
)

denmark

#%%
def allocate_goals(goals: int, players: List[Player]):
    total_xG = sum([player.xG for player in players])
    normalized_xG = {player.name: player.xG / total_xG for player in players}
    goals_distribution = np.random.choice(list(normalized_xG.keys()), size=goals, p=list(normalized_xG.values()))
    return {
        player: goals_distribution.tolist().count(player.name) 
        for player in players
    }

allocate_goals(1000, denmark.roster)
#%%
@dataclass
class MatchResult:
    team_1: Team
    team_2: Team

    goals_team_1: int
    goals_team_2: int

    players_scored_team_1: dict[Player, int]
    players_scored_team_2: dict[Player, int]

    def __post_init__(self):
        if self.goals_team_1 > self.goals_team_2:
            self.winner = self.team_1
        elif self.goals_team_1 < self.goals_team_2:
            self.winner = self.team_2
        else:
            self.winner = None

    def __repr__(self) -> str:
        return f"{self.team_1} {self.goals_team_1} - {self.goals_team_2} {self.team_2}"

class Match:

    def __init__(self, team_1: Team, team_2: Team) -> None:
        self.team_1 = team_1
        self.team_2 = team_2

    def simulate(self) -> MatchResult:
        # Goals scored as team
        # TODO: Figure out how good teams should score more against worse than average teams
        # TODO: Goals scored a quite high. 
        goals_team_1 = np.random.poisson(self.team_1.xG)
        goals_team_2 = np.random.poisson(self.team_2.xG)

        # Individual goals scored
        # TODO: Figure out how to simulate who's playing the game or not
        players_scored_team_1 = allocate_goals(goals_team_1, self.team_1.roster)
        players_scored_team_2 = allocate_goals(goals_team_2, self.team_2.roster)

        return MatchResult(
            team_1=self.team_1,
            team_2=self.team_2,
            goals_team_1=goals_team_1,
            goals_team_2=goals_team_2,
            players_scored_team_1=players_scored_team_1,
            players_scored_team_2=players_scored_team_2
        )

match = Match(denmark, france)
result = match.simulate()
result
print(result)
print(result.players_scored_team_1)
print(result.players_scored_team_2)

#%%
class TournamentGroup:

    def __init__(self, teams: List[Team]) -> None:
        self.teams = teams

    match_results: List[MatchResult] = []
    simulated: bool = False

    def matches(self):
        # Generate all possible matches in the group stage
        # Teams don't play home and away, just once
        for i, team_1 in enumerate(self.teams):
            for team_2 in self.teams[i + 1:]:
                yield Match(team_1, team_2)

    def simulate(self):
        self.points = {team: 0 for team in self.teams}
        self.goals = {team: 0 for team in self.teams}

        # Simulate all matches in the tournament
        for match in self.matches():
            match_result = match.simulate() 
            self.match_results.append(match_result)

            # allocate points to teams
            if match_result.winner:
                self.points[match_result.winner] = self.points[match_result.winner] + 3
            else:
                self.points[match_result.team_1] = self.points[match_result.team_1] + 1
                self.points[match_result.team_2] = self.points[match_result.team_2] + 1

            # store goals
            self.goals[match_result.team_1] = self.goals[match_result.team_1] + match_result.goals_team_1
            self.goals[match_result.team_2] = self.goals[match_result.team_2] + match_result.goals_team_2

        self.simulated = True

    def ranking(self):
        # Return the top two teams in the group
        if not self.simulated:
            raise ValueError("Matches has not been simulated yet")

        # TODO: fix ties by looking at points, then then goals scored?
        sorted_teams = sorted(self.teams, key=lambda team: (self.points[team], self.goals[team]), reverse=True)
        return sorted_teams[:2]

group = TournamentGroup(teams=[denmark, france, mix])
group.simulate()
for match in group.match_results:
    print(match)
print(group.points)
print(group.goals)
print(group.ranking())

#%%


#%%
class Knockout:

    def __init__(self, teams: List[Team]) -> None:
        self.teams = teams

    match_results: List[MatchResult] = []
    simulated: bool = False

    def matches(self):
        pass

    def simulate(self):
        pass

    def ranking(self):
        # Return the top two teams in the group
        if not self.simulated:
            raise ValueError("Matches has not been simulated yet")


#%%
class Tournament:
    def __init__(
            self, 
            group_stage: List[TournamentGroup],
            knockout_stage = Knockout 
        ) -> None:
        self.group_stage = group_stage

    def simulate(self):
        # TournamentGroup stages
        for group in self.group_stage:
            group.simulate()
        
        # Knockout stages



#%%
goals_denmark, goals_france = [], []
for _ in range(10_000):
    math_result = simulate_match(denmark, france)
    goals_denmark.append(math_result.goals_team_1)
    goals_france.append(math_result.goals_team_2)

df = pd.DataFrame({"goals_denmark": goals_denmark, "goals_france": goals_france})
df['outcome'] = df.apply(
    lambda row: 'win_dk' if row['goals_denmark'] > row['goals_france']
    else (
        'draw' if row['goals_denmark'] == row['goals_france'] 
        else 'win_fr'
    ),
    axis=1
)   
df.describe(percentiles=[0.90, 0.95, 0.99])
#%%
df['outcome'].value_counts(normalize=True)
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
class FantasySoccerEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        # Initialize environment parameters
        self.budget = config.get("budget", 50_000_000)  # Default budget DKK
        self.lineup_style = config.get("lineup_style", [4, 4, 3])  # Default lineup style
        self.tournament_schedule = config.get("tournament_schedule", [])  # Future games

        # Other initialization (player values, team composition, etc.)
        # ...

    def reset(self):
        # Reset environment for a new episode
        # Initialize player values, team composition, budget, etc.
        # ...
        observation = self._get_observation()  # Define observation function
        return observation

    def step(self, action):
        # Execute the chosen action (buy/sell players, allocate budget)
        # Update player values, team composition, budget, etc.
        # Compute reward based on team performance
        # Check termination conditions
        # ...
        next_observation = self._get_observation()  # Update observation
        reward = self._compute_reward()  # Define reward function
        done = self._check_termination()  # Define termination conditions
        return next_observation, reward, done, {}

    def render(self):
        # Optionally visualize the environment (print relevant info)
        # ...
        pass

    def _get_observation(self):
        # Define how the agent observes the state
        # Return relevant features (player values, team composition, etc.)
        # ...
        return observation

    def _compute_reward(self):
        # Define the reward function based on team performance
        # Consider player values, match outcomes, etc.
        # ...
        return reward

    def _check_termination(self):
        # Define termination conditions (end of tournament, budget depletion, etc.)
        # ...
        return done
