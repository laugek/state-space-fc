#%%
"""
Simulation of football matches and tournaments using xG models.

I'm working on assumption that xG is available as a static value for a given player. 
I guess it should depend on match context like opponent for example in reality, but in the context of a Fantasy Cup where you purchase a specific player, I think this makes sense.

"""
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Iterator
import numpy as np
import pandas as pd
import plotly.express as px
import gymnasium as gym

def odds_to_probability(odds: float) -> float:
    log_odds = math.log(odds)
    return 1 / (1 + math.exp(-log_odds))

def xG_to_probability(xG: float, max_goals: int = 10) -> Dict[int, float]:
    probabilities: Dict[int, float] = {}
    for k in range(max_goals + 1):
        probabilities[k] = (math.exp(-xG) * xG**k) / math.factorial(k)
    return probabilities

#%%
# Step 0: Define some data structures
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
    position=Position.FORWARD,
    value=8_000_000,
    xG=1.2
)
griez = Player(
    name="Antoine Griezmann", 
    position=Position.FORWARD, 
    value=7_000_000,
    xG=0.56
)

print(rasmus_hojlund)

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
        self.proba_goals_conceded = xG_to_probability(self.xGa)

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

print(denmark)

#%%
# Step 1: Simulation capabilities
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
#%%
match = Match(denmark, france)
result = match.simulate()
print(result)
print(result.players_scored_team_1)
print(result.players_scored_team_2)
#%%
difference_goals = []
for _ in range(100_000):
    match = Match(denmark, france)
    result = match.simulate()
    difference_goals.append(result.goals_team_1 - result.goals_team_2)  
#%%
# Histrograme and dotted line for median
px.histogram(pd.Series(difference_goals), 
    title="Distribution of goal difference (Denmark - France)",
    labels={"value": "Goal difference"},
    nbins=30
).add_vline(
    x=pd.Series(difference_goals).median(),
    line_dash="dash",
    line_color="red",
    annotation_text="Median",
    annotation_position="top left"
).show()
#%%
class TournamentGroup:

    def __init__(self, teams: List[Team]) -> None:
        self.teams = teams
        # instance-level state (previously class-level)
        self.match_results: List['MatchResult'] = []
        self.simulated: bool = False
        self.points: Dict[Team, int] = {}
        self.goals: Dict[Team, int] = {}

    def matches(self) -> Iterator['Match']:
        # Generate all possible matches in the group stage (single round-robin)
        for i, team_1 in enumerate(self.teams):
            for team_2 in self.teams[i + 1:]:
                yield Match(team_1, team_2)

    def simulate(self) -> None:
        # reset state for repeated simulations
        self.points = {team: 0 for team in self.teams}
        self.goals = {team: 0 for team in self.teams}
        self.match_results = []

        for match in self.matches():
            match_result = match.simulate()
            self.match_results.append(match_result)

            if match_result.winner:
                self.points[match_result.winner] += 3
            else:
                self.points[match_result.team_1] += 1
                self.points[match_result.team_2] += 1

            self.goals[match_result.team_1] += match_result.goals_team_1
            self.goals[match_result.team_2] += match_result.goals_team_2

        self.simulated = True

    def ranking(self) -> List[Team]:
        if not self.simulated:
            raise ValueError("Matches has not been simulated yet")
        # sort by points then goals (both descending)
        sorted_teams = sorted(self.teams, key=lambda t: (self.points[t], self.goals[t]), reverse=True)
        return sorted_teams[:2]

group = TournamentGroup(teams=[denmark, france, mix])
group.simulate()
for match in group.match_results:
    print(match)
print(group.points)
print(group.goals)
print(group.ranking())

#%%
class Knockout:

    def __init__(self, teams: List[Team]) -> None:
        self.teams = teams
        self.match_results: List['MatchResult'] = []
        self.simulated: bool = False

    def matches(self) -> Iterator['Match']:
        # simple pairing: 0 vs 1, 2 vs 3, ...
        for i in range(0, len(self.teams), 2):
            if i + 1 < len(self.teams):
                yield Match(self.teams[i], self.teams[i + 1])

    def simulate(self) -> None:
        self.match_results = []
        winners: List[Team] = []
        for match in self.matches():
            result = match.simulate()
            self.match_results.append(result)
            if result.winner:
                winners.append(result.winner)
            else:
                # tie-breaker: random choice between the two teams
                winners.append(np.random.choice([result.team_1, result.team_2]))
        self.teams = winners
        self.simulated = True

    def ranking(self) -> List[Team]:
        if not self.simulated:
            raise ValueError("Knockout has not been simulated yet")
        return self.teams

#%%
class Tournament:
    def __init__(
            self, 
            group_stage: List[TournamentGroup],
            # TODO: implement
            knockout_stage = Knockout 
        ) -> None:
        self.group_stage = group_stage

    def simulate(self):
        # TournamentGroup stages
        for group in self.group_stage:
            group.simulate()
        
        # Knockout stages



#%%
# Step 2: Define a Gym Environment
class FantasySoccerEnv(gym.Env):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = config or {}
        self.budget = config.get("budget", 50_000_000)
        self.lineup_style = config.get("lineup_style", [4, 4, 3])
        self.tournament_schedule = config.get("tournament_schedule", [])
        # state placeholders
        self.team_roster: List[Player] = []
        self.current_step = 0
        self.max_steps = config.get("max_steps", 10)

    def reset(self):
        """Reset environment and return a simple observation dict."""
        self.current_step = 0
        self.team_roster = []
        return self._get_observation()

    def reset(self, *, seed: int | None = None, options: Dict[str, math.Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """Apply action (no-op placeholder), advance step, return obs, reward, done, info."""
        self.current_step += 1
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_termination()
        info = {}
        return obs, reward, done, info

    def render(self):
        print(f"Step {self.current_step} | Budget: {self.budget} | Roster size: {len(self.team_roster)}")

    def _get_observation(self) -> dict:
        return {
            "budget": self.budget,
            "lineup_style": list(self.lineup_style),
            "roster_size": len(self.team_roster),
            "step": self.current_step
        }

    def _compute_reward(self) -> float:
        # placeholder reward, can be replaced with performance-based metric
        return 0.0

    def _check_termination(self) -> bool:
        return self.current_step >= self.max_steps
