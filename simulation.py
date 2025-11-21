#%%
"""
Simulation of football matches and tournaments using xG models.

I'm working on assumption that xG is available as a static value for a given player. 
I guess it should depend on match context like opponent for example in reality, but in the context of a Fantasy Cup where you purchase a specific player, I think this makes sense.

"""
import math
from typing import List, Dict, Iterator
import numpy as np
import pandas as pd
import plotly.express as px

from models import Position, Player, Team, Match

def odds_to_probability(odds: float) -> float:
    log_odds = math.log(odds)
    return 1 / (1 + math.exp(-log_odds))

def xG_to_probability(xG: float, max_goals: int = 10) -> Dict[int, float]:
    probabilities: Dict[int, float] = {}
    for k in range(max_goals + 1):
        probabilities[k] = (math.exp(-xG) * xG**k) / math.factorial(k)
    return probabilities

#%%
# Step 1: Simulation capabilities
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
denmark = Team(
    name="Denmark", 
    xG=1.9, 
    xGa=0.76,
    clean_sheet_pct=0.3,
    roster=[kasper_scmeichel, rasmus_hojlund], 
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

# Teams
france = Team(
    name="France", 
    xG=2.6, 
    xGa=0.73,
    clean_sheet_pct=0.25,
    roster=[mbappe, griez], 
)
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
match = Match(denmark, france)
result = match.simulate()
print(result)
print(result.players_scored_team_1)
print(result.players_scored_team_2)
#%%
difference_goals = []
for _ in range(10_000):
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
        return sorted(self.teams, key=lambda t: (self.points[t], self.goals[t]), reverse=True)

group = TournamentGroup(teams=[denmark, france, mix])
group.simulate()
for match in group.match_results:
    print(match)
print(f"group.points: {group.points}")
print(f"group.goals: {group.goals}")
print(f"group.ranking: {group.ranking()}")

#%%
class Knockout:

    def __init__(self, teams: List[Team]) -> None:
        self.teams = teams
        self.match_results: List['MatchResult'] = []
        self.simulated: bool = False
        # TODO: some concept of stages

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
                random_winner = np.random.choice([result.team_1, result.team_2], 1)[0]
                winners.append(random_winner)
        self.teams = winners
        self.simulated = True

    def ranking(self) -> List[Team]:
        if not self.simulated:
            raise ValueError("Knockout has not been simulated yet")
        return self.teams

knockout = Knockout(teams=[denmark, france, mix])
knockout.simulate()
for match in knockout.match_results:
    print(match)

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



