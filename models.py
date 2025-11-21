from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

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

@dataclass
class Team:
    name: str
    roster: List[Player]
    xG: float
    xGa: float
    clean_sheet_pct: float

    def __post_init__(self):
        self.xGD = self.xG - self.xGa
        # caller can compute distributions if needed
        self.proba_goals_scored = None
        self.proba_goals_conceded = None

    def __repr__(self) -> str:
        return f"{self.name}"

    def __hash__(self):
        return hash(self.name)

@dataclass
class MatchResult:
    team_1: Team
    team_2: Team
    goals_team_1: int
    goals_team_2: int
    players_scored_team_1: Dict[Player, int]
    players_scored_team_2: Dict[Player, int]
    winner: Optional[Team] = field(init=False)

    def __post_init__(self):
        if self.goals_team_1 > self.goals_team_2:
            self.winner = self.team_1
        elif self.goals_team_1 < self.goals_team_2:
            self.winner = self.team_2
        else:
            self.winner = None

    def __repr__(self) -> str:
        return f"{self.team_1} {self.goals_team_1} - {self.goals_team_2} {self.team_2}"

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
if __name__ == "__main__":
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

    # Teams
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