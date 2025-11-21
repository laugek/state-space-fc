
import gymnasium as gym

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

