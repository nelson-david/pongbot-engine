# engine/logic/state_machine.py
from enum import Enum
from typing import Dict, List, Tuple

class GameState(Enum):
    IDLE = 0
    SERVE_WAIT = 1
    RALLY_IN_PROGRESS = 2
    POINT_SCORED = 3

class MatchScoreEngine:
    def __init__(self, player1_name: str = "Player 1", player2_name: str = "Player 2"):
        self.state = GameState.IDLE
        self.score = {player1_name: 0, player2_name: 0}
        self.current_server = player1_name
        self.rally_events: List[Dict] = []
        self.active_rally_start: float = 0.0
        self.vital_clips: List[Tuple[float, float]] = []  # [(start_time, end_time)]

    def update_physics_frame(self, timestamp: float, ball_pos: Tuple[int, int], event_type: str = None):
        """
        Processes frame-level physics events (e.g., 'BOUNCE', 'NET_HIT', 'OUT_OF_BOUNDS')
        """
        if self.state == GameState.IDLE or self.state == GameState.POINT_SCORED:
            if event_type == "SERVE":
                self.state = GameState.RALLY_IN_PROGRESS
                self.active_rally_start = max(0.0, timestamp - 1.5)  # Include 1.5s pre-serve context

        elif self.state == GameState.RALLY_IN_PROGRESS:
            if event_type in ["FAULT", "OUT_OF_BOUNDS", "FLOOR_BOUNCE"]:
                self.state = GameState.POINT_SCORED
                active_rally_end = timestamp + 1.0  # Include 1.0s post-point celebration context
                
                # Flag as a vital highlight clip if rally duration exceeds 3.5 seconds
                if (active_rally_end - self.active_rally_start) >= 3.5:
                    self.vital_clips.append((self.active_rally_start, active_rally_end))
                
                # Award point logic (Example placeholder rule)
                self.award_point(scoring_player="Player 1")

    def award_point(self, scoring_player: str):
        self.score[scoring_player] += 1