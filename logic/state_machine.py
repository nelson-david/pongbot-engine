# engine/logic/state_machine.py
from enum import Enum
from typing import Dict, List, Optional, Tuple


class GameState(Enum):
    WAITING_FOR_SERVE = 1
    RALLY_IN_PROGRESS = 2
    POINT_SCORED = 3
    GAME_FINISHED = 4


class MatchScoreEngine:
    """
    Complete Table Tennis Scoring Engine & Physics State Machine.
    
    Implements official ITTF rules:
      - Point awarded on double bounce, out of bounds, or unreturned hit.
      - Service alternates every 2 points (or every 1 point in deuce >= 10-10).
      - Games won at >= 11 points with a 2-point lead.
      - Automatic rally tracking and vital highlight clip capture.
    """

    def __init__(self, p1_name: str = "Player 1", p2_name: str = "Player 2", initial_server: int = 1):
        self.p1_name = p1_name
        self.p2_name = p2_name

        # Match Scores
        self.p1_pts = 0
        self.p2_pts = 0
        self.p1_games = 0
        self.p2_games = 0

        # Server: 1 for Player 1, 2 for Player 2
        self.current_server = initial_server

        # Rally & State Tracking
        self.state = GameState.WAITING_FOR_SERVE
        self.active_rally_start: float = 0.0
        self.last_event_time: float = 0.0
        self.last_hit_player: Optional[int] = None
        self.last_bounce_side: Optional[int] = None
        self.bounce_count_on_side: int = 0
        self.exchange_count: int = 0

        # Vital highlight clips: [(start_time, end_time, rally_desc)]
        self.vital_clips: List[Tuple[float, float, str]] = []
        self.event_log: List[Dict] = []

    def update_physics_frame(self, timestamp: float, physics_events: List[Dict]):
        """
        Processes vision-detected physics events for the current video frame.
        """
        for event in physics_events:
            ev_type = event.get("type")

            # ---------------------------------------------------------
            # 1. BOUNCE EVENT (Ball bounced on Side 1 or Side 2 of table)
            # ---------------------------------------------------------
            if ev_type == "BOUNCE":
                side = event.get("side")
                self.last_event_time = timestamp

                # If idle or waiting, a bounce indicates rally has begun
                if self.state in [GameState.WAITING_FOR_SERVE, GameState.POINT_SCORED]:
                    self.state = GameState.RALLY_IN_PROGRESS
                    self.active_rally_start = max(0.0, timestamp - 1.0)
                    self.last_hit_player = self.current_server
                    self.last_bounce_side = side
                    self.bounce_count_on_side = 1
                    self.exchange_count = 1
                    self._log_event("RALLY_START", timestamp, f"Rally started. Server P{self.current_server}")

                elif self.state == GameState.RALLY_IN_PROGRESS:
                    if side == self.last_bounce_side:
                        # Ball bounced twice consecutively on the same side -> DOUBLE BOUNCE!
                        self.bounce_count_on_side += 1
                        if self.bounce_count_on_side >= 2:
                            # The player defending this side failed to return it; striker wins point
                            scoring_player = 1 if side == 2 else 2
                            self._conclude_point(scoring_player, timestamp, reason=f"Double bounce on Side {side}")
                    else:
                        # Ball crossed net and bounced on opponent's half (Legitimate return)
                        self.last_bounce_side = side
                        self.bounce_count_on_side = 1
                        self.exchange_count += 1
                        self._log_event("BOUNCE", timestamp, f"Bounce on Side {side} (Exchange {self.exchange_count})")

            # ---------------------------------------------------------
            # 2. PADDLE HIT EVENT (Player 1 or Player 2 struck ball)
            # ---------------------------------------------------------
            elif ev_type == "HIT":
                player = event.get("player")
                self.last_event_time = timestamp

                if self.state == GameState.RALLY_IN_PROGRESS:
                    self.last_hit_player = player
                    self.bounce_count_on_side = 0
                    self._log_event("HIT", timestamp, f"Player {player} hit ball")

            # ---------------------------------------------------------
            # 3. OUT OF BOUNDS / FAULT EVENT
            # ---------------------------------------------------------
            elif ev_type == "OUT":
                if self.state == GameState.RALLY_IN_PROGRESS:
                    # Point goes to the opponent of the player who hit it out
                    scoring_player = 2 if self.last_hit_player == 1 else 1
                    self._conclude_point(scoring_player, timestamp, reason="Out of bounds")

        # ---------------------------------------------------------
        # 4. TIMEOUT / RALLY CONCLUSION CHECK
        # If in active rally and no ball action for > 1.8 seconds, point has concluded
        # ---------------------------------------------------------
        if self.state == GameState.RALLY_IN_PROGRESS and (timestamp - self.last_event_time) > 1.8:
            # If the ball bounced on opponent's side and was never returned, last hitter wins
            if self.last_bounce_side is not None:
                scoring_player = 1 if self.last_bounce_side == 2 else 2
                self._conclude_point(scoring_player, timestamp, reason="Rally concluded (unreturned ball)")
            else:
                self.state = GameState.WAITING_FOR_SERVE

    def _conclude_point(self, scoring_player: int, timestamp: float, reason: str = ""):
        """Awards point, updates games/sets, and rotates service."""
        self.state = GameState.POINT_SCORED
        active_rally_end = timestamp + 0.8
        rally_duration = active_rally_end - self.active_rally_start

        # Record vital highlight clip if rally duration is significant
        if rally_duration >= 3.0 or self.exchange_count >= 3:
            desc = f"{self.p1_name if scoring_player == 1 else self.p2_name} won ({self.exchange_count} exchanges, {rally_duration:.1f}s)"
            self.vital_clips.append((self.active_rally_start, active_rally_end, desc))

        # Award point
        if scoring_player == 1:
            self.p1_pts += 1
        else:
            self.p2_pts += 1

        self._log_event("POINT_AWARDED", timestamp, f"Point awarded to P{scoring_player} ({reason}). Score: {self.p1_pts}-{self.p2_pts}")

        # Check for Game / Set Won (Official ITTF Rule: >= 11 points and >= 2 point lead)
        if (self.p1_pts >= 11 or self.p2_pts >= 11) and abs(self.p1_pts - self.p2_pts) >= 2:
            game_winner = 1 if self.p1_pts > self.p2_pts else 2
            if game_winner == 1:
                self.p1_games += 1
            else:
                self.p2_games += 1

            self._log_event("GAME_WON", timestamp, f"Game won by P{game_winner}! Sets: {self.p1_games}-{self.p2_games}")
            self.p1_pts = 0
            self.p2_pts = 0

        # Rotate Service according to ITTF rules:
        # Standard: service changes every 2 points
        # Deuce (>= 10-10): service changes every 1 point
        total_pts = self.p1_pts + self.p2_pts
        is_deuce = (self.p1_pts >= 10 and self.p2_pts >= 10)
        
        if is_deuce:
            self.current_server = 2 if self.current_server == 1 else 1
        elif total_pts % 2 == 0 and total_pts > 0:
            self.current_server = 2 if self.current_server == 1 else 1

    def _log_event(self, event_name: str, timestamp: float, details: str):
        self.event_log.append({
            "event": event_name,
            "timestamp": timestamp,
            "details": details,
            "score": f"{self.p1_pts}-{self.p2_pts}",
            "server": self.current_server
        })

    def get_score_state(self) -> Dict:
        """Returns current scorecard dictionary for BroadcastRenderer."""
        return {
            "p1_name": self.p1_name,
            "p2_name": self.p2_name,
            "p1_pts": self.p1_pts,
            "p2_pts": self.p2_pts,
            "p1_games": self.p1_games,
            "p2_games": self.p2_games,
            "server": self.current_server,
            "state": self.state.name
        }