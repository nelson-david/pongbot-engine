# engine/vision/tracker.py
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from vision.court_detector import CourtDetector
from vision.ball_tracker import BallTracker


class MultiModelTracker:
    """
    Unified Sports Vision Subsystem:
      1. Player Identification & Bounding Box Tracking (YOLOv8)
      2. Court & Table Geometry Calibration (CourtDetector)
      3. Ball Trajectory & Physics Event Spotting (BallTracker)
    """

    def __init__(self, yolo_model_path: str = "yolov8n.pt", frame_width: int = 1920, frame_height: int = 1080):
        self.model = YOLO(yolo_model_path)
        self.court = CourtDetector(frame_width, frame_height)
        self.ball_tracker = BallTracker(self.court)
        self.players: List[Dict] = []
        self.is_calibrated = False

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[List[Dict], List[Dict]]:
        """
        Runs detection on incoming video frame.
        Returns:
            (players, physics_events)
        """
        h, w, _ = frame.shape

        # 1. Player Detection with YOLOv8
        results = self.model(frame, verbose=False)[0]
        detected_persons = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            conf = float(box.conf[0])
            if label == "person" and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                center_x = (x1 + x2) / 2
                detected_persons.append({
                    "box": (x1, y1, x2, y2),
                    "center_x": center_x,
                    "conf": conf
                })

        # Sort horizontally: Left (P1) to Right (P2)
        detected_persons.sort(key=lambda p: p["center_x"])
        self.players = detected_persons

        # 2. Calibrate Court when 2 players are visible
        if not self.is_calibrated and len(self.players) >= 2:
            self.court.calibrate_from_players(self.players[0]["box"], self.players[1]["box"])
            self.is_calibrated = True
        elif not self.is_calibrated:
            self.court.auto_detect_table(frame)

        # 3. Ball Tracking and Physics Event Spotting
        physics_events = self.ball_tracker.update(frame, timestamp)

        return self.players, physics_events

    def draw_overlays(self, frame: np.ndarray, p1_name: str = "Player 1", p2_name: str = "Player 2", show_debug: bool = False) -> np.ndarray:
        """Draws player bounding boxes and ball trajectory trails."""
        # 1. Draw Player Boxes
        for p in self.players:
            x1, y1, x2, y2 = p["box"]
            player_label = p1_name if p["center_x"] < (self.court.frame_width / 2) else p2_name
            color = (0, 255, 128) if player_label == p1_name else (255, 128, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, player_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 2. Draw Ball Trail
        frame = self.ball_tracker.draw_trajectory(frame)

        # 3. Optional Court Boundary Debug
        if show_debug:
            frame = self.court.draw_debug(frame)

        return frame
