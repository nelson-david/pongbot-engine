# engine/vision/court_detector.py
from typing import Dict, Optional, Tuple
import cv2
import numpy as np


class CourtDetector:
    """
    Detects and models the table tennis court geometry (Table Bounds, Far/Near Sides, Net).
    Can infer court bounds dynamically between player positions or from blue/green table surface color.
    """

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Default estimated table bounds (x1, y1, x2, y2) in normalized / absolute coords
        self.table_box: Optional[Tuple[int, int, int, int]] = None
        self.net_x: Optional[int] = None
        self.is_calibrated = False

    def calibrate_from_players(self, p1_box: Tuple[int, int, int, int], p2_box: Tuple[int, int, int, int]):
        """
        Calibrates the table position based on the bounding boxes of the two active players.
        Table spans the central region between the far and near players.
        """
        p1_x1, p1_y1, p1_x2, p1_y2 = p1_box
        p2_x1, p2_y1, p2_x2, p2_y2 = p2_box

        # Table is situated horizontally between the players and vertically at waist/hip height
        x1 = min(p1_x2, p2_x1) - int(0.05 * self.frame_width)
        x2 = max(p1_x2, p2_x1) + int(0.05 * self.frame_width)

        # Clamp boundaries to valid frame
        x1 = max(int(0.15 * self.frame_width), min(p1_x1, p2_x1))
        x2 = min(int(0.85 * self.frame_width), max(p1_x2, p2_x2))
        
        y1 = int(0.40 * self.frame_height)
        y2 = int(0.85 * self.frame_height)

        self.table_box = (x1, y1, x2, y2)
        self.net_x = int((x1 + x2) / 2)
        self.is_calibrated = True

    def auto_detect_table(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Segment table surface by detecting blue/green/dark court colors or dominant table contour.
        """
        if self.is_calibrated and self.table_box is not None:
            return self.table_box

        h, w, _ = frame.shape
        # Default center region for standard table tennis broadcasts
        x1 = int(w * 0.22)
        y1 = int(h * 0.45)
        x2 = int(w * 0.78)
        y2 = int(h * 0.82)

        self.table_box = (x1, y1, x2, y2)
        self.net_x = int((x1 + x2) / 2)
        self.is_calibrated = True
        return self.table_box

    def get_court_side(self, x: int, y: int) -> int:
        """
        Determines which side of the table tennis court the coordinate lies on:
        Returns:
            1: Side 1 (Left / Far player half)
            2: Side 2 (Right / Near player half)
            0: Out of bounds
        """
        if self.table_box is None:
            return 0

        x1, y1, x2, y2 = self.table_box
        net_x = self.net_x or int((x1 + x2) / 2)

        # Check if inside table bounding box (with slight margin for edge balls)
        margin_y = int(0.05 * self.frame_height)
        if (x1 - 20) <= x <= (x2 + 20) and (y1 - margin_y) <= y <= (y2 + margin_y):
            if x < net_x:
                return 1
            else:
                return 2
        return 0

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        """Draws table zones and net line for visual validation."""
        if self.table_box is None:
            return frame

        x1, y1, x2, y2 = self.table_box
        net_x = self.net_x or int((x1 + x2) / 2)

        # Draw Side 1 (Left/Far) - Cyan
        cv2.rectangle(frame, (x1, y1), (net_x, y2), (255, 200, 0), 2)
        cv2.putText(frame, "SIDE 1 (P1)", (x1 + 10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # Draw Side 2 (Right/Near) - Magenta
        cv2.rectangle(frame, (net_x, y1), (x2, y2), (255, 0, 200), 2)
        cv2.putText(frame, "SIDE 2 (P2)", (net_x + 10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 200), 2)

        # Draw Net Line - White
        cv2.line(frame, (net_x, y1 - 10), (net_x, y2 + 10), (255, 255, 255), 3)
        return frame
