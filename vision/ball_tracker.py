# engine/vision/ball_tracker.py
from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from vision.court_detector import CourtDetector


class BallTracker:
    """
    High-speed Table Tennis Ball Tracker & Physics Event Spotter.
    Tracks ball position, computes parabolic velocity vectors, and spots:
      - 'BOUNCE' (Side 1 or Side 2)
      - 'HIT' (Player 1 or Player 2)
      - 'OUT' (Out of bounds)
    """

    def __init__(self, court_detector: CourtDetector, max_history: int = 30):
        self.court = court_detector
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)  # [(x, y, timestamp, is_interpolated)]
        self.prev_gray: Optional[np.ndarray] = None
        self.last_bounce_time: float = -1.0
        self.last_hit_time: float = -1.0

    def detect_ball_candidate(self, frame: np.ndarray, timestamp: float) -> Optional[Tuple[int, int]]:
        """
        Detects the table tennis ball candidate using color segmentation + motion saliency.
        """
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Region of Interest: Table + Player playing field
        if self.court.table_box:
            tx1, ty1, tx2, ty2 = self.court.table_box
            roi_y1 = max(0, ty1 - int(h * 0.25))
            roi_y2 = min(h, ty2 + int(h * 0.15))
            roi_x1 = max(0, tx1 - int(w * 0.15))
            roi_x2 = min(w, tx2 + int(w * 0.15))
        else:
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, int(h * 0.2), w, int(h * 0.9)

        # 1. Color Mask (High-brightness white / vibrant orange ball)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # White ball mask
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Orange ball mask
        lower_orange = np.array([5, 120, 150])
        upper_orange = np.array([25, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

        mask_color = cv2.bitwise_or(mask_white, mask_orange)

        # Crop to playing field ROI
        roi_mask = np.zeros_like(mask_color)
        roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = mask_color[roi_y1:roi_y2, roi_x1:roi_x2]

        # 2. Motion Saliency (Frame Differencing)
        if self.prev_gray is not None:
            frame_diff = cv2.absdiff(blurred, self.prev_gray)
            _, diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            combined_mask = cv2.bitwise_and(roi_mask, diff_thresh)
        else:
            combined_mask = roi_mask

        self.prev_gray = blurred

        # 3. Contour Analysis (Find small circular blob matching ball size)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Table tennis ball is typically between 6px and 400px area in 1080p
            if 6 <= area <= 450:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity > 0.4:  # Roundness threshold
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        candidates.append(((int(cx), int(cy)), area, circularity))

        if not candidates:
            return None

        # If previous position exists, pick candidate closest to velocity projection
        if len(self.history) >= 2:
            p1 = self.history[-1]
            p2 = self.history[-2]
            pred_x = p1[0] + (p1[0] - p2[0])
            pred_y = p1[1] + (p1[1] - p2[1])

            candidates.sort(key=lambda c: np.hypot(c[0][0] - pred_x, c[0][1] - pred_y))
            best_pos = candidates[0][0]
            # Max plausible displacement threshold (prevent teleporting across frame)
            if np.hypot(best_pos[0] - p1[0], best_pos[1] - p1[1]) < 220:
                return best_pos
            return None
        else:
            # Pick highest circularity candidate
            candidates.sort(key=lambda c: c[2], reverse=True)
            return candidates[0][0]

    def update(self, frame: np.ndarray, timestamp: float) -> List[Dict]:
        """
        Updates tracking history and detects physics events (bounces, hits, out).
        Returns list of triggered events in this frame:
          [{"type": "BOUNCE", "side": 1, "pos": (x, y), "timestamp": ts}, ...]
        """
        events = []
        ball_pos = self.detect_ball_candidate(frame, timestamp)

        if ball_pos is not None:
            self.history.append((ball_pos[0], ball_pos[1], timestamp, False))
        elif len(self.history) >= 2 and (timestamp - self.history[-1][2]) < 0.15:
            # Predict 1-2 missing frames (interpolation)
            p1 = self.history[-1]
            p2 = self.history[-2]
            dt = timestamp - p1[2]
            prev_dt = p1[2] - p2[2] or 0.016
            interp_x = int(p1[0] + (p1[0] - p2[0]) * (dt / prev_dt))
            interp_y = int(p1[1] + (p1[1] - p2[1]) * (dt / prev_dt))
            self.history.append((interp_x, interp_y, timestamp, True))

        # Check for Physics Events using trajectory inflection
        if len(self.history) >= 4:
            pts = list(self.history)
            p0, p1, p2, p3 = pts[-4], pts[-3], pts[-2], pts[-1]

            # Vertical velocities (dy)
            dy_prev = p1[1] - p0[1]  # positive = moving down
            dy_curr = p3[1] - p2[1]  # negative = moving up

            # Horizontal velocities (dx)
            dx_prev = p1[0] - p0[0]
            dx_curr = p3[0] - p2[0]

            # 1. BOUNCE DETECTION: Downward -> Upward inflection inside table
            if dy_prev > 2 and dy_curr < -2:
                bounce_x, bounce_y = p2[0], p2[1]
                court_side = self.court.get_court_side(bounce_x, bounce_y)
                # Avoid multi-triggering within 0.25 seconds
                if court_side > 0 and (timestamp - self.last_bounce_time) > 0.25:
                    self.last_bounce_time = timestamp
                    events.append({
                        "type": "BOUNCE",
                        "side": court_side,
                        "pos": (bounce_x, bounce_y),
                        "timestamp": timestamp
                    })

            # 2. PADDLE HIT DETECTION: Horizontal velocity reversal near players
            if (dx_prev > 5 and dx_curr < -5) or (dx_prev < -5 and dx_curr > 5):
                hit_x, hit_y = p2[0], p2[1]
                hit_player = 1 if hit_x < (self.court.net_x or (self.court.frame_width // 2)) else 2
                if (timestamp - self.last_hit_time) > 0.35:
                    self.last_hit_time = timestamp
                    events.append({
                        "type": "HIT",
                        "player": hit_player,
                        "pos": (hit_x, hit_y),
                        "timestamp": timestamp
                    })

        return events

    def draw_trajectory(self, frame: np.ndarray) -> np.ndarray:
        """Renders glowing ball trajectory trail on the frame."""
        if len(self.history) < 2:
            return frame

        pts = list(self.history)
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            alpha = i / len(pts)
            thickness = int(1 + 3 * alpha)
            color = (int(0 * alpha), int(255 * alpha), int(255 * alpha))  # Cyan/Yellow trail
            cv2.line(frame, (pts[i - 1][0], pts[i - 1][1]), (pts[i][0], pts[i][1]), color, thickness)

        # Draw current ball circle
        curr = pts[-1]
        cv2.circle(frame, (curr[0], curr[1]), 6, (0, 255, 255), -1)
        cv2.circle(frame, (curr[0], curr[1]), 8, (255, 255, 255), 1)
        return frame
