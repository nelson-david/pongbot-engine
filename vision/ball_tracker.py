# engine/vision/ball_tracker.py
from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from vision.court_detector import CourtDetector


class BallTracker:
    """
    Robust Table Tennis Ball Tracker & Physics Event Spotter.
    Uses Player-Masked MOG2 Background Subtraction, Motion Saliency, and Trajectory Inflection.

    Key design choices to prevent MOG2 adaptation killing ball detection:
      - Learning rate is capped at 0.002 (vs default ~0.02) so the background
        model adapts slowly and never "learns away" the fast-moving ball.
      - The MOG2 model is periodically re-initialised every `reset_interval`
        frames (default 300 ≈ 10s @30fps) so even if it starts suppressing
        ball motion, the reset restores sensitivity.
      - A 15-frame warmup grace period after each reset lets the model build a
        clean background before we trust its foreground mask.
    """

    # --- Tuning knobs ---
    MOG2_HISTORY = 120          # Longer history → slower learning baseline
    MOG2_VAR_THRESHOLD = 25     # Slightly raised → more sensitive to small objects
    MOG2_LEARNING_RATE = 0.002  # Very slow adaptation so ball stays "foreground"
    RESET_INTERVAL = 300        # Re-init MOG2 every N frames (~10s @30fps)
    RESET_WARMUP = 15           # Frames to skip after reset while BG stabilises

    # --- Bounce / Hit debounce ---
    MIN_BOUNCE_GAP = 0.30       # Min seconds between two BOUNCE events
    MIN_HIT_GAP = 0.35          # Min seconds between two HIT events

    # --- Trajectory inflection thresholds ---
    BOUNCE_DY_THRESH = 2        # Min vertical px delta for a bounce inflection
    HIT_DX_THRESH = 5           # Min horizontal px delta for a hit reversal

    def __init__(self, court_detector: CourtDetector, max_history: int = 40):
        self.court = court_detector
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)  # [(x, y, timestamp, is_interpolated)]

        # MOG2 background subtractor with slow-adapt settings
        self.fgbg = self._create_mog2()
        self._frame_since_reset: int = 0   # Frames since last MOG2 reset

        self.last_bounce_time: float = -1.0
        self.last_hit_time: float = -1.0

    # ------------------------------------------------------------------
    # MOG2 lifecycle
    # ------------------------------------------------------------------

    def _create_mog2(self):
        """Factory for a fresh MOG2 subtractor with our slow-adapt settings."""
        return cv2.createBackgroundSubtractorMOG2(
            history=self.MOG2_HISTORY,
            varThreshold=self.MOG2_VAR_THRESHOLD,
            detectShadows=False,
        )

    def _maybe_reset_mog2(self):
        """Re-initialise the background model periodically so it never fully adapts."""
        self._frame_since_reset += 1
        if self._frame_since_reset >= self.RESET_INTERVAL:
            self.fgbg = self._create_mog2()
            self._frame_since_reset = 0

    # ------------------------------------------------------------------
    # Ball detection
    # ------------------------------------------------------------------

    def detect_ball_candidate(
        self,
        frame: np.ndarray,
        player_boxes: List[Tuple[int, int, int, int]],
        timestamp: float
    ) -> Optional[Tuple[int, int]]:
        """
        Detects ball candidate by applying background subtraction inside the court with players masked out.
        """
        h, w, _ = frame.shape

        # Periodic MOG2 reset to prevent full adaptation
        self._maybe_reset_mog2()

        # 1. Generate foreground motion mask (slow learning rate keeps ball visible)
        fgmask = self.fgbg.apply(frame, learningRate=self.MOG2_LEARNING_RATE)

        # Skip detection during post-reset warmup (BG model is noisy)
        if self._frame_since_reset <= self.RESET_WARMUP:
            return None

        # 2. Mask out player bodies to isolate the small ball
        for box in player_boxes:
            px1, py1, px2, py2 = box
            # Expand player box slightly to cover limb swings
            px1 = max(0, px1 - 15)
            py1 = max(0, py1 - 15)
            px2 = min(w, px2 + 15)
            py2 = min(h, py2 + 15)
            fgmask[py1:py2, px1:px2] = 0

        # 3. Crop motion search to the playing field ROI
        if self.court.table_box:
            tx1, ty1, tx2, ty2 = self.court.table_box
            roi_y1 = max(0, ty1 - int(h * 0.30))
            roi_y2 = min(h, ty2 + int(h * 0.15))
            roi_x1 = max(0, tx1 - int(w * 0.15))
            roi_x2 = min(w, tx2 + int(w * 0.15))
        else:
            roi_x1, roi_y1, roi_x2, roi_y2 = int(w * 0.15), int(h * 0.30), int(w * 0.85), int(h * 0.85)

        roi_mask = np.zeros_like(fgmask)
        roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = fgmask[roi_y1:roi_y2, roi_x1:roi_x2]

        # 4. Clean up noise with morphological open
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)

        # 5. Extract contours and filter by ball area and circularity
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 4 <= area <= 600:
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                candidates.append(((int(cx), int(cy)), area, circularity))

        if not candidates:
            return None

        # If previous trajectory exists, pick candidate closest to velocity projection
        if len(self.history) >= 2:
            p1 = self.history[-1]
            p2 = self.history[-2]
            pred_x = p1[0] + (p1[0] - p2[0])
            pred_y = p1[1] + (p1[1] - p2[1])

            candidates.sort(key=lambda c: np.hypot(c[0][0] - pred_x, c[0][1] - pred_y))
            best_pos = candidates[0][0]
            # Max plausible displacement threshold (prevent jumping across frame)
            if np.hypot(best_pos[0] - p1[0], best_pos[1] - p1[1]) < 250:
                return best_pos
            return None
        else:
            # Pick candidate with best circularity
            candidates.sort(key=lambda c: c[2], reverse=True)
            return candidates[0][0]

    def update(
        self,
        frame: np.ndarray,
        player_boxes: List[Tuple[int, int, int, int]],
        timestamp: float
    ) -> List[Dict]:
        """
        Updates tracking history and detects physics events (bounces, hits, out).
        """
        events = []
        ball_pos = self.detect_ball_candidate(frame, player_boxes, timestamp)

        if ball_pos is not None:
            self.history.append((ball_pos[0], ball_pos[1], timestamp, False))
        elif len(self.history) >= 2 and (timestamp - self.history[-1][2]) < 0.15:
            # Predict short occlusion (1-2 frames)
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
            if dy_prev > self.BOUNCE_DY_THRESH and dy_curr < -self.BOUNCE_DY_THRESH:
                bounce_x, bounce_y = p2[0], p2[1]
                court_side = self.court.get_court_side(bounce_x, bounce_y)
                if court_side > 0 and (timestamp - self.last_bounce_time) > self.MIN_BOUNCE_GAP:
                    self.last_bounce_time = timestamp
                    events.append({
                        "type": "BOUNCE",
                        "side": court_side,
                        "pos": (bounce_x, bounce_y),
                        "timestamp": timestamp
                    })

            # 2. PADDLE HIT DETECTION: Horizontal velocity reversal near players
            if (dx_prev > self.HIT_DX_THRESH and dx_curr < -self.HIT_DX_THRESH) or \
               (dx_prev < -self.HIT_DX_THRESH and dx_curr > self.HIT_DX_THRESH):
                hit_x, hit_y = p2[0], p2[1]
                net_x = self.court.net_x or (self.court.frame_width // 2)
                hit_player = 1 if hit_x < net_x else 2
                if (timestamp - self.last_hit_time) > self.MIN_HIT_GAP:
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
            color = (int(0 * alpha), int(255 * alpha), int(255 * alpha))  # Yellow/Cyan trail
            cv2.line(frame, (pts[i - 1][0], pts[i - 1][1]), (pts[i][0], pts[i][1]), color, thickness)

        # Draw current ball circle
        curr = pts[-1]
        cv2.circle(frame, (curr[0], curr[1]), 6, (0, 255, 255), -1)
        cv2.circle(frame, (curr[0], curr[1]), 8, (255, 255, 255), 1)
        return frame
