# engine/vision/tracker.py
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from vision.court_detector import CourtDetector
from vision.ball_tracker import BallTracker


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _box_area(box: Tuple[int, int, int, int]) -> int:
    """Returns pixel area of a bounding box (x1, y1, x2, y2)."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


# ---------------------------------------------------------------------------
# Shirt Color & Appearance Profiling
# ---------------------------------------------------------------------------

def extract_shirt_features(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
) -> Optional[Dict]:
    """
    Extracts high-fidelity shirt color features from the upper-body region of a
    person detection.

    ROI: central 56% width x 15-48% height of the bounding box (chest area).
    Features:
      - 2D HSV histogram (Hue x Saturation)
      - Mean CIE Lab color (perceptually uniform lighting/color representation)
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 16:
        return None

    # Crop to shirt/chest area, excluding head (top 15%) and waist/legs (below 48%)
    sx1 = x1 + int(bw * 0.22)
    sx2 = x2 - int(bw * 0.22)
    sy1 = y1 + int(bh * 0.15)
    sy2 = y1 + int(bh * 0.48)

    h, w = frame.shape[:2]
    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(w, sx2), min(h, sy2)

    if sx2 - sx1 < 4 or sy2 - sy1 < 4:
        return None

    crop = frame[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return None

    # HSV 2D histogram
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    # CIE Lab mean color
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    mean_lab = lab.mean(axis=(0, 1))

    return {
        "hist": hist,
        "lab": mean_lab
    }


def compare_appearance(feat_a: Optional[Dict], feat_b: Optional[Dict]) -> float:
    """
    Returns appearance similarity in range [0, 1].
    Combines HSV histogram correlation and CIE Lab color distance.
    """
    if feat_a is None or feat_b is None:
        return -1.0

    # 1. Histogram correlation (range -1 to 1, clamped to >= 0)
    h_sim = max(0.0, float(cv2.compareHist(feat_a["hist"], feat_b["hist"], cv2.HISTCMP_CORREL)))

    # 2. CIE Lab euclidean distance (normalized by typical garment color variance ~80)
    lab_dist = float(np.linalg.norm(feat_a["lab"] - feat_b["lab"]))
    lab_sim = max(0.0, 1.0 - (lab_dist / 80.0))

    # Balanced combined score
    return 0.5 * h_sim + 0.5 * lab_sim


# ---------------------------------------------------------------------------
# Initial candidate filtering (used before appearance profiles are captured)
# ---------------------------------------------------------------------------

def filter_players(
    detected_persons: List[Dict],
    frame_height: int = 1080,
    frame_width: int = 1920,
    max_players: int = 2,
    min_aspect_ratio: float = 0.85,
) -> List[Dict]:
    """
    Filters YOLO person detections down to the 2 active court players.
    Eliminates umpires sitting high up in the background and spectators.
    """
    valid = []
    for p in detected_persons:
        x1, y1, x2, y2 = p["box"]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        aspect = bh / bw

        # Drop background umpires sitting in high chairs in the upper half of screen
        if y2 <= 0.48 * frame_height and bw < 0.30 * frame_width:
            continue

        p["area"] = bw * bh
        p["aspect"] = aspect
        if aspect >= min_aspect_ratio:
            valid.append(p)

    pool = valid if len(valid) >= max_players else detected_persons
    pool.sort(key=lambda p: p.get("area", _box_area(p["box"])), reverse=True)
    return pool[:max_players]


# ---------------------------------------------------------------------------
# Main MultiModelTracker
# ---------------------------------------------------------------------------

class MultiModelTracker:
    """
    Unified Sports Vision Subsystem with appearance-locked player identity.

    Key guarantees:
      1. Player 1 (Left / Server 1) and Player 2 (Right / Server 2) each have a
         locked shirt color appearance profile.
      2. If only one player is visible (e.g. camera close-up or replay angle),
         the player is matched to their correct profile and retains their true identity.
         Player 2 is NEVER labeled as Player 1 just because he is the sole person on screen.
      3. Ball boys, seated umpires, and spectators are rejected because they do
         not match either player's appearance profile.
      4. Off-screen players are not drawn as phantom boxes.
    """

    # Similarity threshold for matching a detection to a known player profile
    APPEARANCE_MATCH_THRESHOLD = 0.34

    def __init__(self, yolo_model_path: str = "yolov8n.pt", frame_width: int = 1920, frame_height: int = 1080):
        self.model = YOLO(yolo_model_path)
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.court = CourtDetector(frame_width, frame_height)
        self.ball_tracker = BallTracker(self.court)

        # Active player detections for current frame (0, 1, or 2 entries)
        self.players: List[Dict] = []
        self.is_calibrated = False

        # Appearance profiles: [p1_feat_dict, p2_feat_dict]
        self._profiles: Optional[List[Optional[Dict]]] = None
        self._profile_frames: int = 0
        self._profile_refresh_interval: int = 180

        # Warmup counter if profiles not yet locked
        self._warmup_frames: int = 0
        self._warmup_target: int = 6

    def lock_player_profiles(self, frame: np.ndarray, p1_box: Tuple[int, int, int, int], p2_box: Tuple[int, int, int, int]):
        """Explicitly sets and locks the appearance profiles of Player 1 and Player 2."""
        f1 = extract_shirt_features(frame, p1_box)
        f2 = extract_shirt_features(frame, p2_box)
        if f1 is not None and f2 is not None:
            self._profiles = [f1, f2]
            self._profile_frames = 0
            # Also calibrate court table position
            self.court.calibrate_from_players(p1_box, p2_box)
            self.is_calibrated = True

    def _match_by_appearance(
        self,
        frame: np.ndarray,
        candidates: List[Dict],
    ) -> List[Dict]:
        """
        Matches detected persons to Player 1 (slot 0) and Player 2 (slot 1) profiles.
        Returns a list of actually detected players, each tagged with its assigned 'slot'.
        Non-matching persons (ball boys, spectators, referees) are rejected.
        """
        if self._profiles is None:
            return candidates[:2]

        h, w = frame.shape[:2]
        scored_pairs = []

        for ci, cand in enumerate(candidates):
            box = cand["box"]
            # Fast filter: drop background persons sitting high up
            if box[3] <= 0.48 * h and (box[2] - box[0]) < 0.30 * w:
                continue

            feat = extract_shirt_features(frame, box)
            if feat is None:
                continue

            for slot in range(2):
                sim = compare_appearance(feat, self._profiles[slot])
                scored_pairs.append((ci, slot, sim))

        # Sort highest similarity first
        scored_pairs.sort(key=lambda item: item[2], reverse=True)

        assigned_slots: Dict[int, Dict] = {}
        used_candidates = set()

        for ci, slot, sim in scored_pairs:
            if slot in assigned_slots or ci in used_candidates:
                continue
            if sim >= self.APPEARANCE_MATCH_THRESHOLD:
                cand_copy = dict(candidates[ci])
                cand_copy["slot"] = slot           # 0 for P1, 1 for P2
                cand_copy["sim"] = sim
                assigned_slots[slot] = cand_copy
                used_candidates.add(ci)

        # Return only the players actually detected in this frame (no phantom boxes)
        matched_players = []
        for slot in (0, 1):
            if slot in assigned_slots:
                matched_players.append(assigned_slots[slot])

        return matched_players

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[List[Dict], List[Dict]]:
        """
        Runs detection and tracking on incoming video frame.
        Returns:
            (players, physics_events)
        """
        h, w, _ = frame.shape

        # 1. Person Detection with YOLOv8
        results = self.model(frame, verbose=False)[0]
        detected_persons = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            conf = float(box.conf[0])
            if label == "person" and conf > 0.38:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                center_x = (x1 + x2) / 2
                detected_persons.append({
                    "box": (x1, y1, x2, y2),
                    "center_x": center_x,
                    "conf": conf
                })

        # 2. Player Identification
        if self._profiles is not None:
            # ── APPEARANCE MODE ──
            filtered = self._match_by_appearance(frame, detected_persons)

            # Periodically adapt profiles if both players are cleanly visible
            self._profile_frames += 1
            if self._profile_frames >= self._profile_refresh_interval and len(filtered) == 2:
                p1_cand = next((p for p in filtered if p["slot"] == 0), None)
                p2_cand = next((p for p in filtered if p["slot"] == 1), None)
                if p1_cand and p2_cand:
                    f1 = extract_shirt_features(frame, p1_cand["box"])
                    f2 = extract_shirt_features(frame, p2_cand["box"])
                    if f1 and f2:
                        self._profiles = [f1, f2]
                        self._profile_frames = 0
        else:
            # ── WARMUP MODE (before profiles are captured) ──
            filtered = filter_players(detected_persons, frame_height=h, frame_width=w, max_players=2)
            filtered.sort(key=lambda p: p["center_x"])
            for idx, p in enumerate(filtered):
                p["slot"] = idx

            if len(filtered) == 2:
                self._warmup_frames += 1
                if self._warmup_frames >= self._warmup_target:
                    self.lock_player_profiles(frame, filtered[0]["box"], filtered[1]["box"])
            else:
                self._warmup_frames = 0

        self.players = filtered

        # 3. Court Calibration (once 2 players are established)
        if not self.is_calibrated and len(self.players) == 2:
            p1_b = next((p["box"] for p in self.players if p.get("slot") == 0), self.players[0]["box"])
            p2_b = next((p["box"] for p in self.players if p.get("slot") == 1), self.players[1]["box"])
            self.court.calibrate_from_players(p1_b, p2_b)
            self.is_calibrated = True
        elif not self.is_calibrated:
            self.court.auto_detect_table(frame)

        # 4. Ball Tracking (only mask actual players on screen)
        player_boxes = [p["box"] for p in self.players]
        physics_events = self.ball_tracker.update(frame, player_boxes, timestamp)

        return self.players, physics_events

    def draw_overlays(
        self,
        frame: np.ndarray,
        p1_name: str = "Player 1",
        p2_name: str = "Player 2",
        show_debug: bool = False
    ) -> np.ndarray:
        """
        Draws player bounding boxes and labels tied directly to their locked slot.
        Player 2 is ALWAYS labeled as Player 2 regardless of whether Player 1 is on screen.
        """
        names = {0: p1_name, 1: p2_name}
        colors = {0: (0, 255, 128), 1: (255, 128, 0)}  # Green for P1, Orange for P2

        for p in self.players:
            slot = p.get("slot")
            if slot is None:
                continue
            x1, y1, x2, y2 = p["box"]
            label = names.get(slot, f"Player {slot+1}")
            color = colors.get(slot, (200, 200, 200))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 2. Draw Ball Trail
        frame = self.ball_tracker.draw_trajectory(frame)

        # 3. Optional Court Boundary Debug
        if show_debug:
            frame = self.court.draw_debug(frame)

        return frame
