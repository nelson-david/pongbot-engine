import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from renderer.video_processor import BroadcastRenderer, merge_audio
from renderer.country_selector import select_country
from enhancer.video_upscaler import RealESRGANEnhancer, enhance_broadcast_4k
from vision.tracker import MultiModelTracker, filter_players
from logic.state_machine import MatchScoreEngine


def setup_player_names(cap, tracker):
    """
    Reads the first valid frame, identifies player positions (filtering out
    umpires/spectators), locks their appearance profiles, and prompts the user in the CLI
    for player names and country selections using arrow keys.
    """
    ret, frame = cap.read()
    if not ret:
        return "Player 1", "NG", "Player 2", "NG"
    
    # Reset video capture back to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    h, w = frame.shape[:2]

    # Run inference on frame 0 to get initial person detections
    results = tracker.model(frame, verbose=False)[0]
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
                "conf": conf,
            })

    # Filter to actual court players only (excludes umpires in high chairs)
    players = filter_players(detected_persons, frame_height=h, frame_width=w, max_players=2)
    players.sort(key=lambda p: p["center_x"])

    n_raw = len(detected_persons)
    n_filtered = len(players)
    print(f"\n--- PLAYER IDENTIFICATION & COUNTRY SETUP ---")
    print(f"  Detected {n_raw} person(s) in frame, filtered to {n_filtered} player(s) (umpire/spectators excluded)")

    # Ask for Left/Far side player name & country
    if len(players) > 0:
        p1_input = input("Enter name for Player 1 [Left / Far Side]: ").strip()
        p1_name = p1_input if p1_input else "Player 1"
    else:
        p1_name = "Player 1"

    p1_country = select_country(f"Select country for {p1_name} [Left / Far Side]", default_code="NG")

    # Ask for Right/Near side player name & country
    if len(players) > 1:
        p2_input = input("Enter name for Player 2 [Right / Near Side]: ").strip()
        p2_name = p2_input if p2_input else "Player 2"
    else:
        p2_name = "Player 2"

    p2_country = select_country(f"Select country for {p2_name} [Right / Near Side]", default_code="NG")

    # Lock appearance profiles & calibrate court geometry immediately
    if len(players) >= 2:
        tracker.lock_player_profiles(frame, players[0]["box"], players[1]["box"])
        print(f"  [LOCKED] Player 1 ({p1_name} - {p1_country}) & Player 2 ({p2_name} - {p2_country}) shirt appearance locked.")

    print(f"Match configured: {p1_name} [{p1_country}] vs {p2_name} [{p2_country}]\n")
    return p1_name, p1_country, p2_name, p2_country

def run_visual_test(
    input_path: str = "test_clip_2.mov",
    output_path: str = "test_output.mp4",
    mode: str = "standard",  # "standard", "broadcast4k", "realesrgan"
    upscale_factor: int = 2,
    model_name: str = "realesrgan-x4plus",
):
    print("Initializing MultiModelTracker and MatchScoreEngine...")

    cap = cv2.VideoCapture(input_path)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize Core Vision & Physics Subsystems
    tracker = MultiModelTracker(yolo_model_path="yolov8n.pt", frame_width=width, frame_height=height)
    renderer = BroadcastRenderer()

    # Prompt for custom player names & country selections using arrow keys
    p1_name, p1_country, p2_name, p2_country = setup_player_names(cap, tracker)

    # Initialize Table Tennis Scoring Engine
    score_engine = MatchScoreEngine(p1_name=p1_name, p2_name=p2_name, initial_server=1)

    # Use avc1 (native H.264) for crystal clear intermediate frames without mp4v degradation
    temp_raw_path = "temp_raw_render.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_raw_path, fourcc, fps, (width, height))

    frame_count = 0
    ball_detect_count = 0
    total_bounce_events = 0
    total_hit_events = 0
    print(f"Processing '{input_path}' ({total_frames} total frames at {fps} fps)...")

    with tqdm(total=total_frames, desc="Rendering Engine", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = frame_count / fps

            # 1. Automated Vision Tracking (Players + Table + Ball Trajectory)
            players, physics_events = tracker.process_frame(frame, timestamp)

            # 2. Automated Physics & Rule Engine Scoring
            score_engine.update_physics_frame(timestamp, physics_events)
            score_state = score_engine.get_score_state()

            # Track ball detection rate
            if len(tracker.ball_tracker.history) > 0 and not tracker.ball_tracker.history[-1][3]:
                ball_detect_count += 1

            # 3. Log real detected game events
            for ev in physics_events:
                if ev.get("type") == "BOUNCE":
                    total_bounce_events += 1
                    tqdm.write(f"[BOUNCE] Side {ev.get('side')} at {timestamp:.2f}s | "
                               f"Score: {score_state['p1_name']} {score_state['p1_pts']}-{score_state['p2_pts']} {score_state['p2_name']}")
                elif ev.get("type") == "HIT":
                    total_hit_events += 1
                    tqdm.write(f"[HIT] Player {ev.get('player')} at {timestamp:.2f}s")

            # Periodic stats every 300 frames
            if frame_count % 300 == 0:
                det_rate = (ball_detect_count / frame_count) * 100 if frame_count > 0 else 0
                tqdm.write(f"[STATS] Frame {frame_count}: {len(players)} players tracked, "
                           f"ball detected {ball_detect_count}/{frame_count} frames ({det_rate:.0f}%), "
                           f"bounces={total_bounce_events}, hits={total_hit_events}")

            # 4. Draw Player Boxes, Ball Trails, and Bounding Boxes
            frame = tracker.draw_overlays(frame, p1_name=p1_name, p2_name=p2_name)

            # 5. Burn WTT Modern Compact Broadcast Scoreboard with National Flags
            frame = renderer.draw_scorecard(
                frame=frame,
                p1_name=score_state["p1_name"],
                p2_name=score_state["p2_name"],
                p1_pts=score_state["p1_pts"],
                p2_pts=score_state["p2_pts"],
                p1_games=score_state["p1_games"],
                p2_games=score_state["p2_games"],
                server=score_state["server"],
                p1_country=p1_country,
                p2_country=p2_country,
            )

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()

    print(f"\n--- MATCH EVENT SUMMARY ---")
    det_rate = (ball_detect_count / frame_count) * 100 if frame_count > 0 else 0
    print(f"Ball Detection Rate: {ball_detect_count}/{frame_count} frames ({det_rate:.1f}%)")
    print(f"Total Physics Events: {total_bounce_events} bounces, {total_hit_events} hits")
    print(f"Final Score: {p1_name} {score_state['p1_pts']} ({score_state['p1_games']} games) - "
          f"{p2_name} {score_state['p2_pts']} ({score_state['p2_games']} games)")
    print(f"Total Highlight Rallies Identified: {len(score_engine.vital_clips)}")
    for start, end, desc in score_engine.vital_clips:
        print(f"  • Highlight [{start:.1f}s - {end:.1f}s]: {desc}")

    # Dump the full event log for debugging
    if score_engine.event_log:
        print(f"\n--- EVENT LOG ({len(score_engine.event_log)} events) ---")
        for entry in score_engine.event_log:
            print(f"  [{entry['timestamp']:6.2f}s] {entry['event']:18s} | {entry['details']} | Score: {entry['score']}")

    # 4. Final Optimization Phase (Audio Merging & Selected Quality Mode)
    if mode == "realesrgan":
        print("\n--- RUNNING REAL-ESRGAN AI SUPER-RESOLUTION ---")
        enhancer = RealESRGANEnhancer(scale=upscale_factor, model_name=model_name)
        enhancer.upscale_video(
            input_video_path=temp_raw_path,
            output_video_path=output_path,
            source_audio_path=input_path,
            preserve_audio=True
        )
        if Path(temp_raw_path).exists():
            Path(temp_raw_path).unlink()

    elif mode == "broadcast4k":
        print("\n--- RUNNING BROADCAST 4K ENHANCEMENT ---")
        enhance_broadcast_4k(
            input_video_path=temp_raw_path,
            output_video_path=output_path,
            source_audio_path=input_path,
            target_width=3840,
            target_height=2160,
            sharpness=0.5
        )
        if Path(temp_raw_path).exists():
            Path(temp_raw_path).unlink()

    else:
        # Standard mode: Pristine H.264 encode with audio merged from input_path
        print("\n--- FINALIZING PRISTINE 1080P OUTPUT WITH AUDIO ---")
        merge_audio(
            video_no_audio=temp_raw_path,
            source_with_audio=input_path,
            output_path=output_path,
            clean_temp=True
        )

    print(f"\nProcessing complete! Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoEngine Test Pipeline")
    parser.add_argument("--input", "-i", type=str, default="test_clip_2.mov", help="Input video path")
    parser.add_argument("--output", "-o", type=str, default="test_output.mp4", help="Output video path")
    parser.add_argument("--mode", "-m", type=str, default=None,
                        choices=["standard", "broadcast4k", "realesrgan"],
                        help="Processing quality mode")
    parser.add_argument("--scale", "-s", type=int, default=2, choices=[2, 3, 4], help="Upscale factor for Real-ESRGAN")
    parser.add_argument("--model", type=str, default="realesrgan-x4plus",
                        choices=["realesrgan-x4plus", "realesr-animevideov3", "realesrgan-x4plus-anime"],
                        help="Real-ESRGAN model to use")
    
    args = parser.parse_args()

    selected_mode = args.mode
    if selected_mode is None:
        print("\n--- OUTPUT RESOLUTION & QUALITY ---")
        print("  [1] Standard 1080p (Fast, pristine H.264 camera quality + Audio)")
        print("  [2] Broadcast 4K (Fast, 3840x2160 Lanczos + Adaptive Sharpening + Audio)")
        print("  [3] Real-ESRGAN AI (Deep AI GAN Super-Resolution 2x/4x + Audio)")
        choice = input("Select quality mode [1/2/3] (default: 1): ").strip()
        if choice == "2":
            selected_mode = "broadcast4k"
        elif choice == "3":
            selected_mode = "realesrgan"
        else:
            selected_mode = "standard"
        print()

    run_visual_test(
        input_path=args.input,
        output_path=args.output,
        mode=selected_mode,
        upscale_factor=args.scale,
        model_name=args.model
    )