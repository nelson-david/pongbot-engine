# engine/main.py
import os
import cv2
import subprocess
from vision.tracker import MultiModelTracker
from logic.state_machine import MatchScoreEngine
from renderer.video_processor import BroadcastRenderer
from enhancer.video_upscaler import RealESRGANEnhancer

def run_engine(
    input_video_path: str,
    output_video_path: str,
    p1_name: str = "Player 1",
    p2_name: str = "Player 2",
    p1_country: str = "NG",
    p2_country: str = "NG",
    initial_server: int = 1,
    enable_upscale: bool = False,
    upscale_factor: int = 2,
    progress_callback = None,
    log_callback = None,
):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    
    # Initialize Core Subsystems
    tracker = MultiModelTracker(yolo_model_path="yolov8n.pt", frame_width=width, frame_height=height)
    score_engine = MatchScoreEngine(p1_name=p1_name, p2_name=p2_name, initial_server=initial_server)
    renderer = BroadcastRenderer()
    
    # Ensure absolute paths and destination directory exist
    input_video_path = os.path.abspath(input_video_path)
    output_video_path = os.path.abspath(output_video_path)
    output_dir = os.path.dirname(output_video_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_processed_path = os.path.join(output_dir, f"temp_{os.path.basename(output_video_path)}")
    out = cv2.VideoWriter(temp_processed_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    init_msg = f"Match configured: {p1_name} [{p1_country}] vs {p2_name} [{p2_country}]"
    print(f"\n{init_msg}", flush=True)
    if log_callback:
        log_callback("INFO", init_msg, "00:00.01")

    proc_msg = f"Processing '{os.path.basename(input_video_path)}' ({total_frames} total frames)..."
    print(proc_msg, flush=True)
    if log_callback:
        log_callback("INFO", proc_msg, "00:00.05")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_idx / fps
        
        # 1. Run Automated Vision Tracking (Players + Table + Ball Trajectory)
        players, physics_events = tracker.process_frame(frame, timestamp)
        
        # 2. Update Engine Logic with Real Physics Events
        prev_p1_pts, prev_p2_pts = score_engine.p1_pts, score_engine.p2_pts
        score_engine.update_physics_frame(timestamp, physics_events)
        score_state = score_engine.get_score_state()

        if score_engine.p1_pts != prev_p1_pts or score_engine.p2_pts != prev_p2_pts:
            score_msg = f"Point scored at {timestamp:.2f}s! Score: {p1_name}={score_engine.p1_pts}, {p2_name}={score_engine.p2_pts}"
            print(f"[EVENT] {score_msg}", flush=True)
            if log_callback:
                log_callback("SCORE", score_msg, f"{timestamp:.2f}s")
        elif physics_events:
            for ev in physics_events:
                ev_side = f"Side {ev.get('side')}" if ev.get('side') else f"Player {ev.get('player')}"
                ev_msg = f"{ev['type']} spotted at {timestamp:.2f}s on {ev_side}"
                print(f"[EVENT] {ev_msg}", flush=True)
                if log_callback:
                    log_callback("BOUNCE" if ev['type'] == "BOUNCE" else "INFO", ev_msg, f"{timestamp:.2f}s")
        
        # 3. Render Visual Overlays & WTT Scoreboard
        frame = tracker.draw_overlays(frame, p1_name=p1_name, p2_name=p2_name)
        frame_with_overlay = renderer.draw_scorecard(
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
        out.write(frame_with_overlay)
        
        frame_idx += 1

        if frame_idx % 60 == 0 or frame_idx == total_frames:
            pct = int((frame_idx / total_frames) * 100)
            prog_msg = f"Frame {frame_idx}/{total_frames} ({pct}%) | Score: {p1_name} {score_state['p1_pts']}-{score_state['p2_pts']} {p2_name}"
            print(f"[PROGRESS {pct}%] {prog_msg}", flush=True)
            if log_callback:
                log_callback("INFO", prog_msg, f"{pct}%")

        if progress_callback and (frame_idx % 20 == 0 or physics_events):
            progress_callback(frame_idx, total_frames, physics_events, score_state)

    cap.release()
    out.release()
    
    # 4. Final Optimization Phase (Audio Merging & Optional Real-ESRGAN Upscaling)
    if enable_upscale:
        print("Running Real-ESRGAN AI Super-Resolution...")
        enhancer = RealESRGANEnhancer(enabled=True, scale=upscale_factor)
        enhancer.upscale_video(
            input_video_path=temp_processed_path,
            output_video_path=output_video_path,
            preserve_audio=True
        )
    else:
        merge_msg = "Video frame processing complete. Merging audio and finalizing output..."
        print(merge_msg, flush=True)
        if log_callback:
            log_callback("INFO", merge_msg)

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", temp_processed_path,
            "-i", input_video_path,
            "-c:v", "libx264", "-crf", "21", "-preset", "fast",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_video_path
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            # Fallback if audio track mapping fails (e.g. video recorded without sound)
            print(f"[WARNING] Audio merge failed ({e}), falling back to video-only encode...", flush=True)
            fallback_cmd = [
                "ffmpeg", "-y",
                "-i", temp_processed_path,
                "-c:v", "libx264", "-crf", "21", "-preset", "fast",
                "-an",
                output_video_path
            ]
            subprocess.run(fallback_cmd, check=True)
    
    if os.path.exists(temp_processed_path):
        try:
            os.remove(temp_processed_path)
        except OSError:
            pass

    success_msg = f"Final sports video rendered successfully: {output_video_path}"
    print(success_msg, flush=True)
    if log_callback:
        log_callback("SUCCESS", f"Broadcast video rendered: {os.path.basename(output_video_path)}")

    return score_engine.get_score_state()

if __name__ == "__main__":
    run_engine("test_clip_2.mov", "main_clip.mp4", p1_name="David", p2_name="Chukwuka", p1_country="NG", p2_country="DE")