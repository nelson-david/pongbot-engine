# engine/main.py
import cv2
import subprocess
from vision.tracker import MultiModelTracker
from logic.state_machine import MatchScoreEngine
from renderer.video_processor import BroadcastRenderer
from enhancer.video_upscaler import RealESRGANEnhancer

def run_engine(
    input_video_path: str,
    output_video_path: str,
    enable_upscale: bool = False,
    upscale_factor: int = 2,
):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize Core Subsystems
    tracker = MultiModelTracker(yolo_model_path="yolov8n.pt", frame_width=width, frame_height=height)
    score_engine = MatchScoreEngine(p1_name="Player 1", p2_name="Player 2", initial_server=1)
    renderer = BroadcastRenderer()
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    temp_processed_path = "temp_overlay_output.mp4"
    out = cv2.VideoWriter(temp_processed_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_idx / fps
        
        # 1. Run Automated Vision Tracking (Players + Table + Ball Trajectory)
        players, physics_events = tracker.process_frame(frame, timestamp)
        
        # 2. Update Engine Logic with Real Physics Events
        score_engine.update_physics_frame(timestamp, physics_events)
        score_state = score_engine.get_score_state()
        
        # 3. Render Visual Overlays & WTT Scoreboard
        frame = tracker.draw_overlays(frame)
        frame_with_overlay = renderer.draw_scorecard(
            frame=frame,
            p1_name=score_state["p1_name"],
            p2_name=score_state["p2_name"],
            p1_pts=score_state["p1_pts"],
            p2_pts=score_state["p2_pts"],
            p1_games=score_state["p1_games"],
            p2_games=score_state["p2_games"],
            server=score_state["server"]
        )
        out.write(frame_with_overlay)
        
        frame_idx += 1

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
        print("Video frame processing complete. Merging audio and finalizing output...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", temp_processed_path,
            "-i", input_video_path,
            "-c:v", "libx265", "-crf", "21", "-preset", "fast",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_video_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)
    
    import os
    if os.path.exists(temp_processed_path):
        os.remove(temp_processed_path)

    print(f"Final sports video rendered successfully: {output_video_path}")

if __name__ == "__main__":
    run_engine("raw_match.mp4", "output_scored_match.mp4", enable_upscale=False)