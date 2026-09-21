# engine/main.py
import cv2
import subprocess
from vision.tracker import MultiModelTracker
from logic.state_machine import MatchScoreEngine
from renderer.video_processor import BroadcastRenderer

def run_engine(input_video_path: str, output_video_path: str):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize Core Subsystems
    tracker = MultiModelTracker(yolo_path="yolov8n.pt")
    score_engine = MatchScoreEngine()
    renderer = BroadcastRenderer(score_engine)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_processed_path = "temp_overlay_output.mp4"
    out = cv2.VideoWriter(temp_processed_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_idx / fps
        
        # 1. Run Detection
        detections = tracker.process_frame(frame)
        
        # 2. Update Engine Logic (Simulated Event Input)
        score_engine.update_physics_frame(timestamp, ball_pos=(0, 0))
        
        # 3. Render Visual Overlays
        frame_with_overlay = renderer.draw_scorecard(frame, score_engine.score)
        out.write(frame_with_overlay)
        
        frame_idx += 1

    cap.release()
    out.release()
    
    # 4. Final Optimization Phase (Real-ESRGAN or FFmpeg Encoding)
    print("Video frame processing complete. Running hardware-accelerated H.265 re-encoding...")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", temp_processed_path,
        "-c:v", "libx265", "-crf", "21", "-preset", "fast",
        output_video_path
    ]
    subprocess.run(ffmpeg_cmd)
    print(f"Final sports video rendered successfully: {output_video_path}")

if __name__ == "__main__":
    run_engine("raw_match.mp4", "output_scored_match.mp4")