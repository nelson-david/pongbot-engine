import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from renderer.video_processor import BroadcastRenderer

def setup_player_names(cap, model):
    """
    Reads the first valid frame, identifies player positions,
    and prompts the user in the CLI to enter custom names.
    """
    ret, frame = cap.read()
    if not ret:
        return "Player 1", "Player 2"
    
    # Reset video capture back to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Run inference on frame 1 to get initial player positions
    results = model(frame, verbose=False)[0]
    player_positions = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        if label == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_x = (x1 + x2) / 2
            player_positions.append((center_x, (x1, y1, x2, y2)))

    # Sort players horizontally (Left to Right) by center X coordinate
    player_positions.sort(key=lambda item: item[0])

    print("\n--- PLAYER IDENTIFICATION SETUP ---")
    
    # Ask for Left/Far side player name
    if len(player_positions) > 0:
        p1_input = input("Enter name for Player 1 [Left / Far Side]: ").strip()
        p1_name = p1_input if p1_input else "Player 1"
    else:
        p1_name = "Player 1"

    # Ask for Right/Near side player name
    if len(player_positions) > 1:
        p2_input = input("Enter name for Player 2 [Right / Near Side]: ").strip()
        p2_name = p2_input if p2_input else "Player 2"
    else:
        p2_name = "Player 2"

    print(f"Match configured: {p1_name} vs {p2_name}\n")
    return p1_name, p2_name


def run_visual_test(input_path: str, output_path: str):
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    # Instantiate the new PIL-based BroadcastRenderer
    renderer = BroadcastRenderer()

    cap = cv2.VideoCapture(input_path)
    
    # Prompt for custom player names using positional cues
    p1_name, p2_name = setup_player_names(cap, model)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    simulated_p1_score = 0
    simulated_p2_score = 0
    simulated_p1_games = 1
    simulated_p2_games = 0
    current_server = 1  # 1 = Player 1 serving, 2 = Player 2 serving

    print(f"Processing '{input_path}' ({total_frames} total frames)...")

    with tqdm(total=total_frames, desc="Rendering Engine", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = frame_count / fps

            results = model(frame, verbose=False)[0]

            # 1. Draw Player Bounding Boxes & Tracking Labels
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = results.names[cls_id]
                if label == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    center_x = (x1 + x2) / 2
                    
                    # Label bounding box based on left vs right side
                    player_label = p1_name if center_x < (width / 2) else p2_name
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, player_label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 2. Simulated Event Trigger (e.g., point scored at 5-second mark)
            if frame_count == fps * 5:
                simulated_p1_score += 1
                current_server = 2  # Switch service to Player 2
                tqdm.write(f"[EVENT] Point scored at {timestamp:.2f}s! Score: {p1_name}={simulated_p1_score}, {p2_name}={simulated_p2_score}")

            # 3. Burn High-Quality WTT Broadcast Overlay
            frame = renderer.draw_scorecard(
                frame=frame,
                p1_name=p1_name,
                p2_name=p2_name,
                p1_pts=simulated_p1_score,
                p2_pts=simulated_p2_score,
                p1_games=simulated_p1_games,
                p2_games=simulated_p2_games,
                server=current_server
            )

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"\nProcessing complete! Output saved to: {output_path}")

if __name__ == "__main__":
    run_visual_test("test_clip.mp4", "test_output.mp4")