# PongCast Engine

The core computer vision and video rendering pipeline for PongCast. It handles ball trajectory tracking, table bounce detection, ITTF match rules enforcement, and WTT-style broadcast overlay rendering.

**Repository**: [https://github.com/nelson-david/pongbot-engine.git](https://github.com/nelson-david/pongbot-engine.git)

---

## Capabilities

- **Player & Ball Tracking**: YOLOv8-powered object detection combined with color and contour trajectory modeling.
- **Physics Spotting**: Frame-by-frame identification of table bounces, net hits, and paddle strikes.
- **ITTF Scoring Engine**: Official 11-point system, service rotation every 2 points, deuce logic (win by 2), and game tracking.
- **Broadcast Renderer**: Generates TV-style scoreboards with player names, national flags, serve indicators, and rally counts.
- **Audio Preservation**: Merges original ambient match audio into the final output via FFmpeg.

---

## Directory Structure

```
engine/
├── assets/          → Fonts and SVG flag assets
├── logic/           → ITTF state machine rules engine (state_machine.py)
├── renderer/        → TV broadcast scoreboard and overlay generator (video_processor.py)
├── vision/          → Ball tracker, court detector, and multi-model pipeline (tracker.py)
├── enhancer/        → Optional Real-ESRGAN super-resolution upscaler
├── main.py          → Engine entry point (run_engine)
└── requirements.txt → Python package dependencies
```

---

## Setup & Installation

### 1. System Requirements
- Python 3.10+
- FFmpeg:
  - **macOS**: `brew install ffmpeg`
  - **Ubuntu / Debian**: `sudo apt update && sudo apt install -y ffmpeg libgl1 libglib2.0-0`

### 2. Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. CLI Execution (Standalone Test)
```bash
python3 main.py
```
This runs the engine against `test_clip_2.mov` and outputs the rendered video with broadcast graphics to `main_clip.mp4`.
