# Real-Time Facial Emotion Detection 

Tiny desktop demo that:

1. Captures live video from your webcam  
2. Detects faces and classifies the dominant emotion with **DeepFace**  
3. Draws a colour-coded bounding-box + Turkish label for each face  
4. Plays a short sound (📢 `.mp3`) whenever a new emotion is recognised

## Features

- **CPU-only** by default (runs ~10–20 FPS on a laptop)  
- Lightweight **OpenCV** detector backend for speed  
- Skips every other frame ↔ balances speed vs. accuracy  
- Clean separation of colour/label/sound maps—easy to customise  
- Works offline once the Python packages are installed

---

## Requirements

| Package | Tested version |
|---------|----------------|
| Python  | 3.9 / 3.10 |
| opencv-python | 4.9+ |
| pygame  | 2.5+ |
| deepface | 0.40+ |
| tensorflow | 2.15 (auto-installed by DeepFace)
---

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/<your-user>/Real-Time-Facial-Emotion-Detection.git
cd Real-Time-Facial-Emotion-Detection

# 2. (Optional) create + activate venv
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt  # <- create this file via `pip freeze` OR:
pip install opencv-python pygame deepface

# 4. Run the demo
python Real-Time-Facial-Emotion-Detection.py
