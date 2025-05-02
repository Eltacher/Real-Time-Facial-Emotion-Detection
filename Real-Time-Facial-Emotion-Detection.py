# ===== Imports =====
import cv2
import pygame
from deepface import DeepFace

# ===== Runtime settings =====
RESIZE       = (640, 480)   # camera frame size
SKIP_FRAMES  = 2            # analyze every N-th frame
BACKEND      = "opencv"     # lightweight detector

# ===== Look-up tables =====
COLOR_MAP = {                       # BGR
    "angry":    (0,   0, 255),
    "disgust":  (255, 0, 255),
    "fear":     (128, 128, 128),
    "happy":    (0, 255, 0),
    "sad":      (255, 0, 0),
    "surprise": (0, 255, 255),
    "neutral":  (255, 255, 255),
}
TR_EMOTION = {                      # English → Turkish label
    "angry": "Kizgin", "disgust": "Tiksinti", "fear": "Korku",
    "happy": "Mutlu", "sad": "Uzgun", "surprise": "Saskin", "neutral": "Notr"
}
SFX = {                             # sound per emotion
    "happy": "happy.mp3", "sad": "sad.mp3",
    "neutral": "neutral.mp3", "angry": "angry.mp3"
}

# ===== Init =====
cap = cv2.VideoCapture(0)
pygame.mixer.init()
frame_idx, cached_faces, last_emotion = 0, [], None

# ===== Main loop =====
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.resize(frame, RESIZE)
    frame_idx += 1
    should_analyze = (frame_idx % SKIP_FRAMES == 0)

    if should_analyze:
        try:
            res = DeepFace.analyze(
                frame, actions=['emotion'],
                detector_backend=BACKEND, enforce_detection=False
            )
            cached_faces = res if isinstance(res, list) else [res]
        except Exception as e:
            print("DeepFace:", e)
            cached_faces = []

    # draw detections + play sounds
    for face in cached_faces:
        x, y, w, h = (face['region'][k] for k in ('x', 'y', 'w', 'h'))
        emo        = face['dominant_emotion']
        cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_MAP.get(emo, (0,255,0)), 2)
        cv2.putText(frame, TR_EMOTION.get(emo, emo), (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_MAP.get(emo, (0,255,0)), 2)

        if emo in SFX and emo != last_emotion and should_analyze:
            pygame.mixer.music.load(SFX[emo])
            pygame.mixer.music.play()
            last_emotion = emo

    cv2.imshow("Real-Time Facial Emotion Detection", frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')) or \
       (cv2.getWindowProperty("Real-Time Facial Emotion Detection",
                              cv2.WND_PROP_VISIBLE) < 1):
        break

# ===== Cleanup =====
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
