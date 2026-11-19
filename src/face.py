"""Face detection (OpenCV Haar cascade) + emotion prediction."""
import cv2
import numpy as np
import torch

from .config import EMOTION_CKPT, IMG_SIZE, MOODS
from .emotion_cnn import load_emotion_model
from .emotion_dataset import preprocess

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_faces(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return sorted((tuple(map(int, f)) for f in faces), key=lambda b: -b[2] * b[3]), gray


class EmotionDetector:
    def __init__(self, ckpt=EMOTION_CKPT, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_emotion_model(ckpt, self.device)

    @torch.no_grad()
    def predict_face(self, gray_face):
        face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
        x = preprocess(face).unsqueeze(0).to(self.device)
        # test-time augmentation: average with the mirrored face
        probs = torch.softmax(self.model(torch.cat([x, x.flip(-1)])), 1).mean(0)
        return probs.cpu().numpy()

    def predict(self, bgr):
        """Returns dict(mood, probs{mood: p}, box) for the biggest face, or None."""
        faces, gray = detect_faces(bgr)
        if not faces:
            return None
        x, y, w, h = faces[0]
        pad = int(0.1 * w)  # a bit of margin, FER faces are loosely cropped
        y0, y1 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        probs = self.predict_face(gray[y0:y1, x0:x1])
        return {"mood": MOODS[int(np.argmax(probs))], "probs": dict(zip(MOODS, probs.tolist())),
                "box": (x, y, w, h)}


def draw(bgr, result):
    if result is None:
        return bgr
    x, y, w, h = result["box"]
    out = bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (80, 200, 120), 2)
    label = f"{result['mood']} {result['probs'][result['mood']]:.0%}"
    cv2.putText(out, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 120), 2)
    return out
