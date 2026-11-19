"""Turns a face into musical controls using MediaPipe FaceLandmarker (478 landmarks + 52 blendshapes).

    mouth  (jaw open)        -> melody on/off and note speed
    brows  (eyebrows raised) -> tempo
    eyes   (eye openness)    -> volume
    tilt   (head roll)       -> pitch
    smile / frown / brow-down -> mood (when the emotion CNN is not used)
"""
import math
import time
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import MODELS_DIR

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = MODELS_DIR / "face_landmarker.task"

# landmark groups used for drawing
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
EYES = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]


def ensure_model():
    """Downloads Google's face landmarker model (~3.7 MB) the first time."""
    if not MODEL_PATH.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmark model -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


@dataclass
class Controls:
    found: bool = False
    mouth: float = 0.0     # 0 closed .. 1 wide open
    brows: float = 0.0     # 0 relaxed .. 1 raised
    eyes: float = 1.0      # 0 closed .. 1 open
    tilt: float = 0.0      # -1 tilted left .. +1 tilted right (as seen on screen)
    smile: float = 0.0
    frown: float = 0.0
    brow_down: float = 0.0
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))

    def face_mood(self):
        """Simple rule-based mood from blendshapes (used when the CNN is off)."""
        if self.smile > 0.45:
            return "happy"
        if self.brow_down > 0.40:
            return "angry"
        if self.frown > 0.25:
            return "sad"
        return "calm"


class FaceTracker:
    def __init__(self, smoothing=0.5):
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=True)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.alpha = smoothing
        self.state = Controls()
        self.baseline = {"mouth": 0.0, "brows": 0.0}
        self._raw = {}
        self._t0 = time.monotonic()
        self._last_ts = -1

    def calibrate(self):
        """Call while making a neutral face: removes your resting mouth/brow position."""
        if self._raw:
            self.baseline = {"mouth": self._raw["mouth"], "brows": self._raw["brows"]}

    def process(self, bgr):
        import mediapipe as mp
        h, w = bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        # VIDEO mode requires strictly increasing timestamps (two frames in the same ms would crash)
        ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)
        self._last_ts = ts
        res = self.landmarker.detect_for_video(image, ts)
        if not res.face_landmarks:
            self.state.found = False
            self.state.mouth *= 0.6  # let the melody fade out when the face is lost
            return self.state

        bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
        pts = np.array([(lm.x * w, lm.y * h) for lm in res.face_landmarks[0]])
        mean = lambda *k: float(np.mean([bs.get(n, 0.0) for n in k]))  # noqa: E731

        # head roll from the outer eye corners (sorted left->right on screen)
        a, b = sorted([pts[33], pts[263]], key=lambda p: p[0])
        roll = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

        raw = {
            "mouth": mean("jawOpen"),
            "brows": mean("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
            "eyes": 1.0 - mean("eyeBlinkLeft", "eyeBlinkRight") + 0.5 * mean("eyeWideLeft", "eyeWideRight"),
            "tilt": roll / 25.0,
            "smile": mean("mouthSmileLeft", "mouthSmileRight"),
            "frown": mean("mouthFrownLeft", "mouthFrownRight"),
            "brow_down": mean("browDownLeft", "browDownRight"),
        }
        self._raw = raw
        target = dict(raw)
        target["mouth"] = (raw["mouth"] - self.baseline["mouth"]) / 0.6
        target["brows"] = (raw["brows"] - self.baseline["brows"]) / 0.6
        lo = {"tilt": -1.0}
        s = self.state
        for k, v in target.items():
            v = float(np.clip(v, lo.get(k, 0.0), 1.0))
            setattr(s, k, self.alpha * getattr(s, k) + (1 - self.alpha) * v)
        s.found = True
        s.points = pts.astype(int)
        return s
