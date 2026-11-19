"""🎻 Face Orchestra - you are the conductor.

    python orchestra.py                 # webcam + speakers
    python orchestra.py --cnn           # mood from the trained emotion CNN instead of blendshape rules
    python orchestra.py --demo          # no webcam: an automatic "ghost face" plays
    python orchestra.py --render-test 12   # no webcam, no speakers: writes output/orchestra_demo.wav

Controls
    mouth open   -> melody (wider = faster notes)      head tilt  -> pitch
    eyebrows up  -> tempo                              eyes       -> volume (close them to fade out)
    smile / frown / angry brows -> mood (scale, instruments, groove)

Keys
    C calibrate (neutral face)   H harmony   D drums   E echo (AI answers your phrase)
    M mood lock (auto/happy/sad/angry/calm)   R record   Q quit
"""
import argparse
import math
import time
from datetime import datetime

import cv2
import numpy as np

from src.config import MOOD_EMOJI, MOOD_STYLE, MOODS, OUTPUT_DIR, EMOTION_CKPT
from src.face_controls import BROWS, EYES, MOUTH, Controls
from src.live_synth import FS, FaceSynth, note_name

PANEL_W = 320
COLORS = {"happy": (60, 200, 255), "sad": (230, 140, 70), "angry": (70, 70, 235), "calm": (170, 210, 120)}
MOOD_CYCLE = ["auto"] + MOODS


class MoodSmoother:
    """Only switch mood after it has been stable for `hold` seconds (avoids flicker)."""

    def __init__(self, hold=1.2):
        self.hold, self.current, self.candidate, self.since = hold, "calm", "calm", time.monotonic()

    def update(self, mood):
        now = time.monotonic()
        if mood != self.candidate:
            self.candidate, self.since = mood, now
        elif mood != self.current and now - self.since > self.hold:
            self.current = mood
        return self.current


def ghost_face(t):
    """Synthetic controls for --demo / --render-test."""
    c = Controls(found=True)
    c.mouth = max(0.0, math.sin(t * 1.3)) * (0.5 + 0.5 * math.sin(t * 0.37))
    c.brows = 0.35 + 0.3 * math.sin(t * 0.11)
    c.eyes = 0.75 + 0.25 * math.sin(t * 0.5)
    c.tilt = math.sin(t * 0.9) * 0.9
    c.smile = max(0.0, math.sin(t * 0.21))
    c.frown = max(0.0, -math.sin(t * 0.21)) * 0.5
    return c


def meter(img, x, y, label, value, text, color, lo=0.0):
    w = PANEL_W - 40
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    cv2.putText(img, text, (x + w - tw, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y + 8), (x + w, y + 20), (60, 60, 60), -1)
    if lo < 0:  # centred meter (tilt)
        mid = x + w // 2
        end = int(mid + value * w / 2)
        cv2.rectangle(img, (min(mid, end), y + 8), (max(mid, end), y + 20), color, -1)
        cv2.line(img, (mid, y + 5), (mid, y + 23), (200, 200, 200), 1)
    else:
        cv2.rectangle(img, (x, y + 8), (x + int(np.clip(value, 0, 1) * w), y + 20), color, -1)


def draw_face(frame, c, color):
    if not len(c.points):
        return
    for i, (x, y) in enumerate(c.points):
        if i % 4 == 0:
            cv2.circle(frame, (int(x), int(y)), 1, (150, 150, 150), -1, cv2.LINE_AA)
    for group, col in ((MOUTH, color), (BROWS, (255, 200, 80)), (EYES, (255, 255, 255))):
        for i in group:
            cv2.circle(frame, tuple(int(v) for v in c.points[i]), 2, col, -1, cv2.LINE_AA)


def draw_panel(h, synth, c, mood_lock, fps):
    panel = np.full((h, PANEL_W, 3), 25, np.uint8)
    col = COLORS[synth.mood]
    x = 20
    cv2.putText(panel, "FACE ORCHESTRA", (x, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    st = MOOD_STYLE[synth.mood]
    cv2.putText(panel, f"mood: {synth.mood.upper()}" + ("" if mood_lock == "auto" else " (locked)"),
                (x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
    note = note_name(synth.last_pitch) if synth.last_pitch else "-"
    meter(panel, x, 110, "MOUTH  > melody", c.mouth, ["rest", "1/4", "1/8", "1/16"][[0, 4, 2, 1].index(synth.note_rate())], col)
    meter(panel, x, 155, "BROWS  > tempo", c.brows, f"{synth.tempo:.0f} bpm", col)
    meter(panel, x, 200, "EYES   > volume", c.eyes, f"{c.eyes:.0%}", col)
    meter(panel, x, 245, "TILT   > pitch", c.tilt, note, col, lo=-1)

    # beat indicator: 16 steps
    pos = (synth.step - 1) % 16
    for i in range(16):
        cx = x + 8 + i * 17
        filled = i == pos
        cv2.circle(panel, (cx, 300), 6 if i % 4 == 0 else 4, col if filled else (90, 90, 90), -1, cv2.LINE_AA)

    # big note flash
    age = time.monotonic() - synth.last_note_time
    if age < 0.25 and synth.last_pitch:
        cv2.putText(panel, note, (x, 370), cv2.FONT_HERSHEY_DUPLEX, 1.6, col, 2, cv2.LINE_AA)
    cv2.putText(panel, f"chord {synth.chord or '-'}", (x + 150, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (220, 220, 220), 1, cv2.LINE_AA)

    toggles = [("H harmony", synth.harmony), ("D drums", synth.drums), ("E echo", synth.echo),
               ("R rec", synth.recording)]
    for i, (name, on) in enumerate(toggles):
        cv2.putText(panel, name, (x + (i % 2) * 140, 392 + (i // 2) * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (80, 220, 80) if on else (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{synth.mood} scale: {len(st['scale'])} notes, GM #{st['program']}", (x, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(panel, "C calibrate  M mood  Q quit", (x, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{fps:.0f} fps", (PANEL_W - 60, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (100, 100, 100), 1, cv2.LINE_AA)
    if synth.recording and int(time.monotonic() * 2) % 2:
        cv2.circle(panel, (PANEL_W - 25, 30), 8, (0, 0, 255), -1)
    return panel


def apply_controls(synth, c, mood):
    synth.mouth = c.mouth if c.found else 0.0
    synth.brows, synth.eyes, synth.tilt = c.brows, c.eyes, c.tilt
    synth.pending_mood = mood


def render_test(seconds):
    """Offline check of the whole music engine with the ghost face (no camera, no speakers)."""
    synth = FaceSynth()
    synth.echo = True
    synth.start_recording()
    block, t = 512, 0.0
    smoother = MoodSmoother(hold=0)
    while t < seconds:
        c = ghost_face(t * 1.5)
        apply_controls(synth, c, smoother.update(c.face_mood()))
        synth.render(block)
        t += block / FS
    mid, wav = synth.stop_recording(OUTPUT_DIR, "orchestra_demo")
    print(f"{len(synth._events)} notes, last mood {synth.mood}, chord {synth.chord}\n -> {mid}\n -> {wav}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn", action="store_true", help="use the trained emotion CNN for the mood")
    p.add_argument("--demo", action="store_true", help="no webcam: ghost face plays")
    p.add_argument("--render-test", type=float, metavar="SECONDS")
    p.add_argument("--camera", type=int, default=0)
    a = p.parse_args()
    if a.render_test:
        return render_test(a.render_test)

    tracker = detector = cam = None
    if not a.demo:
        from src.face_controls import FaceTracker
        tracker = FaceTracker()
        cam = cv2.VideoCapture(a.camera)
        if not cam.isOpened():
            raise SystemExit("No webcam found - try --demo")
    if a.cnn:
        if not EMOTION_CKPT.exists():
            raise SystemExit("No emotion CNN yet: train it (python -m src.train_emotion) or drop --cnn")
        from src.face import EmotionDetector
        detector = EmotionDetector()

    synth = FaceSynth()
    synth.start()
    smoother, mood_lock, cnn_mood = MoodSmoother(), "auto", "calm"
    t0, frame_i, fps, last = time.monotonic(), 0, 0.0, time.monotonic()
    print(__doc__)
    try:
        while True:
            if cam is not None:
                ok, frame = cam.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # mirror: tilting right on screen = higher
                c = tracker.process(frame)
                if detector is not None and frame_i % 10 == 0:
                    res = detector.predict(frame)
                    cnn_mood = res["mood"] if res else cnn_mood
            else:
                frame = np.full((540, 720, 3), 15, np.uint8)
                c = ghost_face(time.monotonic() - t0)
                cv2.putText(frame, "DEMO MODE - ghost face", (200, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (120, 120, 120), 1, cv2.LINE_AA)

            mood = mood_lock if mood_lock != "auto" else smoother.update(cnn_mood if detector else c.face_mood())
            apply_controls(synth, c, mood)

            draw_face(frame, c, COLORS[synth.mood])
            if not c.found and cam is not None:
                cv2.putText(frame, "show your face :)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            cv2.imshow("Face Orchestra", np.hstack([frame, draw_panel(frame.shape[0], synth, c, mood_lock, fps)]))
            frame_i += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty("Face Orchestra", cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord("c") and tracker:
                tracker.calibrate()
                print("calibrated neutral face")
            elif key == ord("h"):
                synth.harmony = not synth.harmony
            elif key == ord("d"):
                synth.drums = not synth.drums
            elif key == ord("e"):
                synth.echo = not synth.echo
            elif key == ord("m"):
                mood_lock = MOOD_CYCLE[(MOOD_CYCLE.index(mood_lock) + 1) % len(MOOD_CYCLE)]
                print("mood:", mood_lock, MOOD_EMOJI.get(mood_lock, ""))
            elif key == ord("r"):
                if synth.recording:
                    mid, wav = synth.stop_recording(OUTPUT_DIR, datetime.now().strftime("orchestra_%Y%m%d_%H%M%S"))
                    print(f"saved {mid}\n      {wav}")
                else:
                    synth.start_recording()
                    print("recording...")
    finally:
        if synth.recording:
            synth.stop_recording(OUTPUT_DIR, datetime.now().strftime("orchestra_%Y%m%d_%H%M%S"))
        synth.stop()
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
