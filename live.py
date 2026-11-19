"""Live webcam mode: shows your mood in real time; SPACE composes and plays a song, Q quits.

    python live.py
"""
import collections
import os
import sys

import cv2
import numpy as np

from src.config import MOODS
from src.face import EmotionDetector, draw
from src.generate import compose, save


def play(path):
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
    elif sys.platform == "darwin":
        os.system(f"afplay '{path}' &")
    else:
        os.system(f"aplay -q '{path}' &")


def main():
    det = EmotionDetector()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise SystemExit("No webcam found (is another app using it?)")
    history = collections.deque(maxlen=15)  # smooth predictions over ~0.5s
    n = 0
    print("SPACE = compose for current mood, Q = quit")
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = det.predict(frame)
        if res:
            history.append([res["probs"][m] for m in MOODS])
            avg = np.mean(history, axis=0)
            res["mood"] = MOODS[int(avg.argmax())]
            res["probs"] = dict(zip(MOODS, avg.tolist()))
            frame = draw(frame, res)
            for i, m in enumerate(MOODS):  # probability bars
                cv2.rectangle(frame, (10, 30 + i * 28), (10 + int(avg[i] * 200), 50 + i * 28), (230, 160, 60), -1)
                cv2.putText(frame, m, (15, 46 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        else:
            history.clear()  # face lost: don't compose for an old mood
        cv2.imshow("EmoMusic live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and history:
            mood = MOODS[int(np.mean(history, axis=0).argmax())]
            pm, _, used = compose(mood, bars=8)
            n += 1
            _, wav = save(pm, f"live_{n:02d}_{mood}")
            print(f"composed {mood} ({used}) -> {wav}")
            play(wav)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
