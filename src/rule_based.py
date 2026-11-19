"""Rule-based melody generator.

Used for: (1) a synthetic training set to test the pipeline before you download EMOPIA,
(2) a fallback when no trained LSTM exists, (3) the "baseline" to compare the LSTM against.
"""
import numpy as np

from .config import MOOD_STYLE

RHYTHM_CELLS = {  # each cell = one bar (4 beats)
    "happy": [[.5, .5, 1, 1, 1], [.25, .25, .5, 1, 1, 1], [.5, .5, .5, .5, 2], [1, .5, .5, 1, 1]],
    "sad":   [[2, 2], [1.5, .5, 2], [1, 1, 2], [3, 1], [4]],
    "angry": [[.25] * 8 + [.5] * 4, [.5, .25, .25] * 2 + [.5] * 4, [.25, .25, .5] * 4],
    "calm":  [[2, 2], [1, 1, 2], [1.5, .5, 2], [4], [3, 1]],
}
STEPS = {  # possible moves in scale degrees and their weights
    "happy": ([-2, -1, 1, 2, 3, 4], [1, 3, 4, 3, 2, 1]),
    "sad":   ([-2, -1, 0, 1, 2], [2, 5, 1, 2, 1]),
    "angry": ([-4, -1, 0, 1, 3, 5], [1, 2, 4, 2, 1, 1]),
    "calm":  ([-1, 1, -2, 2], [4, 4, 1, 1]),
}
REST_P = {"happy": .05, "sad": .1, "angry": .03, "calm": .12}


def rule_melody(mood, n_bars=8, seed=None):
    rng = np.random.default_rng(seed)
    st = MOOD_STYLE[mood]
    scale, root = st["scale"], st["root"] + 12
    moves, w = STEPS[mood]
    w = np.array(w, dtype=float) / sum(w)
    degree = 0
    melody = []
    for bar in range(n_bars):
        cell = RHYTHM_CELLS[mood][rng.integers(len(RHYTHM_CELLS[mood]))]
        for i, d in enumerate(cell):
            last = bar == n_bars - 1 and i == len(cell) - 1
            if last:
                degree = 0  # resolve on the tonic
            elif rng.random() < REST_P[mood] and melody:
                melody.append((None, d))
                continue
            else:
                degree = int(np.clip(degree + rng.choice(moves, p=w), -len(scale), 2 * len(scale)))
            octave, idx = divmod(degree, len(scale))
            melody.append((root + 12 * octave + scale[idx], d))
    return melody
