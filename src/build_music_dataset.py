"""Build the tokenised melody dataset used to train the LSTM.

    # EMOPIA (recommended): files are named Q1_xxx.mid ... Q4_xxx.mid
    python -m src.build_music_dataset --emopia data/EMOPIA_1.0/midis

    # your own MIDI folders with a mood label (can repeat)
    python -m src.build_music_dataset --labeled data/my_happy:happy --labeled data/my_sad:sad

    # synthetic rule-based melodies (to test the pipeline without downloading anything)
    python -m src.build_music_dataset --synthetic 300
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .config import EMOPIA_TO_MOOD, MELODY_DATA, MOODS, SEQ_LEN
from .midi_utils import extract_melody, tokenize, transpose
from .rule_based import rule_melody


def midi_files(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in {".mid", ".midi"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emopia")
    p.add_argument("--labeled", action="append", default=[], help="FOLDER:mood")
    p.add_argument("--synthetic", type=int, default=0, help="melodies per mood")
    p.add_argument("--transpose", type=int, default=3, help="augment with +-N semitone shifts")
    p.add_argument("--out", default=str(MELODY_DATA))
    a = p.parse_args()

    items = []  # (melody, mood_idx)
    if a.emopia:
        for f in midi_files(a.emopia):
            q = f.name[:2]
            if q in EMOPIA_TO_MOOD:
                try:
                    items.append((extract_melody(f), EMOPIA_TO_MOOD[q]))
                except Exception as e:  # noqa: BLE001 - corrupt files exist in the wild
                    print("skip", f.name, e)
    for spec in a.labeled:
        folder, mood = spec.rsplit(":", 1)
        for f in midi_files(folder):
            try:
                items.append((extract_melody(f), MOODS.index(mood)))
            except Exception as e:  # noqa: BLE001
                print("skip", f.name, e)
    for m, mood in enumerate(MOODS):
        for i in range(a.synthetic):
            items.append((rule_melody(mood, n_bars=32, seed=m * 100000 + i), m))

    if not items:
        raise SystemExit("No data: pass --emopia, --labeled or --synthetic")

    seqs, moods, pieces = [], [], []
    for piece_id, (mel, mood) in enumerate(items):
        if len(mel) < SEQ_LEN + 1:
            continue
        for shift in range(-a.transpose, a.transpose + 1):
            seqs.append(torch.tensor(tokenize(transpose(mel, shift)), dtype=torch.long))
            moods.append(mood)
            pieces.append(piece_id)  # lets training split by piece (no transposed copies leaking into val)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sequences": seqs, "moods": moods, "pieces": pieces}, a.out)
    counts = np.bincount(moods, minlength=len(MOODS))
    print(f"{len(items)} pieces -> {len(seqs)} sequences (with transposition) -> {a.out}")
    print("per mood:", dict(zip(MOODS, counts.tolist())))


if __name__ == "__main__":
    main()
