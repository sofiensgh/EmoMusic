"""Compose a piece for a mood.

    python -m src.generate --mood happy --bars 8 --temperature 1.0
"""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .config import MELODY_CKPT, MOODS, OUTPUT_DIR
from .melody_lstm import load_melody_model, sample
from .midi_utils import detokenize, render_midi, synthesize, tokenize, wav_bytes
from .rule_based import rule_melody


@lru_cache(maxsize=1)
def _cached_model(path, mtime):
    return load_melody_model(path)


def _model(path):
    # keyed on the file's modification time, so a newly trained checkpoint is picked up
    # without restarting the app
    return _cached_model(path, Path(path).stat().st_mtime)


def lstm_available(path=MELODY_CKPT):
    return Path(path).exists()


def compose(mood, bars=8, temperature=1.0, top_k=12, accompaniment=True, seed=None,
            engine="auto", ckpt=MELODY_CKPT):
    """Returns (pretty_midi.PrettyMIDI, melody, engine_used)."""
    use_lstm = engine == "lstm" or (engine == "auto" and lstm_available(ckpt))
    target_beats = bars * 4
    if use_lstm:
        model = _model(str(ckpt))
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        seed_mel = rule_melody(mood, n_bars=1, seed=seed)[:3]  # short mood-flavoured motif
        tokens = sample(model, MOODS.index(mood), tokenize(seed_mel), n_tokens=target_beats * 4,
                        temperature=temperature, top_k=top_k, generator=g)
        melody, beats = [], 0.0
        for p, d in detokenize(tokens):
            if beats + d > target_beats:
                break
            melody.append((p, d))
            beats += d
        engine_used = "lstm"
    else:
        melody = rule_melody(mood, n_bars=bars, seed=seed)
        engine_used = "rules"
    pm = render_midi(melody, mood, accompaniment=accompaniment, seed=seed)
    return pm, melody, engine_used


def save(pm, name, out_dir=OUTPUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mid, wav = out_dir / f"{name}.mid", out_dir / f"{name}.wav"
    pm.write(str(mid))
    wav.write_bytes(wav_bytes(synthesize(pm)))
    return mid, wav


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mood", choices=MOODS, default="happy")
    p.add_argument("--bars", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--engine", choices=["auto", "lstm", "rules"], default="auto")
    p.add_argument("--seed", type=int)
    p.add_argument("--no-chords", action="store_true")
    a = p.parse_args()
    pm, mel, used = compose(a.mood, a.bars, a.temperature, accompaniment=not a.no_chords,
                            seed=a.seed, engine=a.engine)
    name = f"{a.mood}_{used}_{np.random.randint(10000):04d}"
    mid, wav = save(pm, name)
    print(f"[{used}] {len(mel)} notes -> {mid}\n{' ' * (len(used) + 3)}audio -> {wav}")


if __name__ == "__main__":
    main()
