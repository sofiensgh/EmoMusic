"""MIDI helpers: melody extraction, tokenisation, mood styling, audio synthesis, piano roll.

A melody is a list of (pitch, beats) tuples; pitch=None means a rest.
"""
import io
import wave

import numpy as np
import pretty_midi

from .config import DURATIONS, MOOD_STYLE, PITCH_MAX, PITCH_MIN, REST

ND = len(DURATIONS)


# ------------------------------------------------------------------ tokens
def quantize_dur(beats):
    return int(np.argmin([abs(beats - d) for d in DURATIONS]))


def encode(pitch, beats):
    p = REST if pitch is None else int(np.clip(pitch, PITCH_MIN, PITCH_MAX)) - PITCH_MIN
    return p * ND + quantize_dur(beats)


def decode(token):
    p, d = divmod(int(token), ND)
    return (None if p == REST else p + PITCH_MIN), DURATIONS[d]


def tokenize(melody):
    return [encode(p, d) for p, d in melody]


def detokenize(tokens):
    return [decode(t) for t in tokens]


def transpose(melody, semitones):
    return [(None if p is None else p + semitones, d) for p, d in melody]


# ------------------------------------------------------------------ extraction
def extract_melody(midi_path, max_rest=4.0):
    """Skyline melody: at every onset keep the highest note, timing in beats."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    _, tempi = pm.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) else 120.0
    spb = 60.0 / tempo
    notes = sorted((n for inst in pm.instruments if not inst.is_drum for n in inst.notes),
                   key=lambda n: (n.start, -n.pitch))
    mel = []
    for n in notes:
        if mel and n.start - mel[-1].start < spb * 0.125:
            continue  # same onset: a higher note was already taken
        mel.append(n)

    seq = []
    for i, n in enumerate(mel):
        nxt = mel[i + 1].start if i + 1 < len(mel) else n.end
        ioi = (nxt - n.start) / spb
        dur = DURATIONS[quantize_dur(min(n.end - n.start, nxt - n.start) / spb)]
        seq.append((n.pitch, dur))
        gap = ioi - dur
        if gap >= 0.25:
            seq.append((None, DURATIONS[quantize_dur(min(gap, max_rest))]))
    return seq


# ------------------------------------------------------------------ styling
def snap_to_scale(pitch, root, scale):
    best = None
    for delta in (0, -1, 1, -2, 2):
        if (pitch + delta - root) % 12 in scale:
            best = pitch + delta
            break
    return best if best is not None else pitch


def _scale_pitch(root, scale, degree):
    octave, idx = divmod(degree, len(scale))
    return root + 12 * octave + scale[idx]


def render_midi(melody, mood, accompaniment=True, humanize=True, seed=None):
    """Turn a melody into a full MIDI piece using the mood's tempo, scale, instruments and chords."""
    st = MOOD_STYLE[mood]
    rng = np.random.default_rng(seed)
    spb = 60.0 / st["tempo"]
    pm = pretty_midi.PrettyMIDI(initial_tempo=st["tempo"])

    # move the melody into a comfortable register around root+12
    pitched = [p for p, _ in melody if p is not None]
    shift = 0
    if pitched:
        target = st["root"] + 12
        shift = int(round((target - np.median(pitched)) / 12.0)) * 12

    lead = pretty_midi.Instrument(program=st["program"], name=f"{mood} lead")
    t = 0.0
    for p, d in melody:
        if p is not None:
            pitch = int(np.clip(snap_to_scale(p + shift, st["root"], st["scale"]), 21, 108))
            vel = st["velocity"] + (int(rng.integers(-8, 9)) if humanize else 0)
            lead.notes.append(pretty_midi.Note(int(np.clip(vel, 1, 127)), pitch, t, t + d * spb * 0.95))
        t += d * spb
    pm.instruments.append(lead)
    total_beats = t / spb

    if accompaniment and total_beats > 0:
        chords = pretty_midi.Instrument(program=st["chord_program"], name=f"{mood} chords")
        bass = pretty_midi.Instrument(program=32 if mood != "calm" else 89, name="bass")
        n_bars = int(np.ceil(total_beats / 4))
        vel = max(30, st["velocity"] - 35)
        for bar in range(n_bars):
            deg = st["progression"][bar % len(st["progression"])]
            start = bar * 4 * spb
            tones = [_scale_pitch(st["root"] - 12, st["scale"], deg + k) for k in (0, 2, 4)]
            bass_pitch = _scale_pitch(st["root"] - 24, st["scale"], deg)
            if mood == "happy":      # chord on beats 1 and 3
                hits = [(0, 2), (2, 2)]
            elif mood == "angry":    # driving eighth notes
                hits = [(i * 0.5, 0.5) for i in range(8)]
            else:                    # sustained pads
                hits = [(0, 4)]
            for off, length in hits:
                s, e = start + off * spb, start + (off + length) * spb * 0.97
                if mood != "angry":
                    for tone in tones:
                        chords.notes.append(pretty_midi.Note(vel, tone, s, e))
                else:  # power chord: root + fifth
                    for tone in (tones[0], tones[0] + 7):
                        chords.notes.append(pretty_midi.Note(vel, tone, s, e))
            bass.notes.append(pretty_midi.Note(vel + 10, bass_pitch, start, start + 4 * spb * 0.97))
        pm.instruments += [chords, bass]
    return pm


# ------------------------------------------------------------------ audio
# timbre per General-MIDI program: harmonics, attack(s), decay rate, distortion drive
TIMBRES = {
    0: ([1, .55, .3, .18, .1, .05], .005, 2.5, 0),    # piano
    42: ([1, .8, .6, .45, .3, .2], .09, .25, 0),     # cello
    48: ([1, .7, .5, .35, .2], .25, .1, 0),          # strings
    30: ([1, .6, .45, .3, .25, .2], .005, .8, 4),     # distortion guitar
    33: ([1, .5, .2], .01, 1.0, 1.5),                # electric bass
    32: ([1, .4, .15], .01, 1.2, 0),                 # acoustic bass
    73: ([1, .25, .08, .03], .06, .3, 0),            # flute
    89: ([1, .5, .3, .2], .4, .05, 0),               # warm pad
}


def synthesize(pm, fs=22050):
    """Small additive synthesiser so the app makes nice sound with no external soundfont."""
    end = pm.get_end_time() + 1.0
    out = np.zeros(int(end * fs) + 1, dtype=np.float64)
    for inst in pm.instruments:
        harm, attack, decay, drive = TIMBRES.get(inst.program, TIMBRES[0])
        gain = 0.35 if "lead" in inst.name else 0.18
        for n in inst.notes:
            length = n.end - n.start + 0.25
            t = np.arange(int(length * fs)) / fs
            f = 440.0 * 2 ** ((n.pitch - 69) / 12)
            w = np.zeros_like(t)
            for k, a in enumerate(harm):
                if f * (k + 1) < fs / 2:
                    w += a * np.sin(2 * np.pi * f * (k + 1) * t + k)
            if drive:
                w = np.tanh(drive * w) / np.tanh(drive)
            env = np.minimum(1.0, t / max(attack, 1e-3)) * np.exp(-decay * t)
            rel = n.end - n.start
            env *= np.clip(1 - (t - rel) / 0.25, 0, 1)  # release
            s = int(n.start * fs)
            out[s:s + len(t)] += gain * (n.velocity / 127) * w * env
    peak = np.abs(out).max()
    return (out / peak * 0.9 if peak > 0 else out).astype(np.float32)


def wav_bytes(audio, fs=22050):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


# ------------------------------------------------------------------ visual
def piano_roll_figure(pm, title=""):
    import matplotlib.pyplot as plt
    colors = ["#e4572e", "#29335c", "#a8c686", "#f3a712"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for i, inst in enumerate(pm.instruments):
        bars = [(n.start, n.end - n.start) for n in inst.notes]
        for (s, d), n in zip(bars, inst.notes):
            ax.broken_barh([(s, d)], (n.pitch - 0.4, 0.8), color=colors[i % 4], alpha=0.9 if i == 0 else 0.45)
        ax.plot([], [], color=colors[i % 4], lw=6, label=inst.name)
    ax.set_xlabel("time (s)"); ax.set_ylabel("MIDI pitch"); ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
