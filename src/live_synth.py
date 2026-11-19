"""Real-time synthesiser + step sequencer for Face Orchestra.

The face only gives rough gestures; this engine makes them musical:
  * melody notes are always snapped to the mood's scale
  * notes land on a rhythmic grid (16th notes) at the current tempo
  * every bar the "AI harmonist" picks the chord that fits the melody note you are playing
  * a drum groove and bass line follow the mood
  * optional echo: when you close your mouth, the engine answers your phrase a third higher
"""
import time

import numpy as np

from .config import MOOD_STYLE
from .midi_utils import TIMBRES, wav_bytes

FS = 44100
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

DRUMS = {  # 16-step patterns: kick, snare, hat
    "happy": ({0, 8, 10}, {4, 12}, set(range(0, 16, 2))),
    "sad":   ({0}, {8}, {0, 4, 8, 12}),
    "angry": ({0, 3, 6, 8, 11, 14}, {4, 12}, set(range(16))),
    "calm":  ({0}, set(), {8}),
}


def note_name(pitch):
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def scale_pitch(mood, degree, octave_shift=1):
    st = MOOD_STYLE[mood]
    octave, idx = divmod(degree, len(st["scale"]))
    return st["root"] + 12 * (octave + octave_shift) + st["scale"][idx]


def chord_name(mood, degree):
    p = [scale_pitch(mood, degree + k, 0) for k in (0, 2, 4)]
    third, fifth = p[1] - p[0], p[2] - p[0]
    quality = "dim" if fifth == 6 else ("m" if third == 3 else "")
    return NOTE_NAMES[p[0] % 12] + quality


class _Tone:
    """Additive-synthesis voice with attack/decay/release (timbres shared with midi_utils)."""

    def __init__(self, pitch, amp, program, gate, delay):
        harm, self.att, self.dec, self.drive = TIMBRES.get(program, TIMBRES[0])
        f = 440.0 * 2 ** ((pitch - 69) / 12)
        k = np.arange(1, len(harm) + 1)
        ok = f * k < FS / 2
        self.w = (2 * np.pi * f * k[ok])[:, None]
        self.harm = np.asarray(harm)[ok][:, None]
        self.phase = k[ok][:, None].astype(float)
        self.amp, self.gate, self.rel = amp, gate, 0.18
        self.age = -delay  # samples until the note starts

    def render(self, n):
        t = (self.age + np.arange(n)) / FS
        self.age += n
        tt = np.maximum(t, 0)
        wave = (self.harm * np.sin(self.w * tt + self.phase)).sum(0)
        if self.drive:
            wave = np.tanh(self.drive * wave) / np.tanh(self.drive)
        env = np.minimum(1, tt / max(self.att, 1e-3)) * np.exp(-self.dec * tt)
        env *= np.clip(1 - (tt - self.gate) / self.rel, 0, 1) * (t >= 0)
        return self.amp * wave * env, t[-1] > self.gate + self.rel


class _Sample:
    def __init__(self, data, amp, delay):
        self.data, self.amp, self.pos = data, amp, -delay

    def render(self, n):
        out = np.zeros(n)
        start = max(0, -self.pos)
        a, b = max(self.pos, 0), max(self.pos + n, 0)
        chunk = self.data[a:b]
        out[start:start + len(chunk)] = chunk
        self.pos += n
        return self.amp * out, self.pos >= len(self.data)


def _drum_kit():
    rng = np.random.default_rng(7)
    t = np.arange(int(0.35 * FS)) / FS
    kick = np.sin(2 * np.pi * np.cumsum(45 + 90 * np.exp(-t * 30)) / FS) * np.exp(-t * 9)
    ts = t[: int(0.2 * FS)]
    snare = 0.6 * rng.standard_normal(len(ts)) * np.exp(-ts * 22) + 0.5 * np.sin(2 * np.pi * 185 * ts) * np.exp(-ts * 30)
    th = t[: int(0.06 * FS)]
    hat = 0.35 * np.diff(rng.standard_normal(len(th) + 1)) * np.exp(-th * 70)
    return kick, snare, hat


class FaceSynth:
    def __init__(self):
        # ---- controls (written by the vision loop)
        self.mouth, self.brows, self.eyes, self.tilt = 0.0, 0.3, 1.0, 0.0
        self.mood = self.pending_mood = "calm"
        self.harmony, self.drums, self.echo = True, True, False
        # ---- state (read by the HUD)
        self.chord, self.last_pitch, self.last_note_time = "", None, 0.0
        self.step = 0
        self.recording = False
        # ---- internals
        self._voices, self._clock, self._next_step = [], 0, 0.0
        self._gain = 0.0
        self._phrase, self._echo_queue, self._mouth_open = [], [], False
        self._kick, self._snare, self._hat = _drum_kit()
        self._events, self._rec_audio, self._rec_t0 = [], [], 0
        self.stream = None

    # ------------------------------------------------------------------ mapping
    @property
    def tempo(self):
        return 70 + 90 * float(np.clip(self.brows, 0, 1))

    def melody_degree(self):
        return int(round(np.clip(self.tilt, -1, 1) * 7))

    def note_rate(self):
        """Steps between melody notes: wider mouth = faster notes; 0 = silent."""
        m = self.mouth
        return 0 if m < 0.12 else 4 if m < 0.35 else 2 if m < 0.65 else 1

    # ------------------------------------------------------------------ events
    def _play(self, pitch, amp, program, gate_s, delay):
        self._voices.append(_Tone(pitch, amp, program, gate_s, delay))
        if self.recording:
            start = (self._clock + delay - self._rec_t0) / FS
            self._events.append((program, int(pitch), int(min(127, amp * 200)), start, gate_s))

    def _on_step(self, s, delay):
        st = MOOD_STYLE[self.mood]
        step_s = 60.0 / self.tempo / 4
        pos = s % 16

        if pos == 0:  # new bar: apply mood change, pick chord
            self.mood = self.pending_mood
            st = MOOD_STYLE[self.mood]
            if self.harmony:
                n = len(st["scale"])
                m = self.melody_degree() % n
                candidates = [(m - k) % n for k in (0, 2, 4)]  # chords containing the melody note
                preferred = [d for d in candidates if d in st["progression"]]
                deg = (preferred or candidates)[s // 16 % len(preferred or candidates)]
                self.chord = chord_name(self.mood, deg)
                bar = 16 * step_s
                for k in (0, 2, 4):
                    self._play(scale_pitch(self.mood, deg + k, 0), 0.07, st["chord_program"], bar * 0.95, delay)
                self._play(scale_pitch(self.mood, deg, -1), 0.16, 32 if self.mood != "calm" else 89, bar * 0.9, delay)

        if self.drums:
            kick, snare, hat = DRUMS[self.mood]
            for hit, data, amp in ((kick, self._kick, .55), (snare, self._snare, .35), (hat, self._hat, .25)):
                if pos in hit:
                    self._voices.append(_Sample(data, amp, delay))

        rate = self.note_rate()
        if rate and s % rate == 0:
            deg = self.melody_degree()
            pitch = scale_pitch(self.mood, deg)
            self._play(pitch, 0.30 + 0.25 * self.mouth, st["program"], rate * step_s * 0.9, delay)
            self.last_pitch, self.last_note_time = pitch, time.monotonic()
            self._phrase.append((s, deg, rate))
            self._mouth_open = True
        elif not rate and self._mouth_open:
            self._mouth_open = False
            if self.echo and self._phrase:  # answer the phrase a third higher
                phrase = self._phrase[-16:]
                first = phrase[0][0]  # offset from the first ECHOED note, not the first note of a long phrase
                self._echo_queue = [(s + 2 + (ps - first), d + 2, r) for ps, d, r in phrase]
            self._phrase = []

        while self._echo_queue and self._echo_queue[0][0] <= s:
            _, d, r = self._echo_queue.pop(0)
            self._play(scale_pitch(self.mood, d), 0.22, 0 if st["program"] != 0 else 73, r * step_s * 0.9, delay)

    # ------------------------------------------------------------------ audio
    def render(self, n):
        while self._next_step < self._clock + n:  # sample-accurate step scheduling
            self._on_step(self.step, max(0, int(self._next_step - self._clock)))
            self.step += 1
            self._next_step += FS * 60.0 / self.tempo / 4
        out = np.zeros(n)
        alive = []
        for v in self._voices:
            buf, done = v.render(n)
            out += buf
            if not done:
                alive.append(v)
        self._voices = alive
        target = 0.1 + 0.9 * float(np.clip(self.eyes, 0, 1))  # eyes = volume (smooth ramp)
        gain = np.linspace(self._gain, target, n)
        self._gain = target
        out = np.tanh(out * gain).astype(np.float32)  # soft limiter
        self._clock += n
        if self.recording:
            self._rec_audio.append(out.copy())
        return out

    def _callback(self, outdata, frames, _time, _status):
        outdata[:, 0] = self.render(frames)

    def start(self):
        import sounddevice as sd
        self.stream = sd.OutputStream(samplerate=FS, channels=1, blocksize=256,
                                      latency="low", callback=self._callback)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    # ------------------------------------------------------------------ recording
    def start_recording(self):
        self._events, self._rec_audio, self._rec_t0 = [], [], self._clock
        self.recording = True

    def stop_recording(self, out_dir, name):
        """Saves the performance as MIDI + WAV; returns the two paths."""
        import pretty_midi
        self.recording = False
        out_dir.mkdir(parents=True, exist_ok=True)
        pm = pretty_midi.PrettyMIDI()
        insts = {}
        for program, pitch, vel, start, dur in self._events:
            inst = insts.setdefault(program, pretty_midi.Instrument(program=program))
            inst.notes.append(pretty_midi.Note(vel, pitch, start, start + dur))
        pm.instruments = list(insts.values())
        mid, wav = out_dir / f"{name}.mid", out_dir / f"{name}.wav"
        pm.write(str(mid))
        audio = np.concatenate(self._rec_audio) if self._rec_audio else np.zeros(1, np.float32)
        wav.write_bytes(wav_bytes(audio, FS))
        return mid, wav
