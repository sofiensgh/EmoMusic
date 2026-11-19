"""Shared configuration for EmoMusic: paths, moods and musical style rules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

EMOTION_CKPT = MODELS_DIR / "emotion_cnn.pth"
MELODY_CKPT = MODELS_DIR / "melody_lstm.pth"
MELODY_DATA = DATA_DIR / "melodies.pt"

# ---------------------------------------------------------------- moods
MOODS = ["happy", "sad", "angry", "calm"]
MOOD_EMOJI = {"happy": "😄", "sad": "😢", "angry": "😠", "calm": "😌"}

# FER-2013 labels: 0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral
FER_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
# Map to our 4 moods (None = dropped, these classes are ambiguous for music)
FER_TO_MOOD = {0: 2, 1: 2, 2: None, 3: 0, 4: 1, 5: None, 6: 3}

# EMOPIA quadrants (Russell's valence/arousal model) -> mood
#   Q1 high valence / high arousal = happy     Q2 low valence / high arousal = angry
#   Q3 low valence / low arousal   = sad       Q4 high valence / low arousal = calm
EMOPIA_TO_MOOD = {"Q1": 0, "Q2": 2, "Q3": 1, "Q4": 3}

IMG_SIZE = 48

# ---------------------------------------------------------------- music tokens
PITCH_MIN, PITCH_MAX = 36, 96             # C2 .. C7
REST = PITCH_MAX - PITCH_MIN + 1          # extra "pitch" index used for rests
N_PITCH = REST + 1
DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]  # in beats
VOCAB_SIZE = N_PITCH * len(DURATIONS)
SEQ_LEN = 32

# ---------------------------------------------------------------- style rules
# scale = semitone offsets from root; progression = scale degrees (0-based), one chord per bar
MOOD_STYLE = {
    "happy": dict(tempo=124, root=60, scale=[0, 2, 4, 5, 7, 9, 11],
                  program=0, chord_program=0, velocity=95, progression=[0, 5, 3, 4]),
    "sad":   dict(tempo=66, root=57, scale=[0, 2, 3, 5, 7, 8, 10],
                  program=42, chord_program=48, velocity=62, progression=[0, 5, 2, 6]),
    "angry": dict(tempo=144, root=52, scale=[0, 1, 3, 5, 7, 8, 10],
                  program=30, chord_program=33, velocity=115, progression=[0, 1, 0, 6]),
    "calm":  dict(tempo=80, root=62, scale=[0, 2, 4, 7, 9],
                  program=73, chord_program=89, velocity=58, progression=[0, 3, 1, 4]),
}
