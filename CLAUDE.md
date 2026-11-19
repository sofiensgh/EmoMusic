# EmoMusic: project context for Claude

## Who / what
- Student project for the **NVIDIA DLI deep-learning course** (CNN = Chapter 2, RNN = Chapter 3, plus autoencoders, diffusion, BERT, RAPIDS and Jetson Nano in the wider course). Course material is in the parent folder `../`.
- Goal: a **creative, attractive demo** for a jury. Two parts:
  1. **EmoMusic**: webcam face → CNN emotion → mood → mood-conditioned LSTM composes a melody → arranged to MIDI/WAV.
  2. **Face Orchestra**: real-time instrument played with the face (mouth = melody, eyebrows = tempo, eyes = volume, head tilt = pitch, expression = mood), with AI auto-harmony.
- **Working preference:** the user likes clear step-by-step explanations. **Don't modify files or run long jobs unless the user asks.** Explain first, act when told.

## Environment
- Windows 11, PowerShell. System Python is **3.14**; the project venv is `.venv` (Python 3.14, **CPU-only torch**, no GPU locally).
- Activate: `.venv\Scripts\Activate.ps1` (or call `.venv\Scripts\python.exe` directly).
- Run everything from the `EmoMusic/` folder as modules: `python -m src.xxx`.
- Installed: torch, torchvision, opencv-python, numpy, pandas, matplotlib, pillow, pretty_midi, streamlit 1.67, mediapipe 1.2.3 (**Tasks API only, no `mp.solutions`**), sounddevice (output device: Realtek speakers).
- For GPU training, the suggested route is a Kaggle notebook (T4).

## Layout
```
app.py                 Streamlit demo: camera/upload/manual mood → compose → audio, piano roll, MIDI/WAV download
live.py                OpenCV webcam window; smoothed CNN mood; SPACE composes + plays (winsound)
orchestra.py           Face Orchestra (OpenCV window + HUD panel). Flags: --cnn, --demo, --render-test SECONDS, --camera N
src/config.py          paths, MOODS=[happy,sad,angry,calm], FER/EMOPIA→mood maps, token vocab, MOOD_STYLE (tempo/root/scale/GM programs/progression)
src/emotion_dataset.py FER-2013 loader (CSV or train/test image folders), drops fear+surprise, augmentation (flip, affine, jitter)
src/emotion_cnn.py     EmotionCNN (4 VGG blocks 32→256, 48x48 gray) + ResNetEmotion (resnet18 transfer, upsample to 96)
src/train_emotion.py   AdamW + OneCycle, class-weighted CE + label smoothing, saves best by val acc, plots curves + confusion matrix
src/face.py            Haar-cascade face detection + EmotionDetector (flip TTA) + draw()
src/midi_utils.py      skyline melody extraction, tokens (pitch×duration), render_midi (scale snap, chords, bass), additive synth (TIMBRES), wav_bytes, piano roll
src/rule_based.py      rule-based melody generator (baseline, fallback, synthetic data)
src/build_music_dataset.py  EMOPIA (Q1..Q4 filename prefix) / labeled folders / --synthetic → data/melodies.pt (sequences, moods, pieces), ±3 semitone transposition
src/melody_lstm.py     MelodyLSTM: token emb(128) ‖ mood emb(16) → 2-layer LSTM(256) → vocab; sample() with temperature + top-k
src/train_melody.py    windows of SEQ_LEN=32, split by PIECE (avoids transposition leakage), grad clip 1.0, cosine LR
src/generate.py        compose(mood, bars, temperature, engine=auto|lstm|rules) → PrettyMIDI; save() → .mid + .wav; CLI
src/survey.py          listening test: `make --n 12` creates clips + answer key; `score answers.csv`
src/face_controls.py   MediaPipe FaceLandmarker (VIDEO mode, blendshapes) → Controls(mouth, brows, eyes, tilt, smile, frown, brow_down); calibrate(); auto-downloads models/face_landmarker.task
src/live_synth.py      FaceSynth: real-time additive synth + sample-accurate 16-step sequencer, auto-harmony (chord containing the melody note), drums per mood, echo (call/response a third higher), recording to MIDI+WAV; sounddevice stream, blocksize 256
```

## Key design decisions
- **4 moods** from FER's 7: happy; sad; angry (angry + disgust); calm (neutral). Fear and surprise are dropped.
- EMOPIA quadrants: Q1 → happy, Q2 → angry, Q3 → sad, Q4 → calm.
- Token = pitch index (36..96 plus REST) × 8 durations (0.25..4 beats) → vocabulary of 496.
- The LSTM produces raw notes; `render_midi` enforces the mood's scale, tempo, instruments and chord progression, so output always sounds "in mood".
- Audio comes from a home-made additive synthesiser (no soundfont/fluidsynth needed). GM program numbers pick the timbre in `TIMBRES`.
- Face Orchestra mood: blendshape rules by default (smile > 0.45 happy, browDown > 0.4 angry, frown > 0.25 sad, else calm), held 1.2 s before switching and applied at the bar start. `--cnn` uses the trained CNN instead.
- Face Orchestra mappings: mouth → note rate (rest / 1/4 / 1/8 / 1/16), brows → 70–160 BPM, eyes → gain 0.1–1.0, tilt ±25° → scale degree ±7.

## Data (in place)
- `data/fer2013/{train,test}/{angry,disgust,fear,happy,neutral,sad,surprise}/*.jpg`: Kaggle msambare/fer2013, 35,887 images.
- `data/EMOPIA_1.0/midis/*.mid`: 1,078 files (Zenodo record 5090631).
- `data/melodies.pt`: built from EMOPIA, 1,073 pieces → 7,511 sequences, moods balanced.

## Model status (as of 2026-11-19)
- Training was started from a Claude desktop session: first `train_melody --epochs 30` (about 180k windows/epoch on CPU, roughly 4 min/epoch), then `train_emotion --data data/fer2013 --epochs 40`. Logs: `output/log_melody.txt`, `output/log_emotion.txt`. **Check these logs to see whether training finished**; if it was interrupted, re-run the commands.
- `models/melody_lstm.pth`: overwritten with the best epoch during training (it was a synthetic-data placeholder before).
- `models/emotion_cnn.pth`: created by the CNN training. Until it exists, the camera/upload tabs and `live.py` can't run, while `orchestra.py` works without `--cnn`.
- `models/smoke_emotion.pth`: pipeline test only, safe to delete.
- Targets: CNN val accuracy about 65–75 %; LSTM val perplexity below about 15.

## Testing status
- Verified: all training scripts (smoke runs), dataset build, generation for all moods, the Streamlit app headless (AppTest: compose works, no exceptions), orchestra `--render-test` (offline audio, about 38× real time), HUD rendering, MediaPipe Image API.
- **Not verified yet:** the live webcam path (FaceTracker on real frames, landmark model download), live speaker output, `live.py`, and the camera tab with a trained CNN.

## Common commands
```bash
python -m streamlit run app.py
python orchestra.py            # or --demo, --cnn, --render-test 20
python live.py
python -m src.generate --mood sad --bars 8 --temperature 0.9 --engine lstm
python -m src.train_emotion --data data/fer2013 --epochs 40      # --arch resnet18 for transfer learning
python -m src.build_music_dataset --emopia data/EMOPIA_1.0/midis
python -m src.train_melody --epochs 30
python -m src.survey make --n 12
```

## Possible next steps
- Evaluate the trained models (plots in `output/`), compare the LSTM against the rules baseline, and run the listening survey.
- Live-test the webcam parts and fix any issues.
- Report and slides; demo video.
- Ideas: continuous re-composition mode, valence/arousal regression, Transformer vs LSTM, Jetson Nano deployment.
