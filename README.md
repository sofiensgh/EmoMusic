# 🎵 EmoMusic: mood-based music generator

A **CNN** reads your facial emotion from the webcam, then a **mood-conditioned LSTM** composes a melody
that matches it. The melody is arranged with mood-specific tempo, scale, instruments and chords, then
exported as MIDI and WAV.

```
Webcam ─► Face detection (OpenCV) ─► Emotion CNN ─► mood ─► LSTM melody ─► style rules ─► MIDI / WAV
            Haar cascade               Chapter 2               Chapter 3     tempo, scale,
                                                                             chords, timbre
```

## Project layout
```
EmoMusic/
├── app.py                     Streamlit demo (camera / upload / manual mood → song)
├── live.py                    real-time webcam window (SPACE = compose, Q = quit)
├── orchestra.py               🎻 Face Orchestra: play music live with your face
├── requirements.txt
├── data/                      datasets go here (not included)
├── models/                    trained weights (.pth)
├── output/                    generated songs, training plots, survey clips
└── src/
    ├── config.py              moods, paths, token vocabulary, musical style per mood
    ├── emotion_dataset.py     FER-2013 loader (CSV or image folders) + augmentation
    ├── emotion_cnn.py         VGG-style CNN  +  ResNet18 transfer-learning option
    ├── train_emotion.py       training, class weights, curves, confusion matrix
    ├── face.py                face detection + prediction (with flip test-time augmentation)
    ├── midi_utils.py          melody extraction, tokens, arrangement, synthesiser, piano roll
    ├── rule_based.py          hand-written melody generator (baseline + fallback)
    ├── build_music_dataset.py MIDI → token sequences (EMOPIA / your folders / synthetic)
    ├── melody_lstm.py         mood-conditioned LSTM + sampling (temperature, top-k)
    ├── train_melody.py        LSTM training (grad clipping, perplexity, curves)
    ├── generate.py            compose(mood) → MIDI + WAV  (CLI too)
    ├── survey.py              listening test for the report
    ├── face_controls.py       MediaPipe landmarks/blendshapes → mouth, brows, eyes, tilt
    └── live_synth.py          real-time synth + sequencer + auto-harmony (Face Orchestra)
```

## Moods
FER-2013 has 7 emotions; they are grouped into 4 moods that sound clearly different:

| Mood | FER classes | EMOPIA | Tempo | Scale | Lead | Accompaniment |
|---|---|---|---|---|---|---|
| 😄 happy | happy | Q1 | 124 | major | piano | chords on beats 1 & 3 |
| 😢 sad | sad | Q3 | 66 | natural minor | cello | sustained strings |
| 😠 angry | angry + disgust | Q2 | 144 | phrygian | distortion guitar | 8th-note power chords |
| 😌 calm | neutral | Q4 | 80 | pentatonic | flute | warm pad |

*fear* and *surprise* are dropped because they don't map cleanly to one musical mood.

---

## Step 0: Setup
Use **Python 3.10–3.12** (or Kaggle/Colab with a GPU for training).
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```
For a GPU, install torch from https://pytorch.org first (pick your CUDA version).

## Step 1: Emotion dataset
Download **FER-2013** from Kaggle (either version works):
* `data/fer2013.csv` (columns `emotion, pixels, Usage`), or
* `data/fer2013/train/<emotion>/*.jpg` and `data/fer2013/test/<emotion>/*.jpg`

## Step 2: Train the emotion CNN
```bash
python -m src.train_emotion --smoke-test                    # 10-second pipeline check
python -m src.train_emotion --data data/fer2013.csv --epochs 40
python -m src.train_emotion --data data/fer2013 --arch resnet18 --epochs 20   # transfer learning
```
Output: `models/emotion_cnn.pth`, `output/emotion_training_curves.png`, `output/emotion_confusion_matrix.png`.
A good target is **65–75 % validation accuracy** on 4 moods. On a GPU it takes ~15 min, on a CPU over an hour.

## Step 3: Music dataset
Download **EMOPIA** (https://zenodo.org/records/5090631) and unzip it into `data/`.
```bash
python -m src.build_music_dataset --emopia data/EMOPIA_1.0/midis
# optional extra MIDI folders:  --labeled data/my_songs_sad:sad
# no data yet? synthetic melodies to test everything:
python -m src.build_music_dataset --synthetic 300
```
Each piece is reduced to its top melody line (skyline algorithm), split into tokens
(`pitch × duration`, 496 possible tokens), and augmented by transposing ±3 semitones.

## Step 4: Train the LSTM
```bash
python -m src.train_melody --epochs 30
```
Output: `models/melody_lstm.pth` and `output/melody_training_curves.png`. The model is
Embedding(token) ‖ Embedding(mood) → 2-layer LSTM(256) → softmax over the next token.

## Step 5: Compose from the command line
```bash
python -m src.generate --mood sad --bars 8 --temperature 0.9
python -m src.generate --mood angry --engine rules     # the hand-written baseline
```

## Step 6: Run the app
```bash
streamlit run app.py      # web demo
python live.py            # real-time webcam window
```
The app works even before training: without the CNN you pick the mood manually, and without the
LSTM it uses the rule-based composer.

## Step 7: Evaluation for the report
1. **CNN**: accuracy, training curves, confusion matrix (generated automatically).
2. **LSTM**: loss and perplexity curves, plus a comparison of the LSTM against the rule-based baseline.
3. **Listening test**:
   ```bash
   python -m src.survey make --n 12
   # play output/survey/clip_XX.wav to friends and collect answers.csv with columns: clip,guess
   python -m src.survey score answers.csv
   ```
   Chance level is 25 %. Anything clearly above that shows the music really carries the mood.

---

# 🎻 Face Orchestra: you are the conductor

A real-time mode where your face *plays* the music instead of only choosing a mood.

```bash
python orchestra.py                  # webcam + speakers
python orchestra.py --cnn            # mood comes from your trained emotion CNN
python orchestra.py --demo           # no webcam: a "ghost face" plays by itself
python orchestra.py --render-test 20 # offline check, writes output/orchestra_demo.wav
```

| Your face | Controls | How |
|---|---|---|
| 👄 mouth open | **melody** | closed = silence, a little = quarter notes, wide = 16th notes |
| 🤨 eyebrows raised | **tempo** | 70 → 160 BPM |
| 👀 eyes open | **volume** | close your eyes to fade out |
| ↔️ head tilt | **pitch** | tilt left = low notes, tilt right = high notes |
| 😄 😢 😠 😌 expression | **mood** | changes the scale, instruments and drum groove |

**The "AI fills in the harmony" part**, which makes any face sound musical:
* every note is snapped to the mood's scale and to a 16th-note rhythmic grid
* each bar, the harmonist picks the chord that contains the note you are playing (prefers the mood's progression)
* bass and drums follow the mood automatically
* **echo mode (E):** when you close your mouth, the engine answers your phrase a third higher (call and response)

**Keys:** `C` calibrate on a neutral face · `H` harmony · `D` drums · `E` echo ·
`M` lock a mood · `R` record (saves MIDI and WAV in `output/`) · `Q` quit

**How it works**
* `src/face_controls.py` uses MediaPipe FaceLandmarker: 478 face landmarks plus 52 "blendshapes"
  (jawOpen, browInnerUp, eyeBlink, mouthSmile…). They are smoothed and calibrated into the 4 controls.
  The model file (~3.7 MB) downloads automatically to `models/` on the first run.
* `src/live_synth.py` is a real-time additive synthesiser with a sample-accurate step sequencer,
  auto-harmony, a drum kit and recording. It renders about 38× faster than real time on a CPU.
* Mood comes either from simple blendshape rules (smile, frown, brow-down) or, with `--cnn`, from the
  emotion CNN you trained in Step 2, which links the two halves of the project.

**Demo tips:** sit facing a light source, press `C` with a relaxed face, start with drums off,
then open your mouth slowly and tilt your head like you're "singing" with it.

## Ideas to go further
* Continuous mode: re-compose every N seconds as your face changes
* Run on a **Jetson Nano** as an emotion music box
* Replace the LSTM with a small Transformer and compare
* Use valence/arousal regression instead of 4 classes for smooth mood transitions
