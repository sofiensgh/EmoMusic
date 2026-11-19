"""EmoMusic - Streamlit demo.

    streamlit run app.py
"""
import random

import cv2
import numpy as np
import streamlit as st

from src.config import EMOTION_CKPT, MELODY_CKPT, MOOD_EMOJI, MOOD_STYLE, MOODS
from src.generate import compose
from src.midi_utils import piano_roll_figure, synthesize, wav_bytes

st.set_page_config(page_title="EmoMusic", page_icon="🎵", layout="wide")


@st.cache_resource
def get_detector():
    from src.face import EmotionDetector
    return EmotionDetector()


st.title("🎵 EmoMusic")
st.caption("Your face → a CNN reads your mood → an LSTM composes music for it")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Composer settings")
    bars = st.slider("Length (bars)", 4, 32, 8, 4)
    temperature = st.slider("Creativity (temperature)", 0.3, 2.0, 1.0, 0.1,
                            help="Low = safe and repetitive, high = surprising and chaotic")
    accompaniment = st.toggle("Chords and bass", value=True)
    engines = ["auto", "lstm", "rules"] if MELODY_CKPT.exists() else ["rules"]
    engine = st.selectbox("Melody engine", engines,
                          help="lstm = trained neural network, rules = hand-written baseline")
    st.divider()
    st.markdown("**Model status**")
    st.write(("✅" if EMOTION_CKPT.exists() else "❌") + " Emotion CNN")
    st.write(("✅" if MELODY_CKPT.exists() else "⚠️") + " Melody LSTM" +
             ("" if MELODY_CKPT.exists() else " (using rules)"))

left, right = st.columns([1, 1.3])

# ---------------------------------------------------------------- mood input
with left:
    st.subheader("1 · How do you feel?")
    tabs = st.tabs(["📷 Camera", "🖼️ Upload photo", "🎚️ Pick a mood"])
    detected = None
    for tab, kind in zip(tabs[:2], ["camera", "upload"]):
        with tab:
            if not EMOTION_CKPT.exists():
                st.info("Train the emotion CNN first (`python -m src.train_emotion`). "
                        "Meanwhile, use **Pick a mood**.")
                continue
            img = (st.camera_input("Make a face!") if kind == "camera"
                   else st.file_uploader("Photo with a face", type=["jpg", "jpeg", "png"]))
            if img is not None:
                bgr = cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                from src.face import draw
                res = get_detector().predict(bgr)
                if res is None:
                    st.warning("No face found - try more light or face the camera.")
                else:
                    detected = res
                    st.image(cv2.cvtColor(draw(bgr, res), cv2.COLOR_BGR2RGB), width="stretch")
                    st.bar_chart({m: res["probs"][m] for m in MOODS}, horizontal=True)
    with tabs[2]:
        manual = st.radio("Mood", MOODS, horizontal=True,
                          format_func=lambda m: f"{MOOD_EMOJI[m]} {m}")
        use_manual = st.checkbox("Use this mood even when a face is detected", value=False)

    mood = manual if (use_manual or detected is None) else detected["mood"]
    style = MOOD_STYLE[mood]
    st.markdown(f"### {MOOD_EMOJI[mood]} Mood: **{mood}**")
    st.caption(f"{style['tempo']} BPM · scale {style['scale']} · GM instrument #{style['program']}")

# ---------------------------------------------------------------- composer
with right:
    st.subheader("2 · Compose")
    if st.button("🎼 Compose my song", type="primary", width="stretch"):
        seed = random.randint(0, 10**6)
        with st.spinner("The LSTM is composing..."):
            pm, melody, used = compose(mood, bars, temperature, accompaniment=accompaniment,
                                       seed=seed, engine=engine)
            audio = wav_bytes(synthesize(pm))
        st.session_state["song"] = (pm, melody, used, mood, audio, seed)

    if "song" in st.session_state:
        pm, melody, used, smood, audio, seed = st.session_state["song"]
        st.success(f"{MOOD_EMOJI[smood]} {smood} piece · {len([n for n in melody if n[0]])} notes · "
                   f"engine: **{used}** · {pm.get_end_time():.1f}s")
        st.audio(audio, format="audio/wav")
        st.pyplot(piano_roll_figure(pm, f"{smood} - piano roll"))
        import io
        buf = io.BytesIO()
        pm.write(buf)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ MIDI", buf.getvalue(), f"emomusic_{smood}_{seed}.mid", "audio/midi",
                           width="stretch")
        c2.download_button("⬇️ WAV", audio, f"emomusic_{smood}_{seed}.wav", "audio/wav",
                           width="stretch")
    else:
        st.info("Pick or show a mood, then press **Compose**.")
