import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import tempfile, os, io
import soundfile as sf

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ambulance Siren Detector",
    page_icon="🚑",
    layout="centered",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background-color: #0f0f1a; }

    h1 { color: #ff4b4b !important; font-size: 2.4rem !important; }
    h3 { color: #f0f0f0 !important; }

    .result-box {
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 20px 0;
    }
    .siren {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
        border: 2px solid #ff4b4b;
        color: #ff4b4b;
    }
    .no-siren {
        background: linear-gradient(135deg, #00c85322, #00c85344);
        border: 2px solid #00c853;
        color: #00c853;
    }
    .info-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        color: #ccc;
        font-size: 0.9rem;
    }
    .badge {
        display: inline-block;
        background: #252545;
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 0.8rem;
        margin: 4px 2px;
        color: #aaa;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = keras.models.load_model("D:\\ML-\\ambulance_siren_model.h5")
    return model

# ─────────────────────────────────────────────
# Feature extraction – 40 MFCCs (matches input shape)
# ─────────────────────────────────────────────
def extract_features(audio_path: str, n_mfcc: int = 40) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = np.mean(mfccs.T, axis=0)          # shape (40,)
    return features.reshape(1, -1)               # shape (1, 40)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("# 🚑 Ambulance Siren Detector")
st.markdown(
    "Upload an audio clip and the model will tell you whether an **ambulance siren** is present."
)

# ── Model info card ──────────────────────────
with st.expander("ℹ️ Model Architecture", expanded=False):
    st.markdown("""
<div class="info-card">
<b>Type:</b> Keras Sequential (binary classifier)<br>
<b>Input:</b> 40 MFCC features (mean-pooled over time)<br>
<b>Architecture:</b>
<span class="badge">Dense 128 · ReLU</span>
<span class="badge">Dropout 0.3</span>
<span class="badge">Dense 64 · ReLU</span>
<span class="badge">Dropout 0.3</span>
<span class="badge">Dense 32 · ReLU</span>
<span class="badge">Dense 1 · Sigmoid</span><br><br>
<b>Loss:</b> Binary Cross-Entropy &nbsp;|&nbsp; <b>Optimizer:</b> Adam (lr=0.001) &nbsp;|&nbsp; <b>Metric:</b> Accuracy
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── File uploader ────────────────────────────
uploaded = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    help="Supports WAV, MP3, OGG, FLAC, M4A",
)

# ── Threshold slider ─────────────────────────
threshold = st.slider(
    "Detection threshold",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="Probability above this → Siren detected",
)

if uploaded:
    # Save to temp file
    suffix = os.path.splitext(uploaded.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    with st.spinner("Extracting features and running inference…"):
        try:
            model = load_model()
            features = extract_features(tmp_path)
            prob = float(model.predict(features, verbose=0)[0][0])
            is_siren = prob >= threshold

            # ── Result ───────────────────────
            if is_siren:
                st.markdown(f"""
<div class="result-box siren">
🚨 Ambulance Siren DETECTED<br>
<span style="font-size:1rem;font-weight:400;opacity:0.85">
Confidence: {prob*100:.1f}%
</span>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="result-box no-siren">
✅ No Siren Detected<br>
<span style="font-size:1rem;font-weight:400;opacity:0.85">
Confidence: {(1-prob)*100:.1f}%
</span>
</div>""", unsafe_allow_html=True)

            # ── Probability bar ───────────────
            st.markdown("### 📊 Prediction probability")
            col1, col2 = st.columns(2)
            col1.metric("Siren probability", f"{prob*100:.2f}%")
            col2.metric("No-siren probability", f"{(1-prob)*100:.2f}%")
            st.progress(prob)

            # ── MFCC preview ──────────────────
            with st.expander("🔬 MFCC Feature Vector (40 values)", expanded=False):
                st.bar_chart(features[0])

        except Exception as e:
            st.error(f"Error during inference: {e}")
        finally:
            os.unlink(tmp_path)

else:
    st.info("👆 Upload a `.wav`, `.mp3`, or other audio file to get started.")

st.markdown("---")
st.caption("Model: `ambulance_siren_model.h5` · Built with TensorFlow/Keras · Features: 40-dim MFCC")
