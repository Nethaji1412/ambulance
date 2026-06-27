import streamlit as st
import numpy as np
import librosa
import librosa.display
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
import tempfile, os

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚑 Ambulance Siren Detector",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
    .hero-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        border: 1px solid #2d6a9f; border-radius: 16px;
        padding: 2rem; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .hero-title { font-size: 2.4rem; font-weight: 700; color: #fff; margin: 0; }
    .hero-subtitle { font-size: 1rem; color: #7eb8e4; margin-top: 0.4rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a2744 0%, #0f1a2e 100%);
        border: 1px solid #2d4a7a; border-radius: 12px;
        padding: 1.2rem 1.5rem; text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    .metric-label { font-size: 0.78rem; color: #7eb8e4; letter-spacing: 0.05em; text-transform: uppercase; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #fff; margin-top: 0.2rem; }
    .metric-sub { font-size: 0.72rem; color: #4a7fa8; margin-top: 0.2rem; }
    .result-siren {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border: 2px solid #ef4444; border-radius: 16px; padding: 1.5rem; text-align: center;
        animation: pulse-red 1.5s infinite; box-shadow: 0 0 30px rgba(239,68,68,0.3);
    }
    .result-non-siren {
        background: linear-gradient(135deg, #14532d 0%, #052e16 100%);
        border: 2px solid #22c55e; border-radius: 16px; padding: 1.5rem; text-align: center;
        box-shadow: 0 0 30px rgba(34,197,94,0.3);
    }
    .result-icon { font-size: 3.5rem; }
    .result-label { font-size: 2rem; font-weight: 700; color: #fff; margin-top: 0.5rem; }
    .result-conf { font-size: 1rem; color: rgba(255,255,255,0.7); margin-top: 0.3rem; }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 20px rgba(239,68,68,0.3); }
        50%       { box-shadow: 0 0 50px rgba(239,68,68,0.7); }
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #7eb8e4;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin-bottom: 0.8rem; border-bottom: 1px solid #2d4a7a; padding-bottom: 0.4rem;
    }
    .info-badge {
        display: inline-block; background: #1e3a5f; border: 1px solid #2d6a9f;
        border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.8rem; color: #7eb8e4; margin: 0.2rem;
    }
    .sidebar-info {
        background: #1a2744; border: 1px solid #2d4a7a; border-radius: 10px;
        padding: 0.8rem 1rem; margin-bottom: 0.8rem; font-size: 0.82rem; color: #a0bdd4;
    }
    .history-item {
        background: #1a2744; border-left: 3px solid #2d6a9f;
        border-radius: 0 8px 8px 0; padding: 0.5rem 0.8rem;
        margin-bottom: 0.5rem; font-size: 0.82rem; color: #a0bdd4;
    }
    .history-siren     { border-left-color: #ef4444; }
    .history-non-siren { border-left-color: #22c55e; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Model path (same folder as this script) ─────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ambulance_siren_model.h5")
DEMO_PATH  = os.path.join(BASE_DIR, "police-siren-sound-effect-240674.mp3")

@st.cache_resource
def load_siren_model():
    return load_model("ambulance_siren_model.h5")

try:
    model = load_siren_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)

# ─── Feature extraction ───────────────────────────────────────────────────────
def extract_features(audio_path):
    signal, sr = librosa.load(audio_path, sr=22050, duration=30)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    return np.mean(mfccs, axis=1), signal, sr

# ─── Plots ────────────────────────────────────────────────────────────────────
LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,26,0.8)",
              font=dict(color="#a0bdd4", size=12), margin=dict(l=40, r=20, t=50, b=40))

def plot_waveform(signal, sr, color="#1d6fa4"):
    times = np.linspace(0, len(signal)/sr, num=len(signal))
    fig = go.Figure(go.Scatter(x=times, y=signal, mode="lines",
                               line=dict(color=color, width=0.8),
                               fill="tozeroy", fillcolor=color+"26"))
    fig.update_layout(title="🎵 Audio Waveform", xaxis_title="Time (s)",
                      yaxis_title="Amplitude", height=260,
                      xaxis=dict(gridcolor="#1e3a5f"),
                      yaxis=dict(gridcolor="#1e3a5f"), **LAYOUT)
    return fig

def plot_spectrogram(signal, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    fig = px.imshow(D, aspect="auto", origin="lower",
                    color_continuous_scale="Blues",
                    labels=dict(x="Time Frames", y="Frequency Bins", color="dB"),
                    title="🎛️ Spectrogram (STFT)")
    fig.update_layout(height=260, **LAYOUT)
    return fig

def plot_mfcc(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    fig = px.imshow(mfcc, aspect="auto", origin="lower",
                    color_continuous_scale="RdBu_r",
                    labels=dict(x="Time Frames", y="MFCC Coefficients", color="Value"),
                    title="📊 MFCC Heatmap (40 Coefficients)")
    fig.update_layout(height=260, **LAYOUT)
    return fig

def plot_mfcc_bar(features):
    colors = [f"hsl({int(200+i*4)},70%,50%)" for i in range(40)]
    fig = go.Figure(go.Bar(x=list(range(1,41)), y=features, marker_color=colors))
    fig.update_layout(title="🔢 Mean MFCC Feature Vector", xaxis_title="MFCC Index",
                      yaxis_title="Mean Value", height=260,
                      xaxis=dict(gridcolor="#1e3a5f"),
                      yaxis=dict(gridcolor="#1e3a5f"), **LAYOUT)
    return fig

def plot_confidence_gauge(confidence, is_siren):
    color = "#ef4444" if is_siren else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=confidence*100,
        title=dict(text="Confidence %", font=dict(color="#a0bdd4", size=14)),
        number=dict(suffix="%", font=dict(color=color, size=28)),
        gauge=dict(axis=dict(range=[0,100], tickcolor="#a0bdd4"),
                   bar=dict(color=color), bgcolor="#1a2744", bordercolor="#2d4a7a",
                   steps=[dict(range=[0,50], color="#1a2744"),
                          dict(range=[50,75], color="#1e3a5f"),
                          dict(range=[75,100], color="#162840")],
                   threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=50))
    ))
    fig.update_layout(height=240, margin=dict(l=30,r=30,t=50,b=20),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#a0bdd4"))
    return fig

def plot_spectral_rolloff(signal, sr):
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    zcr     = librosa.feature.zero_crossing_rate(signal)[0]
    t_r = librosa.frames_to_time(np.arange(len(rolloff)), sr=sr)
    t_z = librosa.frames_to_time(np.arange(len(zcr)),     sr=sr)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Spectral Roll-off","Zero Crossing Rate"],
                        vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=t_r, y=rolloff, line=dict(color="#7eb8e4", width=1),
                             fill="tozeroy", fillcolor="rgba(126,184,228,0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_z, y=zcr,    line=dict(color="#22c55e",  width=1),
                             fill="tozeroy", fillcolor="rgba(34,197,94,0.1)"),  row=2, col=1)
    fig.update_layout(height=320, showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,26,0.8)",
                      font=dict(color="#a0bdd4", size=11),
                      margin=dict(l=50,r=20,t=50,b=40))
    for i in [1,2]:
        fig.update_xaxes(gridcolor="#1e3a5f", row=i, col=1)
        fig.update_yaxes(gridcolor="#1e3a5f", row=i, col=1)
    return fig

def plot_history(history):
    names  = [h["name"][:15]        for h in history]
    confs  = [h["confidence"]*100   for h in history]
    labels = [h["label"]            for h in history]
    colors = ["#ef4444" if l=="🚨 SIREN" else "#22c55e" for l in labels]
    fig = go.Figure(go.Bar(x=names, y=confs, marker_color=colors,
                           text=[f"{c:.1f}%" for c in confs], textposition="outside",
                           textfont=dict(color="#a0bdd4")))
    fig.add_hline(y=50, line_dash="dash", line_color="#7eb8e4", annotation_text="Threshold 50%")
    fig.update_layout(title="📋 Prediction History", xaxis_title="Files",
                      yaxis_title="Confidence (%)", yaxis=dict(range=[0,115], gridcolor="#1e3a5f"),
                      xaxis=dict(gridcolor="#1e3a5f"), height=280,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,26,0.8)",
                      font=dict(color="#a0bdd4", size=11), margin=dict(l=40,r=20,t=50,b=60))
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚑 About")
    st.markdown('<div class="sidebar-info">Deep learning model using 40 MFCC features to classify '
                '<b>ambulance siren</b> vs non-siren audio.</div>', unsafe_allow_html=True)

    st.markdown("### 🧠 Model Architecture")
    for name, detail in [("Input","40 MFCCs"),("Dense","128 · ReLU"),("Dropout","Regularisation"),
                          ("Dense","64 · ReLU"),("Dropout","Regularisation"),
                          ("Dense","32 · ReLU"),("Output","1 · Sigmoid")]:
        st.markdown(f'<div class="sidebar-info"><b>{name}</b>: {detail}</div>', unsafe_allow_html=True)

    st.markdown("### 📂 Model Status")
    if model_loaded:
        st.success("✅ Model loaded")
    else:
        st.error(f"❌ {model_error}")

    if st.session_state.history:
        st.markdown("### 📋 Recent")
        for item in reversed(st.session_state.history[-6:]):
            cls = "history-siren" if item["label"]=="🚨 SIREN" else "history-non-siren"
            st.markdown(f'<div class="history-item {cls}">{item["label"]} · '
                        f'{item["confidence"]*100:.1f}% · <i>{item["name"][:18]}</i></div>',
                        unsafe_allow_html=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
  <p class="hero-title">🚑 Ambulance Siren Detector</p>
  <p class="hero-subtitle">Upload audio — the model classifies it using deep learning on MFCC features</p>
  <span class="info-badge">TensorFlow / Keras</span>
  <span class="info-badge">Librosa</span>
  <span class="info-badge">40 MFCCs</span>
  <span class="info-badge">Binary Classification</span>
</div>""", unsafe_allow_html=True)

# ─── Upload ───────────────────────────────────────────────────────────────────
col_up, col_demo = st.columns([3,1])
with col_up:
    uploaded_file = st.file_uploader("Upload audio (WAV, MP3, OGG, FLAC)",
                                     type=["wav","mp3","ogg","flac"],
                                     label_visibility="collapsed")
with col_demo:
    st.markdown("<br>", unsafe_allow_html=True)
    use_demo = st.button("🎵 Use Demo\n(Police Siren)")

# ─── Resolve source ───────────────────────────────────────────────────────────
audio_path = audio_name = None

if use_demo:
    if os.path.exists(DEMO_PATH):
        audio_path = DEMO_PATH
        audio_name = "police-siren-demo.mp3"
    else:
        st.warning("⚠️ Demo file not found. Place `police-siren-sound-effect-240674.mp3` in the app folder.")

elif uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name
    audio_name = uploaded_file.name

# ─── Prediction ───────────────────────────────────────────────────────────────
if audio_path and model_loaded:
    st.markdown("---")
    st.markdown('<div class="section-header">🔊 Audio Player</div>', unsafe_allow_html=True)
    with open(audio_path, "rb") as af:
        st.audio(af.read())

    with st.spinner("⚙️ Extracting features and running inference…"):
        features, signal, sr = extract_features(audio_path)
        raw_pred = float(model.predict(np.expand_dims(features, 0), verbose=0)[0][0])

    is_siren = raw_pred >= 0.5
    label    = "🚨 SIREN" if is_siren else "✅ NON-SIREN"
    conf     = raw_pred if is_siren else (1 - raw_pred)
    st.session_state.history.append({"name": audio_name, "label": label, "confidence": conf})

    result_class = "result-siren" if is_siren else "result-non-siren"
    st.markdown(f"""
    <div class="{result_class}">
      <div class="result-icon">{"🚨" if is_siren else "✅"}</div>
      <div class="result-label">{label}</div>
      <div class="result-conf">{"Ambulance / Emergency Siren Detected" if is_siren else "No Siren Detected"}</div>
    </div><br>""", unsafe_allow_html=True)

    duration = len(signal)/sr
    rms  = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))

    for col, lbl, val, sub in zip(
        st.columns(5),
        ["Confidence","Raw Score","Duration","Sample Rate","Peak Amp."],
        [f"{conf*100:.1f}%", f"{raw_pred:.4f}", f"{duration:.2f}s", f"{sr//1000}kHz", f"{peak:.4f}"],
        ["Model certainty","Sigmoid output","Audio length",f"{sr} Hz",f"RMS {rms:.4f}"]
    ):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                     f'<div class="metric-value">{val}</div>'
                     f'<div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Signal Visualisations</div>', unsafe_allow_html=True)

    color = "#ef4444" if is_siren else "#22c55e"
    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_waveform(signal, sr, color),    use_container_width=True)
    c2.plotly_chart(plot_spectrogram(signal, sr),         use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(plot_mfcc(signal, sr),                use_container_width=True)
    c4.plotly_chart(plot_mfcc_bar(features),              use_container_width=True)

    c5, c6 = st.columns(2)
    c5.plotly_chart(plot_confidence_gauge(conf, is_siren), use_container_width=True)
    c6.plotly_chart(plot_spectral_rolloff(signal, sr),    use_container_width=True)

    if len(st.session_state.history) > 1:
        st.markdown('<div class="section-header">📋 Prediction History</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_history(st.session_state.history), use_container_width=True)

elif not model_loaded:
    st.error(f"⚠️ Could not load model: {model_error}")
    st.info("Make sure `ambulance_siren_model.h5` is in the same folder as this script.")
else:
    st.info("👆 Upload an audio file above, or click **Use Demo File**.")
    st.markdown('<div class="section-header">ℹ️ How It Works</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, [
        ("1️⃣","Upload Audio","WAV, MP3, OGG, FLAC"),
        ("2️⃣","Feature Extract","40 MFCC via Librosa"),
        ("3️⃣","Inference","Dense neural network"),
        ("4️⃣","Classification","Siren vs Non-Siren"),
        ("5️⃣","Visualisation","6 interactive charts"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="result-icon" style="font-size:2rem">{icon}</div>'
                     f'<div class="metric-value" style="font-size:1rem;margin-top:.4rem">{title}</div>'
                     f'<div class="metric-sub">{desc}</div></div>', unsafe_allow_html=True)
