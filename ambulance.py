import streamlit as st
import numpy as np
import cv2
import librosa
import tempfile, os, time, io
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ambulance Detector — Vision + Audio",
    page_icon="🚑",
    layout="wide",
)

st.markdown("""
<style>
  .stApp, .main { background: #0a0a14; color: #e0e0e0; }

  h1  { color: #ff4444 !important; font-size: 2.2rem !important; }
  h3  { color: #f0f0f0 !important; }

  .metric-card {
    background: #141428;
    border: 1px solid #2a2a4a;
    border-radius: 14px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 10px;
  }
  .metric-label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 1.9rem; font-weight: 700; margin-top: 4px; }

  .alert-box {
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 14px 0;
    animation: pulse 1s infinite alternate;
  }
  @keyframes pulse { from { opacity:1; } to { opacity:0.75; } }
  .alert-danger { background:#ff000018; border:2px solid #ff4444; color:#ff4444; }
  .alert-safe   { background:#00c85318; border:2px solid #00c853; color:#00c853; }
  .alert-warn   { background:#ffb30018; border:2px solid #ffb300; color:#ffb300; }

  .frame-log {
    background:#0e0e20; border:1px solid #2a2a4a; border-radius:10px;
    padding:12px; font-size:0.78rem; font-family:monospace;
    max-height:300px; overflow-y:auto; color:#aaa;
  }
  .log-alert  { color:#ff4444; font-weight:bold; }
  .log-warn   { color:#ffb300; }
  .log-safe   { color:#00c853; }

  .tag {
    display:inline-block; border-radius:6px; padding:2px 10px;
    font-size:0.75rem; font-weight:600; margin:2px;
  }
  .tag-red  { background:#ff444422; border:1px solid #ff4444; color:#ff4444; }
  .tag-grn  { background:#00c85322; border:1px solid #00c853; color:#00c853; }
  .tag-yel  { background:#ffb30022; border:1px solid #ffb300; color:#ffb300; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading siren audio model…")
def load_audio_model():
    return keras.models.load_model("ambulance_siren_model.h5")

@st.cache_resource(show_spinner="Loading YOLOv8 object detector…")
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")   # auto-downloads ~6 MB on first run

# COCO vehicle class IDs that could be an ambulance (car=2, bus=5, truck=7)
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def extract_mfcc_segment(y: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    if len(y) < 512:
        y = np.pad(y, (0, 512 - len(y)))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)


def is_ambulance_by_color(frame: np.ndarray, box) -> bool:
    """Heuristic: check if the bounding-box region contains white/yellow ambulance colors."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # White mask
    white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))
    # Yellow mask (ambulance markings)
    yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    # Red/orange emergency lights
    red1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 120, 100]), np.array([180, 255, 255]))
    total = roi.size // 3
    score = (cv2.countNonZero(white) / total +
             cv2.countNonZero(yellow) / total * 0.5 +
             (cv2.countNonZero(red1) + cv2.countNonZero(red2)) / total * 0.3)
    return score > 0.25


def draw_box(frame, box, label, color, conf):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def overlay_status(frame, visual_detected: bool, audio_detected: bool, alert: bool):
    h, w = frame.shape[:2]
    if alert:
        label = "🚨 AMBULANCE + SIREN DETECTED"
        color = (0, 60, 255)
        bg    = (0, 0, 180)
    elif visual_detected:
        label = "🚑 Ambulance visible — no siren"
        color = (0, 165, 255)
        bg    = (0, 80, 180)
    elif audio_detected:
        label = "🔊 Siren heard — no ambulance"
        color = (0, 200, 255)
        bg    = (0, 100, 160)
    else:
        label = "✅ Clear"
        color = (0, 200, 80)
        bg    = (0, 100, 40)

    bar_h = 36
    cv2.rectangle(frame, (0, h - bar_h), (w, h), bg, -1)
    cv2.putText(frame, label, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

# ──────────────────────────────────────────────────────────────────────────────
# Core processor
# ──────────────────────────────────────────────────────────────────────────────
def process_video(video_path: str, audio_threshold: float, visual_conf: float,
                  frame_placeholder, metrics_placeholder, log_placeholder,
                  progress_bar):

    audio_model = load_audio_model()
    yolo        = load_yolo()

    # ── Extract audio track ───────────────────────────────────────────────────
    y_full, sr = librosa.load(video_path, sr=22050, mono=True)
    audio_duration = len(y_full) / sr

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    analyze_every = max(1, int(fps // 4))   # ~4 frames/sec analysis

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = dict(frames=0, alerts=0, visual_hits=0, audio_hits=0,
                 max_audio_prob=0.0, log=[])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))

        if frame_idx % analyze_every != 0:
            continue

        stats["frames"] += 1
        timestamp = frame_idx / fps

        # ── 1. Visual detection via YOLO ──────────────────────────────────────
        results = yolo(frame, verbose=False, conf=visual_conf)[0]
        visual_detected = False
        annotated = frame.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            coords = box.xyxy[0].cpu().numpy()
            is_amb = is_ambulance_by_color(frame, coords)
            lbl    = "Ambulance?" if is_amb else VEHICLE_CLASSES[cls_id]
            color  = (0, 60, 255) if is_amb else (0, 165, 255)
            draw_box(annotated, coords, lbl, color, conf)
            if is_amb:
                visual_detected = True

        if visual_detected:
            stats["visual_hits"] += 1

        # ── 2. Audio detection for this segment ──────────────────────────────
        t0  = max(0, timestamp - 0.5)
        t1  = min(audio_duration, timestamp + 0.5)
        s0, s1 = int(t0 * sr), int(t1 * sr)
        segment = y_full[s0:s1] if s1 > s0 else y_full[:1024]
        feats   = extract_mfcc_segment(segment, sr)
        audio_prob = float(audio_model.predict(feats, verbose=0)[0][0])
        audio_detected = audio_prob >= audio_threshold
        stats["max_audio_prob"] = max(stats["max_audio_prob"], audio_prob)
        if audio_detected:
            stats["audio_hits"] += 1

        # ── 3. Combined alert ─────────────────────────────────────────────────
        alert = visual_detected and audio_detected
        if alert:
            stats["alerts"] += 1

        # ── 4. Annotate & display frame ───────────────────────────────────────
        annotated = overlay_status(annotated, visual_detected, audio_detected, alert)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, use_container_width=True)

        # ── 5. Live metrics ───────────────────────────────────────────────────
        with metrics_placeholder.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"""<div class="metric-card">
                <div class="metric-label">🕐 Timestamp</div>
                <div class="metric-value" style="font-size:1.3rem">{timestamp:.1f}s</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="metric-card">
                <div class="metric-label">🚑 Ambulance</div>
                <div class="metric-value" style="color:{'#ff4444' if visual_detected else '#00c853'}">
                {'YES' if visual_detected else 'NO'}</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class="metric-card">
                <div class="metric-label">🔊 Siren Prob</div>
                <div class="metric-value" style="color:{'#ff4444' if audio_detected else '#00c853'}">
                {audio_prob*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
            c4.markdown(f"""<div class="metric-card">
                <div class="metric-label">🚨 Alerts</div>
                <div class="metric-value" style="color:{'#ff4444' if stats['alerts']>0 else '#aaa'}">
                {stats['alerts']}</div>
            </div>""", unsafe_allow_html=True)

        # ── 6. Log entry ──────────────────────────────────────────────────────
        if alert:
            entry = f'<span class="log-alert">[{timestamp:6.2f}s] 🚨 ALERT — Ambulance + Siren (audio {audio_prob:.2%})</span>'
        elif visual_detected:
            entry = f'<span class="log-warn">[{timestamp:6.2f}s] 🚑 Ambulance visible, no siren</span>'
        elif audio_detected:
            entry = f'<span class="log-warn">[{timestamp:6.2f}s] 🔊 Siren heard ({audio_prob:.2%}), no ambulance</span>'
        else:
            entry = f'<span class="log-safe">[{timestamp:6.2f}s] ✅ Clear — audio {audio_prob:.2%}</span>'
        stats["log"].append(entry)

        log_placeholder.markdown(
            f'<div class="frame-log">' + "<br>".join(stats["log"][-40:]) + '</div>',
            unsafe_allow_html=True,
        )

    cap.release()
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# UI Layout
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# 🚑 Ambulance Detector — Vision + Audio")
st.markdown("Upload a video. The app analyses **every frame** with YOLOv8 + your MFCC siren model and raises a **combined alert only when both ambulance is visible AND siren is heard.**")

st.markdown("---")

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    audio_threshold = st.slider(
        "🔊 Siren detection threshold",
        0.1, 0.9, 0.5, 0.05,
        help="Probability above this = siren detected",
    )
    visual_conf = st.slider(
        "👁️ YOLO confidence threshold",
        0.1, 0.9, 0.35, 0.05,
        help="Min confidence for vehicle detection",
    )

    st.markdown("---")
    st.markdown("### 🧠 Pipeline")
    st.markdown("""
1. **YOLOv8n** detects vehicles each frame  
2. **Color heuristic** flags white/yellow vehicles as ambulance candidates  
3. **MFCC model** classifies ±0.5s audio around the frame  
4. **Alert** = visual **AND** audio both fire  
""")
    st.markdown("### 📦 Models")
    st.markdown("""
- `ambulance_siren_model.h5` — your trained model  
- `yolov8n.pt` — COCO pretrained (auto-downloaded)
""")

uploaded = st.file_uploader(
    "📁 Upload a video file",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="MP4 recommended for best compatibility",
)

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.markdown("---")
    st.markdown("### 🎬 Live Frame Analysis")

    frame_col, _ = st.columns([3, 1])
    with frame_col:
        frame_ph = st.empty()

    metrics_ph = st.empty()
    progress   = st.progress(0.0, text="Analysing…")

    st.markdown("### 📋 Frame-by-Frame Log")
    log_ph = st.empty()

    with st.spinner(""):
        stats = process_video(
            tmp_path, audio_threshold, visual_conf,
            frame_ph, metrics_ph, log_ph, progress
        )

    os.unlink(tmp_path)

    progress.progress(1.0, text="✅ Analysis complete!")
    st.markdown("---")
    st.markdown("### 📊 Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"""<div class="metric-card">
        <div class="metric-label">Frames analysed</div>
        <div class="metric-value">{stats['frames']}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card">
        <div class="metric-label">🚑 Visual hits</div>
        <div class="metric-value" style="color:#ffb300">{stats['visual_hits']}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card">
        <div class="metric-label">🔊 Audio hits</div>
        <div class="metric-value" style="color:#ffb300">{stats['audio_hits']}</div></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card">
        <div class="metric-label">🚨 Combined alerts</div>
        <div class="metric-value" style="color:{'#ff4444' if stats['alerts']>0 else '#aaa'}">{stats['alerts']}</div></div>""", unsafe_allow_html=True)
    c5.markdown(f"""<div class="metric-card">
        <div class="metric-label">Peak siren prob</div>
        <div class="metric-value">{stats['max_audio_prob']*100:.1f}%</div></div>""", unsafe_allow_html=True)

    if stats["alerts"] > 0:
        st.markdown("""<div class="alert-box alert-danger">🚨 AMBULANCE WITH ACTIVE SIREN DETECTED IN THIS VIDEO</div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown("""<div class="alert-box alert-safe">✅ No combined ambulance + siren event detected</div>""",
                    unsafe_allow_html=True)

else:
    st.info("👆 Upload a video file to begin real-time frame-by-frame analysis.")

st.markdown("---")
st.caption("YOLOv8n (COCO) · ambulance_siren_model.h5 · Streamlit · OpenCV · Librosa")
