import streamlit as st
import numpy as np
import cv2
import librosa
import tempfile
import os
from pathlib import Path
from collections import deque
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Ambulance Detection System", layout="wide")

st.title("🚑 Production Ambulance Detection System (Vision + Audio)")

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_audio_model():
    return keras.models.load_model("ambulance_siren_model.h5")

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")  # replace with ambulance.pt in real production

audio_model = load_audio_model()
yolo = load_yolo()

# COCO vehicle classes
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_mfcc(y, sr):
    if len(y) < 512:
        y = np.pad(y, (0, 512 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

def detect_ambulance_yolo(results):
    """
    Production assumption:
    Replace with custom-trained YOLO ambulance model.
    """
    detected = False
    best_conf = 0.0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in VEHICLE_CLASSES:
            if conf > best_conf:
                best_conf = conf
            detected = True

    return detected, best_conf


# ─────────────────────────────────────────────
# TEMPORAL STATE MACHINE (VERY IMPORTANT)
# ─────────────────────────────────────────────
class AlertState:
    def __init__(self):
        self.visual_history = deque(maxlen=8)
        self.audio_history = deque(maxlen=8)
        self.alert_active = False

    def update(self, visual, audio):
        self.visual_history.append(visual)
        self.audio_history.append(audio)

        visual_score = sum(self.visual_history) / len(self.visual_history)
        audio_score = sum(self.audio_history) / len(self.audio_history)

        # thresholds tuned for stability
        visual_ok = visual_score >= 0.5
        audio_ok = audio_score >= 0.5

        return visual_ok, audio_ok, visual_score, audio_score


# ─────────────────────────────────────────────
# VIDEO PROCESSING
# ─────────────────────────────────────────────
def process_video(video_path, audio_threshold, visual_conf):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    y, sr = librosa.load(video_path, sr=22050, mono=True)

    alert_state = AlertState()

    frame_slot = st.empty()
    log_slot = st.empty()
    progress = st.progress(0)

    frame_idx = 0
    logs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        progress.progress(frame_idx / max(total_frames, 1))

        # ─────────────────────────────
        # YOLO inference (optimized)
        # ─────────────────────────────
        results = yolo(frame, imgsz=640, verbose=False)[0]

        visual_detected, conf = detect_ambulance_yolo(results)

        # ─────────────────────────────
        # AUDIO WINDOW
        # ─────────────────────────────
        t = frame_idx / fps
        start = max(0, int((t - 0.5) * sr))
        end = min(len(y), int((t + 0.5) * sr))

        segment = y[start:end]
        feats = extract_mfcc(segment, sr)

        audio_prob = float(audio_model.predict(feats, verbose=0)[0][0])
        audio_detected = audio_prob >= audio_threshold

        # ─────────────────────────────
        # TEMPORAL SMOOTHING
        # ─────────────────────────────
        visual_smooth, audio_smooth, v_score, a_score = alert_state.update(
            int(visual_detected),
            int(audio_detected)
        )

        alert = visual_smooth and audio_smooth

        # ─────────────────────────────
        # DRAW OVERLAY
        # ─────────────────────────────
        label = "CLEAR"
        color = (0, 255, 0)

        if alert:
            label = "🚨 AMBULANCE + SIREN"
            color = (0, 0, 255)
        elif visual_smooth:
            label = "🚑 AMBULANCE ONLY"
            color = (0, 165, 255)
        elif audio_smooth:
            label = "🔊 SIREN ONLY"
            color = (255, 165, 0)

        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame_slot.image(frame, channels="BGR")

        # ─────────────────────────────
        # LOGGING
        # ─────────────────────────────
        log = f"[{t:5.1f}s] V:{v_score:.2f} A:{a_score:.2f} ALERT:{alert}"
        logs.append(log)

        log_slot.text("\n".join(logs[-15:]))

    cap.release()

    return logs


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

audio_threshold = st.sidebar.slider("Audio threshold", 0.1, 0.9, 0.5)
visual_conf = st.sidebar.slider("Visual confidence", 0.1, 0.9, 0.4)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    st.info("Processing video...")

    logs = process_video(path, audio_threshold, visual_conf)

    os.remove(path)

    st.success("Processing complete")

    if any("True" in l and "ALERT" in l for l in logs):
        st.error("🚨 Emergency vehicle detected with siren")
    else:
        st.success("No emergency event detected")
