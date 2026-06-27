import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("ambulance_siren_model.h5")

st.title("🚑 Ambulance Siren Detection")

audio_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3"]
)

def extract_features(file):
    audio, sr = librosa.load(file, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc_scaled = np.mean(mfcc.T, axis=0)

    return mfcc_scaled

if audio_file is not None:

    st.audio(audio_file)

    features = extract_features(audio_file)

    features = features.reshape(1, 40)

    prediction = model.predict(features)

    confidence = prediction[0][0]

    if confidence > 0.5:
        st.success(
            f"🚑 Siren Detected ({confidence:.2f})"
        )
    else:
        st.error(
            f"❌ No Siren ({1-confidence:.2f})"
        )
