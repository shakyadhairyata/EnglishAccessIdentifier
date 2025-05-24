import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from utils.downloader import download_video
from utils.classifier import classify_accent
from utils.trainer import (
    download_common_voice,
    preprocess_and_extract,
    train_and_test_model,
    save_model
)
import joblib


st.header("Accent Model Trainer")
global dataset
if st.button("Download Dataset"):
    st.info("Downloading Common Voice dataset...")

    dataset = download_common_voice()
    st.success("Dataset downloaded")
    st.success(dataset)
    df = preprocess_and_extract(dataset)
    st.success("Feature extraction complete and saved to CSV.")
    st.success(df)
    
if st.button("Train & Test Model"):
    with st.spinner("Training and Testing model..."):
        clf, le, report = train_and_test_model()
    st.success("Model trained successfully!")
    st.text_area("Classification Report", report, height=250)
    st.success("Model trained and tested.")
    try:
        save_model(clf, le)
        st.success("Model and label encoder saved.")
    except:
        st.error("Model not found. Please train the model first.")

    
MODEL_PATH = "models/accent_classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
classifier = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

st.title("English Accent Identifier")
st.markdown("Paste a **video URL (e.g. YouTube)** to detect speaker's English accent.")

url = st.text_input("Enter video URL:")

if st.button("Analyze") and url:
    with st.spinner("Downloading and processing audio..."):
        audio_path = download_video(url)
        accent, confidence = classify_accent(classifier, audio_path, label_encoder)

    st.success(f"**Predicted Accent:** {accent}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    if confidence < 70:
        st.warning("Low confidence. Try a clearer clip or better quality audio.")
