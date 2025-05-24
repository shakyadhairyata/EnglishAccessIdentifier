import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import streamlit as st
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login


tqdm.pandas()

# Global Paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_common_voice(lang="en"):
    login("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")
    dataset = load_dataset("mozilla-foundation/common_voice_13_0", lang, split="train[:1%]", trust_remote_code=True)
    return dataset

def extract_mfcc(path, sr=22050, n_mfcc=13):
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        if audio is None or len(audio) == 0:
            print("Audio is empty.")
            return None

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)  # Shape: (n_mfcc,)
        
        if np.isnan(mfcc_mean).any():
            print("MFCC mean has NaNs.")
            return None

        return mfcc_mean
    except Exception as e:
        print(f"MFCC extraction failed for {path}: {e}")
        return None


def preprocess_and_extract(dataset, save_path="preprocessed_data.csv"):
    data = []
    for example in tqdm(dataset, desc="Extracting MFCC"):
        if example["accent"] and example["audio"]:
            file_path = example["audio"]["path"]
            accent = example["accent"]
            features = extract_mfcc(file_path)
            if features is not None:
                data.append((features.tolist(), accent))
    df = pd.DataFrame(data, columns=["features", "accent"])
    df.to_csv(save_path, index=False)
    return df

def train_and_test_model(csv_path="preprocessed_data.csv"):
    df = pd.read_csv(csv_path)
    X = df["features"].apply(eval).tolist()
    y = df["accent"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42,verbose=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    labels_in_test = np.unique(y_test)
    target_names_in_test = le.inverse_transform(labels_in_test)

    report = classification_report(
        y_test, 
        y_pred, 
        labels=labels_in_test, 
        target_names=target_names_in_test
    )
    return clf, le, report

def save_model(clf, le):
    joblib.dump(clf, os.path.join(MODEL_DIR, "accent_classifier.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
