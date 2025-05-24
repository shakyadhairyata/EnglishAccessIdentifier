import numpy as np
from utils.trainer import extract_mfcc
def classify_accent(classifier, audio_path, label_encoder):
    mfcc = extract_mfcc(audio_path)
    if mfcc is None:
        print("MFCC is None")
        return "Unknown", 0.0

    mfcc = np.array(mfcc)

    if np.isnan(mfcc).any():
        print("MFCC contains NaNs")
        return "Unknown", 0.0

    mfcc = mfcc.reshape(1, -1)

    try:
        pred = classifier.predict(mfcc)[0]
        probas = classifier.predict_proba(mfcc)[0]
        confidence = float(np.max(probas) * 100)
        return label_encoder.inverse_transform([pred])[0], confidence
    except Exception as e:
        print(f"Classification failed: {e}")
        return "Unknown", 0.0