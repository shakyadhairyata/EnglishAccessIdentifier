import numpy as np
from utils.trainer import extract_mfcc
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]
def keyword_classifier(transcript):
    british_keywords = ['colour', 'favourite', 'mum', 'bloody', 'cheers', 'bloke', 'queue',
'lorry', 'boot', 'biscuit', 'flat', 'holiday', 'petrol', 'lift','autumn', 'nappies', 'garden', 'crisps', 'fortnight', 'postcode']
    american_keywords = ['truck', 'vacation', 'sidewalk', 'color', 'favorite', 'mom','anywho','dude','gas','awesome','gonna','wanna']
    australian_keywords = ['gday', 'arvo', 'brekkie','arvo']
    indian_keywords = ['outstation', 'crores', 'prepone']
    canadian_keywords=['colour', 'neighbour', 'centre']

    text_lower = text.lower()
    scores = {'British': 0, 'American': 0, 'Australian': 0,, 'Indian': 0, 'Canadian': 0}

    for word in british_keywords:
        if word in text_lower:
            scores['British'] += 1
    for word in american_keywords:
        if word in text_lower:
            scores['American'] += 1
    for word in australian_keywords:
        if word in text_lower:
            scores['Australian'] += 1
    for word in indian_keywords:
        if word in text_lower:
            scores['Indian'] += 1
    for word in canadian_keywords:
        if word in text_lower:
            scores['Canadian'] += 1

    # Pick accent with max score
    accent = max(scores, key=scores.get)
    confidence = scores[accent] / max(1, sum(scores.values())) * 100

    if sum(scores.values()) == 0:
        accent = "Unknown"
        confidence = 0

    return accent, confidence
def classify_accent(classifier, audio_path, label_encoder):
    mfcc = extract_mfcc(audio_path)
    if mfcc is None:
        print("MFCC is None")

    mfcc = np.array(mfcc)

    if np.isnan(mfcc).any():
        print("MFCC contains NaNs")

    mfcc = mfcc.reshape(1, -1)

    try:
        pred = classifier.predict(mfcc)[0]
        probas = classifier.predict_proba(mfcc)[0]
        confidence = float(np.max(probas) * 100)
        if confidence == 0:
            # fallback
            transcript = transcribe_audio(audio_path)
            kb_accent, kb_confidence = keyword_classifier(transcript)
            return kb_accent, kb_confidence
        return label_encoder.inverse_transform([pred])[0], confidence
    except Exception as e:
        print(f"Classification failed: {e}")
        return "Unknown", 0.0