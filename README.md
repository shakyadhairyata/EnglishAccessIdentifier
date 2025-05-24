# English Accent Detector

A browser-based tool and API for detecting English accents (American, British, Australian, Indian, Canadian, etc.) from video URLs using a hybrid deep learning pipeline.

---

## 🚀 Features

* **Input**: Public video URL (YouTube, Loom, direct MP4 link)
* **Audio extraction** using `yt-dlp` and `ffmpeg-python`
* **Feature extraction**: MFCCs via `librosa`
* **Model**: RandomForest trained on Common Voice MFCC features
* **Output**:

  * Accent classification
  * Confidence score (0–100%)
  * Confidence breakdown for all classes
* **Training pipeline**: Download, preprocess, train, test
* **UI**: Streamlit app to drive all steps from the browser

---

## 🏗️ Project Structure

```
english-accent-detector/
├── app/
│   └── main.py                  # Streamlit UI and orchestrator
├── models/
│   ├── accent_classifier.pkl    # Trained RandomForest model
│   └── label_encoder.pkl        # Fitted LabelEncoder
├── utils/
│   ├── downloader.py            # video/audio download utilities
│   ├── audio_utils.py           # MFCC extraction & prediction logic
│   └── trainer.py               # download, preprocess, train/test, save
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md                    # Project overview and instructions
```

---

## 🛠️ Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/english-accent-detector.git
   cd english-accent-detector
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv accentenv
   # Windows
   accentenv\Scripts\activate
   # macOS / Linux
   source accentenv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install system FFmpeg**

   * **Windows**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add `ffmpeg/bin` to PATH
   * **macOS**: `brew install ffmpeg`
   * **Linux**: `sudo apt install ffmpeg`

---

## 🧩 Usage

1. **Run the Streamlit app**

   ```bash
   streamlit run app/main.py
   ```

2. **Follow the browser UI** to:

   * Download Common Voice subset (5–30%)
   * Preprocess (extract MFCCs)
   * Train & test the RandomForest model
   * Save the model
   * Input a YouTube URL to analyze accent

3. **View results**:

   * Predicted accent
   * Confidence score
   * Breakdown of probabilities for all accent classes

---

## 📦 Training & Testing Pipeline

The training pipeline is in `utils/trainer.py` and includes functions to:

* **`download_common_voice(lang, split_percent)`**: Authenticates with Hugging Face and downloads a slice (e.g. `[:5%]`) of the Common Voice dataset.
* **`preprocess_and_extract(dataset, save_path)`**: Extracts MFCC features and saves them to `preprocessed_data.csv`.
* **`train_and_test_model(csv_path)`**: Loads the CSV, splits into train/test, trains a `RandomForestClassifier`, evaluates on test set, and returns the model, label encoder, and classification report.
* **`save_model(clf, le)`**: Saves `clf` and `le` into `models/`.

---

## 🔮 Inference Logic

* **Download and convert** video URL to audio WAV via `yt-dlp` and `ffmpeg-python` (`utils/downloader.py`).
* **Extract MFCC** features with `librosa`
* **Load model & encoder** from `models/`
* **Predict** accent and confidence in `utils/classifier.py`

---

## ⚙️ Configuration

* Adjust dataset slice in `utils/trainer.py`:

  ```python
  ```

dataset = load\_dataset(..., split="train\[:1%]", ...)

```
- Modify MFCC parameters in `utils/audio_utils.py`

---

## 🌍 Deployment

- **Local**: `streamlit run app/main.py`

---

## 🤝 Contributing

Contributions welcome! Please open issues or pull requests for feature requests, bug fixes, or improvements.

---

## 📄 License

MIT License © Dhairyata Shakya

```
