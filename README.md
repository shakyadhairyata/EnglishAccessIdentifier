# English Accent Identifier

This project identifies the **English accent** (e.g., British, American, Australian) from a public video URL using a deep learning model.

---

## Features

- Accepts a video URL (e.g., Loom, direct .mp4)
- Extracts audio and transcribes it using Whisper or Wav2Vec2
- Classifies accent from the audio
- Outputs:
  - Accent label (e.g., American)
  - Confidence score (e.g., 87%)
  - Optional summary explanation

---

##  Quick Start

### 1. Clone this Repo

```bash
git clone https://github.com/yourusername/english-accent-identifier.git
cd english-accent-identifier
