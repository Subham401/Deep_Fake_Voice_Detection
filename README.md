# Deep_Fake_Voice_Detection
To develop a system that detects whether an audio file is real or AI-generated using machine learning and signal processing.


# 🛡️ AI Guardian – Deepfake Voice Detection

**AI Guardian** is a web-based tool designed to detect whether an audio file contains a real human voice or an AI-generated deepfake. Built with a PyTorch-based model and powered by a Flask API, it allows users to upload audio files and instantly get a prediction — *Real* or *Fake*.


---

## 🚀 Features

- 🎤 Upload `.wav` or any audio file via drag & drop or file picker.
- 🔍 Detects AI-generated (deepfake) vs. Real human voice.
- 🧠 Machine Learning model trained on MFCC audio features.
- 📦 Flask backend API with PyTorch inference.
- 🌐 Interactive frontend UI with live prediction results.
- 🎨 Stylish design with video background and responsive layout.

---

## MIT Licensed

---

## 💡 How It Works

1. **Frontend (`front.html`)**:
   - Lets user upload or drag-and-drop an audio file.
   - Uses JavaScript `fetch()` to call the Flask API with the audio.

2. **Backend (`app.py`)**:
   - Receives the audio file at `/detect`.
   - Saves and processes it via `predict_audio()` from `detect_audio.py`.
   - Returns a JSON result: `{ "result": "Real" }` or `{ "result": "Fake" }`.

3. **Model (`audio_model.pth`)**:
   - A PyTorch classifier trained on MFCC features to distinguish real from fake voice samples.

---

## 🧪 Local Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/yourusername/ai-guardian.git
cd ai-guardian
pip install -r Requirements.txt

---

