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

## 📁 Project Structure

project/
│
├── app.py # Flask server & API endpoints
├── detect_audio.py # ML model logic & prediction
├── preprocess_audio.py # MFCC feature extraction utils
├── audio_model.pth # Trained PyTorch model
│
├── templates/
│ └── front.html # User-facing web interface (served by Flask)
│
├── static/
│ ├── logo.jpg # Header logo image
│ └── background.mp4 # Background video
│
├── uploads/ # Temporary folder for audio uploads
└── README.md
