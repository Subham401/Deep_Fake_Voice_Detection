import torch
import librosa
import numpy as np

# Load Trained Model
class AudioClassifier(torch.nn.Module):
    def __init__(self, input_size=13):
        super(AudioClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Function to Predict Real/Fake
def predict_audio(file_path, model_path="audio_model.pth"):
    model = AudioClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    audio, _ = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    mfcc = np.mean(mfcc, axis=1)
    tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, 1).item()

    return "Fake" if prediction == 1 else "Real"

# Run Detection
if __name__ == "__main__":
    file_path = "D:\Subham\Desktop\Project\WhatsApp Audio 2025-05-17 at 22.46.00_83bda967.waptt.wav"  # Replace with your test file
    result = predict_audio(file_path)
    print(f"Prediction: {result}")
