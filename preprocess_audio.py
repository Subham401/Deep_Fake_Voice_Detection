import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def extract_mfcc(file_path, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        # Load the audio file
        audio, _ = librosa.load(file_path, sr=sr)
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return mfcc
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_mfcc(file_path):
    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time", cmap="magma")
    plt.colorbar(label="MFCC Coefficients")
    plt.title(f"MFCC of {os.path.basename(file_path)}")
    plt.xlabel("Time (frames)")
    plt.ylabel("MFCC Coefficients")
    plt.show()

def save_mfcc_to_csv(file_path, output_csv="mfcc_features.csv"):
    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return

    # Convert MFCCs to a DataFrame
    df = pd.DataFrame(mfcc.T)  # Transpose to make time steps rows
    df.insert(0, "Filename", os.path.basename(file_path))  # Add filename column
    
    # Save to CSV
    df.to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=False)
    print(f"MFCC features saved to {output_csv}")

if __name__ == "__main__":
    file_path = input("Enter the audio file path: ")  # Allow user input
    if os.path.exists(file_path):
        plot_mfcc(file_path)
        save_mfcc_to_csv(file_path)
    else:
        print("File not found. Please check the path.")
