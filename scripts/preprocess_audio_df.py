
import os
import json
import librosa
import numpy as np

def wav_to_mel(wav_path, out_path, sr=16000, n_mels=128):
    y, sr = librosa.load(wav_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)
    np.save(out_path, mel_db)
    return mel_db

def preprocess_asvspoof(wav_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for root, dirs, files in os.walk(wav_dir):
        for f in files:
            if f.endswith(".flac") or f.endswith(".wav"):
                wav_path = os.path.join(root, f)
                speaker = f.split("_")[0]
                label = "live" if "bonafide" in f else "spoof"
                mel_out = os.path.join(out_dir, f.replace(".flac",".npy").replace(".wav",".npy"))
                wav_to_mel(wav_path, mel_out)
                meta.append({
                    "wav": wav_path,
                    "mel_path": mel_out,
                    "speaker": speaker,
                    "label": label
                })
    return meta
