# VLM-AntiSpoof (Face + Voice Multimodal Anti-Spoofing)

A lightweight multimodal anti-spoofing system that combines **CLIP-based visual encoding** with **audio spectrogram encoding** for detecting spoof attacks across face and voice modalities.

---

## 🚀 Key Features
- **Vision–Language Model (CLIP) Encoder** used for face embeddings  
- **Audio Encoder** based on ResNet18 operating on mel-spectrograms  
- **Fusion Head** that jointly learns cross-modal correlations  
- Support for **MAC (Modality Alignment Constraint)**  
- Support for **MoPE (Modality Prompt Enhancement)**  
- Clean, modular code under `scripts/`

---

## 📁 Repository Structure
scripts/
dataset.py # Loads face images + mel spectrogram pairs
model.py # CLIP image encoder + audio encoder + fusion head
train.py # Training loop with optional MAC/MoPE losses
launcher.py # Runs selected experiment
smoke_test.py # Verifies model+dataset pipeline quickly

data/
face/ # CASIA-FASD frames
audio/ # ASVspoof DF mel-spectrograms
fusion/ # Pre-built pairings of (face, audio, label)



---

## 🔧 Quick Start
Run this to verify everything works:

```bash
python scripts/smoke_test.py
Then train:

python scripts/launcher.py

📦 Datasets Used
CASIA-FASD (face anti-spoofing)

ASVspoof 2021 DF (voice anti-spoofing)

Custom fusion pairs linking face frames to audio samples

📝 License
MIT License — free to use, modify, and distribute with attribution.


