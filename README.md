# VLM AntiSpoof
# VLM AntiSpoof (Face + Voice Multimodal Anti-Spoofing)

A multimodal anti-spoofing system combining CLIP (image encoder) and ResNet18 (audio encoder) with optional MAC Loss and MoPE.

This project targets **real-world** face & voice spoof detection using:
- CASIA-FASD (hard face spoof dataset)
- ASVspoof 2021 DF (deepfake audio)
- Multimodal fusion with shared embeddings

---

## 🚀 Features
- CLIP ViT-B/32 image encoder (frozen)
- ResNet18 audio encoder (trainable)
- Fusion MLP (1024 → 512 → 2)
- MAC Loss (modality alignment)
- MoPE (prompt enhancement)
- Ablation-ready training pipeline
- Clean subject-disjoint fusion dataset
- Ready for cross-dataset generalization

---

## 📂 Repository Structure

