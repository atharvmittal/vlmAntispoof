
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class FusionDataset(Dataset):
    def __init__(self, fusion_json, transform_img=None):
        import json
        self.items = json.load(open(fusion_json))
        self.transform_img = transform_img

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        frames_dir = item["face_frames"]
        mel_path   = item["audio_mel"]
        label      = 1 if item["label"] == "live" else 0

        # pick a random frame from face folder
        frame_list = sorted(os.listdir(frames_dir))
        frame_path = os.path.join(frames_dir, random.choice(frame_list))
        img = Image.open(frame_path).convert("RGB")

        if self.transform_img:
            img = self.transform_img(img)
        else:
            img = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.

        # load mel spectrogram
        mel = np.load(mel_path)
        mel = torch.tensor(mel).unsqueeze(0).float()  # shape: (1, N, T)

        return {
            "img": img,
            "audio": mel,
            "label": torch.tensor(label).long()
        }
