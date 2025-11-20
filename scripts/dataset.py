import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

class FusionDataset(Dataset):
    def __init__(self, fusion_json, max_mel_len=400):
        # fusion_json = path to fusion_xxx.json
        with open(fusion_json, "r") as f:
            self.items = json.load(f)

        self.max_mel_len = max_mel_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]

        # -------- Load Image --------
        img_path = it["img"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img).astype("float32") / 255.0
        img = torch.tensor(img).permute(2, 0, 1)  # HWC → CHW

        # -------- Load Mel --------
        mel = np.load(it["mel"])  # shape: (128, T)
        mel = torch.tensor(mel).unsqueeze(0)      # → (1, 128, T)
        T = mel.shape[-1]

        # pad or trim
        if T < self.max_mel_len:
            pad = self.max_mel_len - T
            mel = torch.nn.functional.pad(mel, (0, pad))
        else:
            mel = mel[:, :, :self.max_mel_len]

        # -------- Label --------
        label = 1 if it["label"] == "live" else 0
        label = torch.tensor(label, dtype=torch.long)

        return img, mel, label
