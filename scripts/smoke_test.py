# Minimal smoke test that uses the new clean scripts
import os, json, torch
from dataset import FusionDataset
from model import get_image_encoder, get_audio_encoder, FusionHead

fusion_train = "/content/drive/MyDrive/VLM_AntiSpoof_HARD/data/fusion/fusion_train_fixed.json"
with open(fusion_train, "r") as f:
    data = json.load(f)

print("Fusion list length:", len(data))

# create tiny dataset and dataloader
ds = FusionDataset(fusion_train, max_mel_len=400)
dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

imgs, mels, labels = next(iter(dl))
print("Image batch:", imgs.shape)
print("Mel batch:", mels.shape)
print("Labels:", labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_enc = get_image_encoder(device=device).to(device)
aud_enc = get_audio_encoder().to(device)
fusion = FusionHead(in_dim=512+512).to(device)

imgs = imgs.to(device)
mels = mels.to(device)

with torch.no_grad():
    try:
        img_emb = img_enc(imgs)
    except Exception as e:
        # sometimes CLIP expects normalized input; if fallback triggers, try different path
        img_emb = img_enc(imgs)

aud_emb = aud_enc(mels)

print("Image emb shape:", getattr(img_emb, 'shape', None))
print("Audio emb shape:", getattr(aud_emb, 'shape', None))

fused = torch.cat([img_emb.view(img_emb.size(0), -1), aud_emb.view(aud_emb.size(0), -1)], dim=-1)
logits = fusion(fused)
print("Fusion logits shape:", logits.shape)
print("Smoke test passed (if shapes look correct).")
