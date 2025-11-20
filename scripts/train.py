import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FusionDataset
from model import FusionModel


def train_and_eval(exp_name, fusion_json, use_mac=False, use_mope=False, epochs=3,
                   batch_size=4, lr=1e-4, device="cuda",
                   max_steps_per_epoch=2000):    # ★ LIMIT STEPS PER EPOCH

    print(f"\n=== Starting HARD Dataset Training ===")
    print(f">>> Running: {exp_name} (MAC={use_mac}, MoPE={use_mope})\n")

    # Load fusion JSON
    with open(fusion_json, "r") as f:
        fusion_list = json.load(f)

    dataset = FusionDataset(fusion_list)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Create fusion model
    model = FusionModel(use_mac=use_mac, use_mope=use_mope).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    save_dir = f"/content/drive/MyDrive/VLM_AntiSpoof_HARD/ablation_experiments/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Training Loop
    # -------------------------
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        step = 0

        for batch in loader:
            imgs, mels, labels = batch
            imgs, mels, labels = imgs.to(device), mels.to(device), labels.to(device)

            optim.zero_grad()
            logits, mac_loss = model(imgs, mels)

            loss = ce_loss(logits, labels)
            if use_mac:
                loss += 0.1 * mac_loss

            loss.backward()
            optim.step()

            running_loss += loss.item()

            if step % 50 == 0:
                print(f"[{exp_name}] Ep{ep} Step{step} loss:{loss.item():.4f}")

            step += 1

            # ★ STOP EARLY FOR FAST TRAINING
            if step >= max_steps_per_epoch:
                print(f"\nReached max {max_steps_per_epoch} steps. Ending epoch early.\n")
                break

        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"epoch_{ep}.pt")
        torch.save(model.state_dict(), ckpt_path)

    print("\nTraining finished.\n")
    return True
