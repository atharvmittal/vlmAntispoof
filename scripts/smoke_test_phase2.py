import sys; sys.path.append("/content/drive/MyDrive/VLM_AntiSpoof_HARD/scripts")
import sys, os
sys.path.append("/content/drive/MyDrive/VLM_AntiSpoof_HARD/scripts")


from fusion_dataset import FusionDataset
from launcher_phase2 import train_and_eval_phase2
from torchvision import transforms
from torch.utils.data import DataLoader
import os

def smoke_test(fusion_dir):
    train_json = os.path.join(fusion_dir, "fusion_train_fixed.json")
    val_json   = os.path.join(fusion_dir, "fusion_val_fixed.json")

    if not os.path.exists(train_json):
        print("Fusion JSONs not found. Preprocess CASIA & DF first.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = FusionDataset(train_json, transform)
    val_ds   = FusionDataset(val_json, transform)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

    print("Running 1-epoch smoke test…")
    history = train_and_eval_phase2(
        exp_name="fusion_only_smoke",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1
    )
    print("Smoke test complete:", history)

