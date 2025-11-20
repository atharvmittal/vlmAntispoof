import sys; sys.path.append("/content/drive/MyDrive/VLM_AntiSpoof_HARD/scripts")

import os, json
from training_phase2 import run_experiment

def train_and_eval_phase2(exp_name, train_loader, val_loader, epochs=10, lr=1e-4,
                          img_dim=512, audio_dim=512, out_base="/content/drive/MyDrive/VLM_AntiSpoof_HARD/ablation_experiments"):
    out_dir = os.path.join(out_base, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    config = {
        "epochs": epochs,
        "lr": lr,
        "img_dim": img_dim,
        "audio_dim": audio_dim,
        "use_mac": ("mac" in exp_name),
        "use_mope": ("mope" in exp_name),
        "train_loader": train_loader,
        "val_loader": val_loader,
        "out_dir": out_dir
    }
    history = run_experiment(config)
    # save summary
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(history, f, indent=4)
    return history
