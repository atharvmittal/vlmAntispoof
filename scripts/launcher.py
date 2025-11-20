import os
from train import train_and_eval

# Registry of ablations
REGISTRY = {
    "fusion_only": {"use_mac": False, "use_mope": False},
    "fusion_mac":  {"use_mac": True,  "use_mope": False},
    "fusion_mope": {"use_mac": False, "use_mope": True},
    "fusion_full": {"use_mac": True,  "use_mope": True},
}

# Fusion JSON file paths
FUSION_DIR = "/content/drive/MyDrive/VLM_AntiSpoof_HARD/data/fusion"
FUSION_TRAIN = os.path.join(FUSION_DIR, "fusion_train_fixed.json")
FUSION_VAL   = os.path.join(FUSION_DIR, "fusion_val_fixed.json")
FUSION_TEST  = os.path.join(FUSION_DIR, "fusion_test_fixed.json")


def run(name, json_path, epochs=3):
    cfg = REGISTRY[name]
    print(f"\n=== Starting HARD Dataset Training ===")
    print(f">>> Running: {name} (MAC={cfg['use_mac']}, MoPE={cfg['use_mope']})\n")

    return train_and_eval(
        exp_name=name,
        fusion_json=json_path,        # ★ pass the PATH, not a list
        use_mac=cfg["use_mac"],
        use_mope=cfg["use_mope"],
        epochs=epochs
    )


if __name__ == "__main__":
    # Run fusion_only first
    run("fusion_only", FUSION_TRAIN, epochs=3)
