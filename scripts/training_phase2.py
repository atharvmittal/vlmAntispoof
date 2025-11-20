
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json, os
from sklearn.metrics import roc_auc_score
import numpy as np

### ---------------------------
### MODEL DEFINITIONS
### ---------------------------

class FusionModel(nn.Module):
    def __init__(self, img_dim=512, audio_dim=512, use_mac=False, use_mope=False):
        super().__init__()
        self.use_mac = use_mac
        self.use_mope = use_mope

        # frozen CLIP encoder placeholder (to be replaced at runtime)
        self.img_encoder = None

        # audio encoder (ResNet18)
        self.audio_encoder = None

        # fusion head
        self.mlp = nn.Sequential(
            nn.Linear(img_dim + audio_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, img_feat, audio_feat):
        fused = torch.cat([img_feat, audio_feat], dim=1)
        logits = self.mlp(fused)
        return logits


### ---------------------------
### EVALUATION METRICS
### ---------------------------

def compute_eer(fpr, tpr):
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2

def compute_metrics(labels, scores):
    auc = roc_auc_score(labels, scores)
    # threshold for AUC-ROC
    thresh = np.percentile(scores, 50)
    preds = (scores >= thresh).astype(int)

    # APCER / BPCER
    live = labels == 1
    spoof = labels == 0
    apcer = np.mean(preds[spoof] == 1)
    bpcer = np.mean(preds[live] == 0)
    acer = (apcer + bpcer) / 2

    # compute EER
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    eer = compute_eer(fpr, tpr)

    return {
        "auc": float(auc),
        "eer": float(eer),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float(acer)
    }


### ---------------------------
### TRAINING LOOP
### ---------------------------

def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total = 0
    for batch in loader:
        img = batch["img"].to(device)
        aud = batch["audio"].to(device)
        lab = batch["label"].to(device)

        optim.zero_grad()
        logits = model(img, aud)
        loss = loss_fn(logits, lab)
        loss.backward()
        optim.step()
        total += loss.item()
    return total / len(loader)


def validate(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device)
            aud = batch["audio"].to(device)
            lab = batch["label"].cpu().numpy()

            logits = model(img, aud)
            prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()

            scores.extend(prob)
            labels.extend(lab)

    return compute_metrics(np.array(labels), np.array(scores))


### ---------------------------
### EXPERIMENT RUNNER
### ---------------------------

def run_experiment(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FusionModel(
        img_dim=config["img_dim"],
        audio_dim=config["audio_dim"],
        use_mac=config["use_mac"],
        use_mope=config["use_mope"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    train_loader = config["train_loader"]
    val_loader = config["val_loader"]

    history = []
    for epoch in range(1, config["epochs"]+1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val = validate(model, val_loader, device)
        print(f"Epoch {epoch}/{config['epochs']} | Train loss: {tr:.4f} | Val: {val}")
        history.append(val)

    # save results
    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(history, f, indent=4)

    return history
