
import torch
import torch.nn as nn
import torchvision.models as models
import clip

### ---------------------------------------
### CLIP Image Encoder (Frozen)
### ---------------------------------------

class CLIPImageEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, img):
        # CLIP expects normalized float32
        img = img.float()
        return self.model.encode_image(img)

### ---------------------------------------
### Audio Encoder (ResNet18)
### ---------------------------------------

class AudioEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Identity()  # output dim = 512

    def forward(self, mel):
        return self.model(mel)

### ---------------------------------------
### MAC Loss (Modality Alignment Contrastive Loss)
### ---------------------------------------

class MACLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, img_feat, aud_feat, label):
        # label: 1 = live, 0 = spoof
        pos = (label == 1).float().unsqueeze(1)
        neg = (label == 0).float().unsqueeze(1)

        dist = torch.norm(img_feat - aud_feat, dim=1, keepdim=True)

        pos_loss = pos * dist**2
        neg_loss = neg * torch.clamp(self.margin - dist, min=0.0)**2

        return (pos_loss + neg_loss).mean()

### ---------------------------------------
### MoPE (Simple Prompt Enhancement)
### ---------------------------------------

class MoPE(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(1, dim))

    def forward(self, x):
        return x + self.prompt

