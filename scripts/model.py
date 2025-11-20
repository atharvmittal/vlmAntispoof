import torch
import torch.nn as nn
import torch.nn.functional as F

# Try import CLIP, fallback to torchvision resnet18
def get_image_encoder(device='cpu', out_dim=512, frozen=True):
    try:
        import clip
        model, _ = clip.load("ViT-B/32", device=device, jit=False)

        class ImgEnc(nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip = clip_model
            def forward(self, x):
                return self.clip.encode_image(x)

        enc = ImgEnc(model)
        if frozen:
            for p in enc.parameters():
                p.requires_grad = False
        return enc

    except Exception:
        from torchvision import models
        res = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        res.fc = nn.Linear(res.fc.in_features, out_dim)
        if frozen:
            for p in res.parameters():
                p.requires_grad = False
        return res


def get_audio_encoder(out_dim=512, pretrained=False):
    from torchvision import models
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

    conv1 = model.conv1
    model.conv1 = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size,
                            stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)

    model.fc = nn.Linear(model.fc.in_features, out_dim)
    return model


class FusionHead(nn.Module):
    def __init__(self, in_dim=1024, hidden=512, out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out)
        )
    def forward(self, x):
        return self.net(x)


class MACLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, img_emb, aud_emb, labels):
        img_emb = F.normalize(img_emb, dim=-1)
        aud_emb = F.normalize(aud_emb, dim=-1)
        sim = (img_emb * aud_emb).sum(dim=-1)

        pos_mask = labels == 1
        neg_mask = ~pos_mask

        pos_loss = (1 - sim[pos_mask]).mean() if pos_mask.sum() > 0 else 0.0
        neg_loss = (sim[neg_mask] + 1).clamp(min=0).mean() if neg_mask.sum() > 0 else 0.0

        return pos_loss + neg_loss


class MoPE(nn.Module):
    def __init__(self, emb_dim=512, n_prompts=4):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(n_prompts, emb_dim) * 0.02)

    def forward(self, x):
        return x + self.prompts.mean(dim=0)


# ------------------------------------------------------------
# ★★★★★ FINAL FUSION MODEL (used by train.py)
# ------------------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self, use_mac=False, use_mope=False, device="cuda"):
        super().__init__()
        self.use_mac = use_mac
        self.use_mope = use_mope

        self.img_encoder = get_image_encoder(device=device, out_dim=512, frozen=True)
        self.aud_encoder = get_audio_encoder(out_dim=512)

        self.mope_img = MoPE(512) if use_mope else None
        self.mope_aud = MoPE(512) if use_mope else None

        self.fusion = FusionHead(1024, 512, 2)
        self.mac_loss = MACLoss() if use_mac else None

    def forward(self, img, mel, labels=None):
        img_emb = self.img_encoder(img)
        aud_emb = self.aud_encoder(mel)

        if self.use_mope:
            img_emb = self.mope_img(img_emb)
            aud_emb = self.mope_aud(aud_emb)

        fused = torch.cat([img_emb, aud_emb], dim=-1)
        logits = self.fusion(fused)

        if self.use_mac and labels is not None:
            mac = self.mac_loss(img_emb, aud_emb, labels)
        else:
            mac = None

        return logits, mac
