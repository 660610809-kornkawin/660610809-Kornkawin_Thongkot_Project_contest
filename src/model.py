import torch
import torch.nn as nn
from torchvision import models


class FoodPreferenceNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights="IMAGENET1K_V1")
        backbone.fc = nn.Identity()  # ตัด classifier เดิมออก ได้ feature 512 มิติ

        self.encoder = backbone

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def encode(self, x):
        return self.encoder(x)  # → [B, 512]

    def forward(self, img1, img2):
        f1 = self.encode(img1)  # [B, 512]
        f2 = self.encode(img2)  # [B, 512]
        feat = torch.cat([f1, f2], dim=1)  # [B, 1024]
        return self.fc(feat)  # [B, 1]  (logit ก่อน sigmoid)
