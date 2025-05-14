import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal

# --- Model parameters
RESNET_TYPE = 'resnet18'
PRETRAINED = True
DROPOUT_P = 0.2

class RGB_ResNetInverseDynamics(nn.Module):
    def __init__(self,
                 num_frames: int, *,
                 output_dim=8, pretrained=True,
                 resnet_type='resnet18',
                 dropout_p=0.2):
        super().__init__()
        self.pretrained = pretrained

        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 512
        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 512
        elif resnet_type == 'resnet50':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 2048
        elif resnet_type == 'resnet101':
            self.resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 2048
        else:
            raise ValueError(f'Invalid resnet type: {resnet_type}')

        # Remove classification head
        self.resnet.fc = nn.Identity()

        self.feature_extractor = self.resnet
        # Regressor head with dropout
        # Takes features from all frames (concatenated)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * num_frames, 256),  # Doubled due to concatenation
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, output_dim)
        )
  
    def forward(self, frames_btchw: torch.Tensor):
        B, T, C, H, W = frames_btchw.shape          # expect C=3
        feats = self.feature_extractor(frames_btchw.view(B*T, C, H, W))
        feats = feats.view(B, -1)                   # (B, T*feature_dim)
        actions = self.regressor(feats)
        return actions


    def save(self, path): torch.save(self.state_dict(), path)

    def load(self, path): self.load_state_dict(torch.load(path))