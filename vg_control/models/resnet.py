import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal

# --- Model parameters
RESNET_TYPE = 'resnet18'
PRETRAINED = True
DROPOUT_P = 0.2
NUM_DOTS: Literal[1, 3] = 1

class RGB_ResNetInverseDynamics(nn.Module):
    def __init__(self,
                 kernel_size=7,
                 padding=3,
                 output_dim=2 * NUM_DOTS,
                 pretrained=True,
                 resnet_type='resnet18',
                 dropout_p=0.2):
        super().__init__()
        self.pretrained = pretrained

        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif resnet_type == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif resnet_type == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f'Invalid resnet type: {resnet_type}')

        # Adapt the first conv layer for RGB input
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        # Remove classification head
        self.resnet.fc = nn.Identity()
        
        # Feature extractor (shared for both frames)
        self.feature_extractor = nn.Sequential(
            self.resnet,
            nn.Flatten()
        )

        # Regressor head with dropout
        # Takes features from both frames (concatenated)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),  # Doubled due to concatenation
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, frame1, frame2):
        """
        Forward pass for inverse dynamics:
        - Extract features from both frames
        - Concatenate features
        - Predict the action that caused the transition
        
        If predict_per_dot is True and num_dots is provided, predicts
        separate actions for each dot (useful for multi-dot datasets)
        
        Args:
            frame1: First frame [batch, channels, height, width]
            frame2: Second frame [batch, channels, height, width]
            num_dots: Number of dots to predict actions for (optional)
            
        Returns:
            actions: Predicted actions with shape:
                    - [batch, num_dots, 2]
        """
        # Extract features from both frames
        B, *_ = frame1.shape
        features1 = self.feature_extractor(frame1)
        features2 = self.feature_extractor(frame2)
        
        # Concatenate features
        combined_features = torch.cat((features1, features2), dim=1)
        
        # Predict actions
        actions = self.regressor(combined_features)
        return actions

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))