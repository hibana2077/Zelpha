import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import timm

class ScalePooledBackbone(nn.Module):
    def __init__(self, feature_dim=128, scales=[0.8, 1.0, 1.2], use_spectral_norm=True):
        super().__init__()
        self.scales = scales
        self.feature_dim = feature_dim
        self.use_spectral_norm = use_spectral_norm

        # Simple CNN Backbone
        # 4 Conv blocks
        self.conv1 = self._make_conv(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = self._make_conv(64, 128)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = self._make_conv(128, 256)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = self._make_conv(256, feature_dim)
        self.bn4 = nn.BatchNorm2d(feature_dim)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def _make_conv(self, in_c, out_c):
        conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        if self.use_spectral_norm:
            return spectral_norm(conv)
        return conv

    def _forward_one_scale(self, x):
        # x: [B, 3, H, W]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x) # 128
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x) # 64
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x) # 32
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.gap(x) # [B, D, 1, 1]
        return x.flatten(1) # [B, D]

    def forward(self, x):
        # x: [B, 3, H, W]
        multi_scale_feats = []
        
        for s in self.scales:
            if s == 1.0:
                x_s = x
            else:
                # Resize input tensor
                # Note: F.interpolate expects [B, C, H, W]
                x_s = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
            
            z_s = self._forward_one_scale(x_s)
            multi_scale_feats.append(z_s)
        
        # Stack: [B, S, D]
        Z = torch.stack(multi_scale_feats, dim=1)
        
        # Scale Pooling (Mean or Max)
        z_pooled = Z.mean(dim=1) 
        
        return z_pooled

class TimmBackbone(nn.Module):
    def __init__(self, model_name, feature_dim=128, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = feature_dim
        
        # Get the output channels of the timm model
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = self.model(dummy_input)
            in_features = out.shape[1]
            
        self.proj = nn.Linear(in_features, feature_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.proj(x)
        return x

class PrototypeHead(nn.Module):
    def __init__(self, num_classes, feature_dim, num_prototypes=1):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        # Initialize prototypes: [C, K, D]
        self.mu = nn.Parameter(torch.randn(num_classes, num_prototypes, feature_dim))

    def forward(self, z):
        # z: [B, D]
        # mu: [C, K, D]
        
        # Expand z: [B, 1, 1, D]
        z_expanded = z.unsqueeze(1).unsqueeze(1) # [B, 1, 1, D]
        
        # Expand mu: [1, C, K, D]
        mu_expanded = self.mu.unsqueeze(0) # [1, C, K, D]
        
        # Dist sq: |z - mu|^2 = sum((z - mu)^2, dim=-1)
        # [B, C, K]
        dist_sq = (z_expanded - mu_expanded).pow(2).sum(dim=-1)
        
        # Min distance per class: [B, C]
        min_dist_sq, _ = dist_sq.min(dim=2)
        
        # Logits = -min_dist_sq
        logits = -min_dist_sq
        
        return logits, min_dist_sq

class LinearHead(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, z):
        logits = self.fc(z)
        return logits, None

class ZelphaModel(nn.Module):
    def __init__(self, num_classes=21, feature_dim=128, 
                 model_name='zelpha', scales=[0.8, 1.0, 1.2],
                 use_spectral_norm=True, use_prototype=True, use_scale_pooling=True,
                 num_prototypes=1):
        super().__init__()
        
        # Backbone Selection
        if model_name == 'zelpha':
            eff_scales = scales if use_scale_pooling else [1.0]
            self.backbone = ScalePooledBackbone(feature_dim, eff_scales, use_spectral_norm)
        else:
            # For timm models, we assume standard single scale processing
            # If user wants scale pooling with timm, it's more complex, assuming no for now.
            self.backbone = TimmBackbone(model_name, feature_dim)

        # Head Selection
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        
        if use_prototype:
            self.classifier = PrototypeHead(num_classes, feature_dim, num_prototypes)
        else:
            self.classifier = LinearHead(num_classes, feature_dim)

    def set_classifier(self, type='linear', num_prototypes=None):
        if num_prototypes is None:
            num_prototypes = self.num_prototypes
            
        if type == 'linear':
            self.classifier = LinearHead(self.num_classes, self.feature_dim)
        elif type == 'prototype':
            self.classifier = PrototypeHead(self.num_classes, self.feature_dim, num_prototypes)
            self.num_prototypes = num_prototypes

    def forward(self, x):
        z = self.backbone(x)
        logits, dist_sq = self.classifier(z)
        return logits, z, dist_sq
