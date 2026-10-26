"""
Re-implemented Baseline Architectures for Multi-Label Chest X-Ray Classification
Implements the 6 baselines described in Section 2, Section 4.1, Table 2, Table 4, and Table 5:
1. Mezina & Burget (2024) [Inception-ViT Hybrid]
2. CNN-O-ELMNet (Agarwal et al. 2024) [CNN + Optimized ELM]
3. DTRSN-XRI-CPI (Prasath et al. 2025) [Double Transformer Residual Network]
4. Zhao & Wang (2025) [Large Kernel Attention + GCN]
5. CAMSGNeT (Prince et al. 2025) [Texture Descriptor + CAM Attention]
6. DLSAC-Net (Bhardwaj & Kaur 2025) [Dual Attention Spatial & Channel Net]

Under the Fair Comparison Protocol (Section 4.1), all baselines use identical
multi-label sigmoid output heads trained with binary cross-entropy.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# 1. Mezina & Burget: Hybrid CNN-Transformer
# -------------------------------------------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        c = out_c // 4
        self.b1 = nn.Conv2d(in_c, c, kernel_size=1)
        self.b2 = nn.Sequential(nn.Conv2d(in_c, c, kernel_size=1), nn.Conv2d(c, c, kernel_size=3, padding=1))
        self.b3 = nn.Sequential(nn.Conv2d(in_c, c, kernel_size=1), nn.Conv2d(c, c, kernel_size=5, padding=2))
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), nn.Conv2d(in_c, c, kernel_size=1))
        self.norm = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.act(self.norm(out))


class MezinaBurgetNet(nn.Module):
    """Mezina & Burget (2024): Hybrid CNN-Transformer"""
    def __init__(self, num_classes: int = 14, in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inc1 = InceptionBlock(64, 128)
        self.inc2 = InceptionBlock(128, 256)
        self.token_pool = nn.AdaptiveAvgPool2d((14, 14))  # 196 tokens
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.head = nn.Linear(256, num_classes)

    def forward(self, x, *args, **kwargs):
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.token_pool(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        logits = self.head(pooled)
        return torch.sigmoid(logits)


# -------------------------------------------------------------
# 2. CNN-O-ELMNet: CNN + Optimized Extreme Learning Machine
# -------------------------------------------------------------
class CNNOELMNet(nn.Module):
    """Agarwal et al. (2024): CNN-O-ELMNet"""
    def __init__(self, num_classes: int = 14, in_channels: int = 1, hidden_elm: int = 1024):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        # Random frozen projection simulating Extreme Learning Machine input weights
        self.elm_weight = nn.Parameter(torch.randn(128 * 4 * 4, hidden_elm) / math.sqrt(hidden_elm), requires_grad=False)
        self.elm_bias = nn.Parameter(torch.zeros(hidden_elm), requires_grad=False)
        self.out_linear = nn.Linear(hidden_elm, num_classes)

    def forward(self, x, *args, **kwargs):
        f = self.features(x).flatten(1)
        h = torch.relu(f @ self.elm_weight + self.elm_bias)
        logits = self.out_linear(h)
        return torch.sigmoid(logits)


# -------------------------------------------------------------
# 3. DTRSN-XRI-CPI: Double Transformer Residual Super-Resolution
# -------------------------------------------------------------
class DTRSNXRICPI(nn.Module):
    """Prasath et al. (2025): Double Transformer Residual Network"""
    def __init__(self, num_classes: int = 14, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, embed_dim, kernel_size=16, stride=16)
        # Stage 1: Spatial Transformer
        l1 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer1 = nn.TransformerEncoder(l1, num_layers=3)
        # Residual bridge
        self.res_conv = nn.Linear(embed_dim, embed_dim)
        # Stage 2: Cross-Scale Transformer
        l2 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer2 = nn.TransformerEncoder(l2, num_layers=3)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, *args, **kwargs):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        t1 = self.transformer1(tokens)
        t_bridge = t1 + self.res_conv(tokens)
        t2 = self.transformer2(t_bridge)
        pooled = t2.mean(dim=1)
        logits = self.head(pooled)
        return torch.sigmoid(logits)


# -------------------------------------------------------------
# 4. Zhao & Wang: Large-Kernel CNN + Graph Convolutional Network
# -------------------------------------------------------------
class LargeKernelBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Depthwise Large Kernel (e.g. 7x7 or 13x13 decomposed)
        self.dw = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.pw(self.dw(x)))


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, K, C], adj: [K, K]
        h = self.linear(x)
        return torch.einsum("vw,bwk->bvk", adj, h)


class ZhaoWangNet(nn.Module):
    """Zhao & Wang (2025): Large-Kernel Attention + GCN"""
    def __init__(self, num_classes: int = 14, in_channels: int = 1, channels: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, channels, kernel_size=4, stride=4),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.lka = LargeKernelBlock(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # GCN for multi-label dependency modeling
        self.label_embeddings = nn.Parameter(torch.randn(num_classes, 64))
        self.gcn = GCNLayer(64, channels)
        
        # Symmetric adjacency matrix initialized with uniform prior
        adj = torch.ones(num_classes, num_classes) / num_classes
        self.register_buffer("adj", adj)
        self.classifier = nn.Linear(channels * 2, num_classes)

    def forward(self, x, *args, **kwargs):
        feat = self.stem(x)
        feat = self.lka(feat)
        img_rep = self.pool(feat).flatten(1)  # [B, channels]
        
        # GCN label reasoning
        B = x.size(0)
        lbl_rep = self.label_embeddings.unsqueeze(0).repeat(B, 1, 1)
        gcn_out = self.gcn(lbl_rep, self.adj).mean(dim=1)  # [B, channels]
        
        combined = torch.cat([img_rep, gcn_out], dim=-1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits)


# -------------------------------------------------------------
# 5. CAMSGNeT: Texture Descriptors + CAM Attention
# -------------------------------------------------------------
class SpatialGatingUnit(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim // 2)
        self.proj = nn.Linear(dim // 2, dim // 2)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v)
        return u * (v + 1)


class CAMSGNeT(nn.Module):
    """Prince et al. (2025): CAMSGNeT with Texture & CAM Gating"""
    def __init__(self, num_classes: int = 14, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.sgu = SpatialGatingUnit(256)
        self.head = nn.Linear(128 * 7 * 7, num_classes)

    def forward(self, x, *args, **kwargs):
        f = self.features(x)  # [B, 256, 7, 7]
        B, C, H, W = f.shape
        f_flat = f.permute(0, 2, 3, 1).reshape(B, H * W, C)
        gated = self.sgu(f_flat)  # [B, H*W, 128]
        logits = self.head(gated.reshape(B, -1))
        return torch.sigmoid(logits)


# -------------------------------------------------------------
# 6. DLSAC-Net: Dual Attention (Spatial and Channel) Network
# (Strongest Baseline across all experiments)
# -------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        scale = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class DLSACNet(nn.Module):
    """
    Bhardwaj & Kaur (2025): DLSAC-Net (Dual Attention Spatial & Channel Network).
    Highest-performing baseline in Table 2, Table 3, Table 4, and Table 5.
    """
    def __init__(self, num_classes: int = 14, in_channels: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.channel_att = ChannelAttention(512)
        self.spatial_att = SpatialAttention()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, *args, **kwargs):
        feat = self.encoder(x)
        feat = self.channel_att(feat)
        feat = self.spatial_att(feat)
        pooled = self.global_pool(feat).flatten(1)
        logits = self.classifier(pooled)
        return torch.sigmoid(logits)
