# -*- coding: utf-8 -*-
"""
TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
参考: Chen et al., 2021 (https://arxiv.org/abs/2102.04306)

架构:
    CNN Encoder (ResNet50, ImageNet 预训练) → ViT-B/16 Transformer → CUP Decoder (级联上采样)

参数量: ~105M (其中 ResNet50 ~25M, ViT ~86M)
输入: [B, 3, 256, 256] → 输出: [B, out_channels, 256, 256]

显存估算 (1080Ti 11GB):
    batch_size=4, AMP → ~9-10GB/GPU
    batch_size=16, 4×GPU DataParallel → 每卡 4 样本, ~9-10GB/GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# 基础模块
# ============================================================
class DoubleConv(nn.Module):
    """Conv3x3 → BN → ReLU × 2"""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """2× Upsample + Skip Concat + DoubleConv"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ============================================================
# TransUNet 主体
# ============================================================
class TransUNetModel(nn.Module):
    """
    TransUNet: CNN-Transformer 混合架构

    Encoder:
        ResNet50 (ImageNet 预训练) → H/16 特征图
        → 1×1 Conv 投影到 768 维
        → ViT-B/16: 12 层 Transformer (d=768, head=12, ff=3072)

    Decoder (CUP):
        H/16 → H/8 → H/4 → H/2 → H
        跳跃连接: ResNet layer2 (H/8), ResNet layer1 (H/4)

    参数:
        in_channels:  输入通道 (默认 3)
        out_channels: 分割类别数 (默认 2)
        img_size:     输入尺寸 (默认 256)
    """

    def __init__(self, in_channels=3, out_channels=2, img_size=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size

        # ========================
        # 1. ResNet50 Encoder (ImageNet 预训练)
        # ========================
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Stage0: conv1+bn+relu+maxpool → H/4, 64ch
        self.encoder_stage0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        if in_channels != 3:
            # 适配非 3 通道输入 (保留预训练权重的均值)
            old_conv = self.encoder_stage0[0]
            new_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # 对 1 通道输入: 取 RGB 三通道均值
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            self.encoder_stage0[0] = new_conv

        # Stage1: layer1 → H/4, 256ch
        self.encoder_stage1 = resnet.layer1
        # Stage2: layer2 → H/8, 512ch
        self.encoder_stage2 = resnet.layer2
        # Stage3: layer3 → H/16, 1024ch
        self.encoder_stage3 = resnet.layer3

        # ========================
        # 2. Patch Embedding: 1024 → 768
        # ========================
        self.patch_embed = nn.Conv2d(1024, 768, kernel_size=1)

        # ========================
        # 3. Position Embedding
        # ========================
        grid_size = img_size // 16  # 256 → 16×16 = 256 patches
        num_patches = grid_size * grid_size
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, 768))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

        # ========================
        # 4. ViT-B/16 Transformer Encoder
        # ========================
        # 12 layers, 768 dim, 12 heads, MLP 3072 (标准 ViT-B 配置)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (ViT 标准做法)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # ========================
        # 5. CNN Decoder (CUP: Cascaded Upsampler)
        # ========================
        # H/16→H/8: skip from encoder_stage2 (512ch)
        self.decoder3 = DecoderBlock(768, 512, 512)
        # H/8→H/4: skip from encoder_stage1 (256ch)
        self.decoder2 = DecoderBlock(512, 256, 256)
        # H/4→H/2: no skip
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(256, 128),
        )
        # H/2→H: no skip
        self.decoder0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(128, 64),
        )

        # ========================
        # 6. Segmentation Head
        # ========================
        self.seg_head = nn.Conv2d(64, out_channels, kernel_size=1)

        # ========================
        # 7. 初始化非预训练部分
        # ========================
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """对解码器和 Transformer 进行 Kaiming 初始化"""
        pretrained_modules = set()
        for stage in [self.encoder_stage0, self.encoder_stage1,
                       self.encoder_stage2, self.encoder_stage3]:
            for m in stage.modules():
                pretrained_modules.add(m)

        for m in self.modules():
            if m in pretrained_modules:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image):
        """
        参数:
            image: [B, in_channels, H, W]
        返回:
            [B, out_channels, H, W]  (无 deep supervision, 单一输出)
        """
        B = image.shape[0]

        # ---- CNN Encoder ----
        s0 = self.encoder_stage0(image)      # [B, 64,   H/4, W/4]
        s1 = self.encoder_stage1(s0)         # [B, 256,  H/4, W/4]
        s2 = self.encoder_stage2(s1)         # [B, 512,  H/8, W/8]
        s3 = self.encoder_stage3(s2)         # [B, 1024, H/16, W/16]

        # ---- Patch Embedding ----
        x = self.patch_embed(s3)              # [B, 768, H/16, W/16]
        _, _, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)      # [B, N, 768]  N = Hp × Wp
        x = x + self.position_embeddings

        # ---- Transformer ----
        x = self.transformer(x)               # [B, N, 768]

        # ---- Reshape → 2D ----
        x = x.transpose(1, 2).view(B, 768, Hp, Wp)  # [B, 768, H/16, W/16]

        # ---- CNN Decoder ----
        x = self.decoder3(x, s2)              # [B, 512, H/8, W/8]
        x = self.decoder2(x, s1)              # [B, 256, H/4, W/4]
        x = self.decoder1(x)                  # [B, 128, H/2, W/2]
        x = self.decoder0(x)                  # [B, 64,  H,   W]

        # ---- Segmentation Head ----
        out = self.seg_head(x)                # [B, out_channels, H, W]

        return out
