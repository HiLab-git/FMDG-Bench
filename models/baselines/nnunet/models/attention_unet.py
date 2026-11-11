# -*- coding: utf-8 -*-
"""
Attention U-Net 实现
基于论文 "Attention U-Net: Learning Where to Look for the Pancreas" (Oktay et al., 2018)

架构: 标准 UNet 编码器-解码器, 在每个跳跃连接处插入 Attention Gate (AG).
AG 用解码器上采样特征作为门控信号, 对编码器跳跃连接特征进行空间注意力加权。

无额外依赖, 纯 PyTorch 实现。
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention Gate: 用门控信号 g (解码器) 对跳跃连接 x (编码器) 做加权。

    输入:
        x: 编码器特征 [B, C_x, H, W]
        g: 解码器门控信号 [B, C_g, H, W]  (已上采样与 x 同尺寸)
    输出:
        x * alpha: 注意力加权后的特征 [B, C_x, H, W]

    参数量很小 (~C_x * C_g / 2 的 1x1 卷积)
    """

    def __init__(self, F_g, F_l, F_int):
        """
        参数:
            F_g:   门控信号通道数 (decoder feature channels)
            F_l:   跳跃连接通道数 (encoder feature channels)
            F_int: 中间通道数 (通常 F_int = F_l // 2)
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: [B, F_l, H, W],  g: [B, F_g, H, W]
        g1 = self.W_g(g)                            # [B, F_int, H, W]
        x1 = self.W_x(x)                            # [B, F_int, H, W]
        psi = self.relu(g1 + x1)                    # [B, F_int, H, W]
        alpha = self.psi(psi)                       # [B, 1, H, W]
        return x * alpha                            # 加权跳跃连接


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """2x2 MaxPool + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """上采样 + DoubleConv + Attention Gate (门控跳跃连接)"""
    def __init__(self, in_ch, out_ch, skip_ch, bilinear=True):
        """
        in_ch:   上采样输入通道数 (来自上一解码层)
        out_ch:  输出通道数
        skip_ch: 跳跃连接通道数 (来自对应编码层)
        """
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)

        # Attention Gate: 门控信号通道 = in_ch (上采样后), 跳跃连接通道 = skip_ch
        self.attention = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)

        # 注意: 经过 AG 后 concat(skip_ch, in_ch) = skip_ch + in_ch
        self.conv = DoubleConv(skip_ch + in_ch, out_ch)

    def forward(self, x1, x2):
        # x1: decoder feature (to be upsampled)
        # x2: encoder skip feature (to be gated)
        x1 = self.up(x1)

        # 尺寸对齐 (处理奇偶尺寸差异)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])

        # Attention Gate: 用 x1 作为门控信号, 对 x2 加权
        x2 = self.attention(x2, x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionUNetModel(nn.Module):
    """
    Attention U-Net (Oktay et al., 2018)

    编码器: DoubleConv × 4 → 通道 64, 128, 256, 512
    瓶颈:   DoubleConv → 1024
    解码器: Up (上采样 + AG + DoubleConv) × 4 → 通道 512, 256, 128, 64
    输出:   1×1 Conv → out_channels

    单一输出, 无 deep supervision。
    """

    def __init__(self, in_channels=3, out_channels=2, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 编码器
        self.inc = DoubleConv(in_channels, 64)          # 64
        self.down1 = Down(64, 128)                      # 128
        self.down2 = Down(128, 256)                     # 256
        self.down3 = Down(256, 512)                     # 512
        self.down4 = Down(512, 1024)                    # 1024 (瓶颈, 无用 MaxPool)

        # 解码器 (上采样 + AG)
        self.up1 = Up(1024, 512, 512, bilinear)         # 1024 -> 512, skip=512
        self.up2 = Up(512, 256, 256, bilinear)          # 512  -> 256, skip=256
        self.up3 = Up(256, 128, 128, bilinear)          # 256  -> 128, skip=128
        self.up4 = Up(128, 64, 64, bilinear)            # 128  -> 64,  skip=64

        # 输出
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, image):
        # image: [B, in_channels, H, W]
        x1 = self.inc(image)                            # [B, 64, H, W]
        x2 = self.down1(x1)                             # [B, 128, H/2, W/2]
        x3 = self.down2(x2)                             # [B, 256, H/4, W/4]
        x4 = self.down3(x3)                             # [B, 512, H/8, W/8]
        x5 = self.down4(x4)                             # [B, 1024, H/16, W/16]

        x = self.up1(x5, x4)                            # [B, 512, H/8, W/8]
        x = self.up2(x, x3)                             # [B, 256, H/4, W/4]
        x = self.up3(x, x2)                             # [B, 128, H/2, W/2]
        x = self.up4(x, x1)                             # [B, 64, H, W]

        logits = self.outc(x)                           # [B, out_channels, H, W]
        return logits
