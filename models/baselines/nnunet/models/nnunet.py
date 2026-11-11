# -*- coding: utf-8 -*-
"""
nnU-Net 2D 网络架构

从 nnU-Net v2 框架中提取的 2D U-Net 架构, 用于集成到统一训练流程中。
特点: Conv-InstanceNorm-LeakyReLU 块, Deep Supervision, 通道上限 320。

网络自动根据输入尺寸计算下采样层数 (最多 6 层)。
输入 256x256 时: 6 层下采样 (直到 4x4), deep supervision 在最低 2 层输出。

参考: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based
      biomedical image segmentation", Nature Methods, 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _maybe_crop(tensor, target_size):
    """如果 tensor 尺寸大于 target_size, 居中裁剪"""
    if tensor.shape[2:] == target_size:
        return tensor
    h, w = tensor.shape[2], tensor.shape[3]
    th, tw = target_size
    dh = (h - th) // 2
    dw = (w - tw) // 2
    return tensor[:, :, dh:dh + th, dw:dw + tw]


class ConvBlock(nn.Module):
    """
    nnU-Net 基础卷积块:
    Conv3x3 -> InstanceNorm2d -> LeakyReLU
    连续两次 (与 nnU-Net 原版一致)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.lrelu1(self.norm1(self.conv1(x)))
        x = self.lrelu2(self.norm2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """Strided Conv (2x2 stride=2) + ConvBlock (替代 MaxPool, nnU-Net 原版设计)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2,
                              stride=2, bias=False)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        return self.conv(self.down(x))


class UpsampleBlock(nn.Module):
    """
    nnU-Net 上采样块:
    - 上采样 (bilinear 或转置卷积)
    - 与跳跃连接 (skip) 拼接
    - ConvBlock
    可选: deep supervision 输出 (1x1 conv -> 各类别 logits)
    """

    def __init__(self, in_channels, skip_channels, out_channels,
                 use_deep_supervision=False, num_classes=2):
        """
        in_channels:   上采样输入通道
        skip_channels: 跳跃连接通道
        out_channels:  输出通道
        """
        super().__init__()
        # 上采样: 直接用插值 (nnU-Net v2 默认 bilinear)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # 拼接后: in_channels + skip_channels
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

        # Deep supervision 输出头
        self.use_ds = use_deep_supervision
        if self.use_ds:
            self.ds_out = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x, skip, return_ds=False):
        x = self.upsample(x)
        # 对齐尺寸
        x = _maybe_crop(x, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        if self.use_ds and return_ds:
            ds = self.ds_out(x)
            return x, ds
        return x


class nnUNetModel(nn.Module):
    """
    nnU-Net 2D 架构 (提取自 nnU-Net v2)

    配置:
        - 初始通道: 32, 每层翻倍, 上限 320
        - InstanceNorm2d (适合 batch_size=1)
        - LeakyReLU (negative_slope=0.01)
        - 2 层 Deep Supervision (最低两个分辨率)

    对于 256x256 输入:
        编码器: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        瓶颈:  8 -> 4
        解码器: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        最低 2 层提供 deep supervision 输出
    """

    def __init__(self, in_channels=3, out_channels=2,
                 init_channels=32, max_channels=320,
                 n_conv_per_stage=2,
                 deep_supervision=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision

        # ---------- 计算通道列表 ----------
        # 最多 6 层下采样 (256 -> 4), 实际层数由输入尺寸决定
        max_stages = 6
        encoder_channels = []
        ch = init_channels
        for _ in range(max_stages + 1):  # +1 for bottleneck
            encoder_channels.append(min(ch, max_channels))
            ch *= 2
        # encoder_channels: [32, 64, 128, 256, 320, 320, 320]

        # ---------- 编码器 ----------
        self.encoder_stages = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        current_ch = in_channels
        for stage_ch in encoder_channels[:-1]:  # 最后一层进入瓶颈
            stage = nn.Sequential(*[
                ConvBlock(current_ch if i == 0 else stage_ch, stage_ch)
                for i in range(n_conv_per_stage)
            ])
            self.encoder_stages.append(stage)
            self.down_blocks.append(DownBlock(stage_ch, stage_ch))
            current_ch = stage_ch

        # 瓶颈
        bottleneck_ch = encoder_channels[-1]
        self.bottleneck = nn.Sequential(*[
            ConvBlock(current_ch if i == 0 else bottleneck_ch, bottleneck_ch)
            for i in range(n_conv_per_stage)
        ])

        # ---------- 解码器 ----------
        # 解码器层数 = 编码器层数 (去除瓶颈)
        num_decoder_stages = len(encoder_channels) - 1

        # 预计算每层解码器的通道数
        decoder_channels = []
        for i in range(num_decoder_stages):
            out_ch = encoder_channels[num_decoder_stages - 1 - i]
            decoder_channels.append(out_ch)

        # 构建 decoder_stages
        self.decoder_stages = nn.ModuleList()
        for i in range(num_decoder_stages):
            in_ch = bottleneck_ch if i == 0 else decoder_channels[i - 1]
            skip_ch = encoder_channels[num_decoder_stages - 1 - i]
            out_ch = encoder_channels[num_decoder_stages - 1 - i]
            use_ds = self.deep_supervision and (i < 2)
            self.decoder_stages.append(
                UpsampleBlock(in_ch, skip_ch, out_ch,
                              use_deep_supervision=use_ds,
                              num_classes=out_channels)
            )

        # ---------- 最终输出层 ----------
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, image):
        """
        训练模式 (self.training=True, deep_supervision=True):
            返回 list: [main_output, ds_lowest, ds_second_lowest]
        推理模式 (self.training=False):
            返回 Tensor: [B, out_channels, H, W]
        """
        # 编码
        skips = []
        x = image
        for stage, down in zip(self.encoder_stages, self.down_blocks):
            x = stage(x)
            skips.append(x)
            x = down(x)

        # 瓶颈
        x = self.bottleneck(x)

        # 解码 (反向遍历)
        ds_outputs = []
        for i, decoder in enumerate(self.decoder_stages):
            skip = skips[-(i + 1)]
            result = decoder(x, skip, return_ds=self.training)
            if self.training and isinstance(result, tuple):
                x, ds = result
                ds_outputs.append(ds)
            else:
                x = result

        # 主输出
        main_out = self.final_conv(x)

        if self.training and len(ds_outputs) > 0:
            # nnU-Net 顺序: [主输出最前, 然后是低分辨率辅助输出 (从小到大)]
            return [main_out] + ds_outputs

        return main_out
