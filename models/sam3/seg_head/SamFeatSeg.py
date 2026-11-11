from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Type


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SamFeatSeg(nn.Module):
    def __init__(
        self,
        image_encoder,
        seg_decoder,
        img_size=1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.mask_decoder = seg_decoder

    def forward(self,
                x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x) #[B, 256, 64, 64]
        out = self.mask_decoder(image_embedding)

        if out.shape[-1] != original_size:
            out = F.interpolate(
                out,
                (original_size, original_size),
                mode="bilinear",
                align_corners=False,
            )
        return out

    def get_embedding(self, x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x)
        out = nn.functional.adaptive_avg_pool2d(image_embedding, 1).squeeze()
        return out


class SegDecoderLinear(nn.Module):
    def __init__(self,
                 num_classes=14,
                 emb_dim=256,
                 activation: Type[nn.Module] = nn.GELU,
                 ):
        super().__init__()

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, emb_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(emb_dim // 4),
            activation(),
            nn.ConvTranspose2d(emb_dim // 4, emb_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(emb_dim // 8, emb_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim // 8, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.output_upscaling(x)
        return self.final(out)


class SegDecoderCNN(nn.Module):
    def __init__(self,
                 num_classes=1,
                 embed_dim=256,
                 num_depth=2,
                 top_channel=64,
                 ):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(embed_dim, top_channel*2**num_depth, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_channel * 2 ** num_depth, top_channel * 2 ** num_depth, kernel_size=1),
            nn.ReLU(inplace=True),

        )
        self.blocks = nn.ModuleList()
        for i in range(num_depth):
            if num_depth > 2 > i:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 2,
                                       stride=2)
                )

            self.blocks.append(block)

        self.final = nn.Sequential(
            nn.Conv2d(top_channel, top_channel, 3, padding=1),
            nn.Conv2d(top_channel, num_classes, kernel_size=1)
        )
        # self.learnable_tensor = nn.Parameter(torch.zeros(1, 1, 64, 64))
    def forward(self, x):
        # expanded_tensor = self.learnable_tensor.expand(x.size(0), -1, -1, -1)
        # x = x + expanded_tensor#自己加的可学习的mask
        x = self.input_block(x)
        for blk in self.blocks:
            x = blk(x)
        return self.final(x)

# DPT Head Implementation
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# DPT相关组件
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = in_shape[3]

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


# 修正的DPTHead
class DPTHead(nn.Module):
    def __init__(
        self,
        nclass,
        in_channels_list,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 2048],
    ):
        super(DPTHead, self).__init__()
        self.nclass = nclass
        self.features = features
        self.out_channels = out_channels

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channel, 1)
            for in_channels, out_channel in zip(in_channels_list, out_channels)
        ])

        self.scratch = _make_scratch(out_channels, features)

        total_in_channels = features * 3 + out_channels[3]

        self.scratch.output_conv = nn.Conv2d(
            total_in_channels, nclass, kernel_size=1
        )

    # ✅ 一定要在类里面！！！！
    def forward(self, out_features):
        out = []

        for i, x in enumerate(out_features):
            B, L, C = x.shape
            hw = int(L ** 0.5)
            h = w = hw

            x = x.permute(0, 2, 1).reshape(B, C, h, w)
            x = self.projects[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        target_hw = layer_1_rn.shape[-2:]

        layer_2_up = F.interpolate(layer_2_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_3_up = F.interpolate(layer_3_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_4_up = F.interpolate(layer_4_rn, size=target_hw, mode="bilinear", align_corners=True)

        fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)
        out = self.scratch.output_conv(fused)

        return out


# 简化的DPT解码器（更适合医学图像分割）
class SimpleDPTDecoder(nn.Module):
    """简化的DPT解码器，计算效率更高，更适合医学图像分割"""
    def __init__(
        self,
        num_classes=3,
        embed_dim=256,
        features=128,
        out_channels=[64, 128, 256, 512],  # 使用更小的通道数
        use_bn=False,
    ):
        super(SimpleDPTDecoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.out_channels = out_channels
        
        # 特征金字塔生成
        self.fpn_conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[0], 1),
            nn.BatchNorm2d(out_channels[0]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv2 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[1]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv3 = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[2]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv4 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[3]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion_conv1 = nn.Conv2d(out_channels[0], features, 3, padding=1)
        self.fusion_conv2 = nn.Conv2d(out_channels[1], features, 3, padding=1)
        self.fusion_conv3 = nn.Conv2d(out_channels[2], features, 3, padding=1)
        self.fusion_conv4 = nn.Conv2d(out_channels[3], features, 3, padding=1)
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(features * 4, features, 3, padding=1),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, num_classes, 1)
        )
    
    def forward(self, x):
        """
        输入: [B, 256, 64, 64]
        输出: [B, num_classes, 64, 64]
        """
        # 生成多尺度特征
        f1 = self.fpn_conv1(x)  # [B, 64, 64, 64]
        f2 = self.fpn_conv2(x)  # [B, 128, 32, 32]
        f3 = self.fpn_conv3(f2)  # [B, 256, 16, 16]
        f4 = self.fpn_conv4(f3)  # [B, 512, 8, 8]
        
        # 特征融合
        f1_fused = self.fusion_conv1(f1)
        f2_fused = F.interpolate(
            self.fusion_conv2(f2), 
            size=f1.shape[-2:], 
            mode='bilinear', 
            align_corners=True
        )
        f3_fused = F.interpolate(
            self.fusion_conv3(f3), 
            size=f1.shape[-2:], 
            mode='bilinear', 
            align_corners=True
        )
        f4_fused = F.interpolate(
            self.fusion_conv4(f4), 
            size=f1.shape[-2:], 
            mode='bilinear', 
            align_corners=True
        )
        
        # 融合所有特征
        fused = torch.cat([f1_fused, f2_fused, f3_fused, f4_fused], dim=1)
        
        # 输出
        output = self.output_conv(fused)
        
        return output


# 标准DPT解码器（修复了通道数问题）
class DPTDecoderForSAM(nn.Module):
    """标准DPT解码器，修复了通道数不匹配问题"""
    def __init__(
        self,
        num_classes=3,
        embed_dim=256,
        features=128,
        out_channels=[256, 512, 1024, 2048],
        use_bn=False,
    ):
        super(DPTDecoderForSAM, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.features = features
        
        # 特征金字塔生成
        self.fpn_conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[0], 1),
            nn.BatchNorm2d(out_channels[0]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv2 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[1]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv3 = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[2]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv4 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[3]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        # 计算总的输入通道数
        total_in_channels = features + features + features + out_channels[3]
        
        # 创建DPT Head
        self.dpt_head = DPTHead(
            nclass=num_classes,
            in_channels_list=out_channels,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels
        )
    
    def forward(self, x):
        """
        输入: [B, 256, 64, 64]
        输出: [B, num_classes, 64, 64]
        """
        # 生成多尺度特征
        f1 = self.fpn_conv1(x)  # [B, 256, 64, 64]
        f2 = self.fpn_conv2(x)  # [B, 512, 32, 32]
        f3 = self.fpn_conv3(f2)  # [B, 1024, 16, 16]
        f4 = self.fpn_conv4(f3)  # [B, 2048, 8, 8]
        
        # 将特征转换为序列格式
        seq_features = []
        for feat in [f1, f2, f3, f4]:
            B, C, H, W = feat.shape
            # 将空间维度展平为序列
            seq_feat = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            seq_features.append(seq_feat)
        
        # 通过DPT Head
        output = self.dpt_head(seq_features)
        
        return output


# 轻量级DPT解码器
class LightDPTDecoder(nn.Module):
    """轻量级DPT解码器，计算效率更高"""
    def __init__(
        self,
        num_classes=3,
        embed_dim=256,
        features=64,
        out_channels=[64, 128, 256, 512],
        use_bn=False,
    ):
        super(LightDPTDecoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # 特征金字塔生成
        self.fpn_conv1 = nn.Conv2d(embed_dim, out_channels[0], 1)
        
        self.fpn_conv2 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[1]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv3 = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[2]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv4 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels[3]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(
            out_channels[0] + out_channels[1] + out_channels[2] + out_channels[3], 
            features, 3, padding=1
        )
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, num_classes, 1)
        )
    
    def forward(self, x):
        """
        输入: [B, 256, 64, 64]
        输出: [B, num_classes, 64, 64]
        """
        # 生成多尺度特征
        f1 = self.fpn_conv1(x)  # [B, 64, 64, 64]
        f2 = self.fpn_conv2(x)  # [B, 128, 32, 32]
        f3 = self.fpn_conv3(f2)  # [B, 256, 16, 16]
        f4 = self.fpn_conv4(f3)  # [B, 512, 8, 8]
        
        # 上采样到相同尺寸
        f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=True)
        f3_up = F.interpolate(f3, size=f1.shape[-2:], mode='bilinear', align_corners=True)
        f4_up = F.interpolate(f4, size=f1.shape[-2:], mode='bilinear', align_corners=True)
        
        # 融合所有特征
        fused = torch.cat([f1, f2_up, f3_up, f4_up], dim=1)
        fused = self.fusion_conv(fused)
        
        # 输出
        output = self.output_conv(fused)
        
        return output
