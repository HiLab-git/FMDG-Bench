# dpt_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import _make_scratch

class DPTHead(nn.Module):
    """
    完整的DPT解码器头
    与DINOv2版本保持一致
    """
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()
        
        # 投影层，将CLIP特征投影到指定通道数
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        # 创建scratch网络
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None
        
        # 输出卷积层
        self.scratch.output_conv = nn.Conv2d(
            features * 4, nclass, kernel_size=1, stride=1, padding=0
        )
        
        # 可选：添加批量归一化
        if use_bn:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, out_features, patch_h, patch_w):
        """
        前向传播
        Args:
            out_features: 4个中间层特征的列表，每个形状为 [B, L, C]
            patch_h: patch网格的高度
            patch_w: patch网格的宽度
        Returns:
            分割结果，形状为 [B, nclass, H, W]
        """
        out = []
        
        # 处理每个特征层
        for i, x in enumerate(out_features):
            B, L, C = x.shape
            
            # 重塑为空间特征图 [B, C, patch_h, patch_w]
            x = x.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
            
            # 1x1卷积投影
            x = self.projects[i](x)
            out.append(x)
        
        # 获取4个特征层
        layer_1, layer_2, layer_3, layer_4 = out
        
        # 通过scratch网络的残差块
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # 上采样到第一层的尺寸
        target_hw = layer_1_rn.shape[-2:]
        layer_2_up = F.interpolate(layer_2_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_3_up = F.interpolate(layer_3_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_4_up = F.interpolate(layer_4_rn, size=target_hw, mode="bilinear", align_corners=True)
        
        # 特征融合
        fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)
        
        # 最终输出
        out = self.scratch.output_conv(fused)
        return out