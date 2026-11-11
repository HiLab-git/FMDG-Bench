# dpt_D_dinov1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import _make_scratch

class DPTHead_DINOv1(nn.Module):
    """
    适配DINOv1的DPTHead
    DINOv1的特征维度: base=768, small=384
    """
    def __init__(
        self, 
        nclass,
        in_channels,  # DINOv1-base: 768, DINOv1-small: 384
        features=256, 
        use_bn=False, 
        out_channels=[192, 384, 768, 1536],  # 适配DINOv1-base
    ):
        super(DPTHead_DINOv1, self).__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None
        self.scratch.output_conv = nn.Conv2d(features*4, nclass, kernel_size=1, stride=1, padding=0)  
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            # DINOv1特征形状: [B, 257, C] -> reshape -> [B, C, H, W]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
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


class DPT_DINOv1(nn.Module):
    """
    适配DINOv1的DPT模型
    关键修改：
    1. 特征提取方式
    2. 通道数适配
    3. 层索引适配
    """
    def __init__(
        self, 
        encoder_size='base', 
        nclass=1,
        features=128, 
        out_channels=None,  # 设为None，根据encoder_size自动设置
        use_bn=False,
        backbone=None,
        freeze_backbone: bool = True
    ):
        super(DPT_DINOv1, self).__init__()
        
        # 层索引 - 适配DINOv1
        self.intermediate_layer_idx = {
            's': [2, 5, 8, 11],  # ViT-small (12层)
            'b': [2, 5, 8, 11],   # ViT-base (12层)
        }
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        
        # 根据encoder_size设置特征维度和输出通道
        if encoder_size == 'b':
            in_channels = 768  # DINOv1-base特征维度
            if out_channels is None:
                out_channels = [192, 384, 768, 1536]  # 适配768维特征
        else:  # 'small'
            in_channels = 384  # DINOv1-small特征维度
            if out_channels is None:
                out_channels = [96, 192, 384, 768]  # 适配384维特征
        
        self.head = DPTHead_DINOv1(
            nclass=nclass,
            in_channels=in_channels,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels
        )
        
        # 冻结backbone
        if freeze_backbone and self.backbone is not None:
            self.lock_backbone()
    
    def lock_backbone(self):
        """冻结backbone所有参数"""
        if self.backbone is None:
            return
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 设置评估模式
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.Dropout, nn.Dropout2d)):
                m.eval()
        print("✅ DINOv1 Backbone has been frozen!")
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        
        # DINOv1特征提取
        with torch.no_grad():
            # 提取中间层特征
            features = []
            x_ = self.backbone.prepare_tokens(x)
            for i, blk in enumerate(self.backbone.blocks):
                x_ = blk(x_)
                if i in self.intermediate_layer_idx[self.encoder_size]:
                    features.append(x_)
        
        # 提取CLS token和patch tokens
        out_features = []
        for feat in features:
            # feat形状: [B, 257, C]
            # 分离CLS token
            cls_token = feat[:, 0, :]  # [B, C]
            patch_tokens = feat[:, 1:, :]  # [B, 256, C]
            out_features.append(patch_tokens)
        
        # 通过头部网络
        out = self.head(out_features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
        return out