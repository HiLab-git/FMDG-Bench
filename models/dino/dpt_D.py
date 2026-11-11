import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import _make_scratch

class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024],
    ):
        super(DPTHead, self).__init__()
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

class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=2,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
        backbone = None,
        freeze_backbone: bool = True  # 新增：是否冻结backbone（默认冻结）
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            "large": [4, 11, 17, 23],
        }
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
        
        # ========== 核心修改1：初始化时自动冻结backbone ==========
        if freeze_backbone and self.backbone is not None:
            self.lock_backbone()

    # ========== 核心修改2：增强冻结逻辑（递归冻结所有子模块） ==========
    def lock_backbone(self):
        """递归冻结backbone的所有参数，确保无遗漏（适配嵌套结构的backbone如CLIP/DINO）"""
        if self.backbone is None:
            return
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 额外确保backbone的所有子模块处于eval模式（避免BatchNorm/Dropout更新）
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.Dropout, nn.Dropout2d)):
                m.eval()
        print("✅ Backbone (encoder) has been frozen completely!")

    # ========== 可选：解冻backbone（如需后续微调） ==========
    def unlock_backbone(self, lr: float = 1e-5):
        """解冻backbone，可指定微调学习率（通常远小于解码器）"""
        if self.backbone is None:
            return
        for p in self.backbone.parameters():
            p.requires_grad = True
        # 恢复训练模式
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.Dropout, nn.Dropout2d)):
                m.train()
        print(f"✅ Backbone (encoder) has been unlocked (finetune lr: {lr})!")
        return lr

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        # ========== 关键：冻结后强制backbone为eval模式（避免BatchNorm更新） ==========
        with torch.no_grad() if not self.training or all(not p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            features = self.backbone.get_intermediate_layers(
                x, n = self.intermediate_layer_idx[self.encoder_size]
            )
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
        return out