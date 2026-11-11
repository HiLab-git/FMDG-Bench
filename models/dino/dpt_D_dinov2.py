# dpt_D_14_518.py
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
            if x.dim() == 3:  # [B, L, C]
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
        encoder_size='s',
        nclass=1,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
        backbone=None,
        freeze_backbone: bool = True
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            's': [2, 5, 8, 11],  # small (12 layers)
            'b': [2, 5, 8, 11],  # base (12 layers)
        }
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
        
        if freeze_backbone and self.backbone is not None:
            self.lock_backbone()

    def lock_backbone(self):
        if self.backbone is None:
            return
        for p in self.backbone.parameters():
            p.requires_grad = False
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.Dropout, nn.Dropout2d)):
                m.eval()
        print("✅ Backbone (encoder) has been frozen completely!")

    def forward(self, x):
        # 使用patch_size=14
        patch_size = 14
        patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size
        
        # 检查输入尺寸
        expected_h, expected_w = 518, 518
        if x.shape[-2] != expected_h or x.shape[-1] != expected_w:
            print(f"WARNING: Input size {x.shape[-2:]} doesn't match expected {expected_h}x{expected_w}")
            print(f"Resizing input to {expected_h}x{expected_w}")
            x = F.interpolate(x, size=(expected_h, expected_w), mode='bilinear', align_corners=True)
            patch_h, patch_w = expected_h // patch_size, expected_w // patch_size
        
        # 确保输入能被14整除
        if x.shape[-2] % patch_size != 0 or x.shape[-1] % patch_size != 0:
            raise ValueError(f"Input size {x.shape[-2:]} must be divisible by patch_size={patch_size}")
        
        # DINOv2需要3通道输入
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]} channels")
        
        # 获取中间层特征
        features = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size], reshape=True, return_class_token=False
        )
        
        if len(features) != 4:
            raise ValueError(f"Expected 4 feature layers, got {len(features)}")
        
        # 将特征传递给解码器
        out = self.head(features, patch_h, patch_w)
        
        # 上采样
        out = F.interpolate(out, (patch_h * patch_size, patch_w * patch_size), mode='bilinear', align_corners=True)
        return out