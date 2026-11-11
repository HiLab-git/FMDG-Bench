# dpt_clip_full_compatible.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip_feature_extractor_pubmed import CLIPFeatureExtractor

class DPT_CLIP_Full_Compatible(nn.Module):
    """
    兼容 PubMedCLIP 的完整 DPT 模型
    """
    def __init__(
        self, 
        nclass=1,
        features=128, 
        out_channels=[96, 192, 384, 768],
        use_bn=False,
        clip_model=None,
        model_name="ViT-B/16",
        freeze_backbone: bool = True
    ):
        super(DPT_CLIP_Full_Compatible, self).__init__()
        
        # 特征提取器
        self.feature_extractor = CLIPFeatureExtractor(clip_model, model_name)
        
        # 根据模型类型设置参数
        if "ViT-B" in model_name:
            embed_dim = 768
        elif "ViT-L" in model_name:
            embed_dim = 1024
        else:
            embed_dim = 768
        
        # 简单的 DPT 解码器
        self.head = DPTHead_Compatible(
            nclass=nclass,
            in_channels=embed_dim,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels
        )
        
        # 冻结backbone
        if freeze_backbone and clip_model is not None:
            self.lock_backbone()
    
    def lock_backbone(self):
        """冻结 CLIP backbone"""
        if hasattr(self.feature_extractor.clip_visual, 'model'):
            # PubMedCLIP
            for param in self.feature_extractor.clip_visual.model.parameters():
                param.requires_grad = False
        else:
            # 标准 CLIP
            for param in self.feature_extractor.clip_visual.parameters():
                param.requires_grad = False
        
        print("✅ CLIP backbone (encoder) has been frozen completely!")
    
    def unlock_backbone(self, lr: float = 1e-5):
        """解冻backbone"""
        if hasattr(self.feature_extractor.clip_visual, 'model'):
            # PubMedCLIP
            for param in self.feature_extractor.clip_visual.model.parameters():
                param.requires_grad = True
        else:
            # 标准 CLIP
            for param in self.feature_extractor.clip_visual.parameters():
                param.requires_grad = True
        
        print(f"✅ CLIP backbone (encoder) has been unlocked (finetune lr: {lr})!")
        return lr
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            分割结果 [B, nclass, H, W]
        """
        B, C, H, W = x.shape
        
        # 确保输入是float32
        if x.dtype != torch.float32:
            x = x.float()
        
        # CLIP需要3通道输入
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        elif C != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {C} channels")
        
        # 获取中间层特征
        features, patch_h, patch_w = self.feature_extractor(x)
        
        # 通过DPT解码器
        out = self.head(features, patch_h, patch_w)
        
        # 上采样到原始输入尺寸
        out = F.interpolate(
            out, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=True
        )
        
        return out
    
    def __del__(self):
        """析构函数，移除钩子"""
        if hasattr(self.feature_extractor, 'remove_hooks'):
            self.feature_extractor.remove_hooks()


class DPTHead_Compatible(nn.Module):
    """
    兼容的 DPT 解码器头
    """
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead_Compatible, self).__init__()
        
        # 投影层
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        # 特征融合层
        self.fuse_layers = nn.ModuleList()
        for i in range(4):
            fuse_layer = nn.Sequential(
                nn.Conv2d(out_channels[i], features, 3, padding=1),
                nn.BatchNorm2d(features) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True)
            )
            self.fuse_layers.append(fuse_layer)
        
        # 输出卷积层
        self.output_conv = nn.Sequential(
            nn.Conv2d(features * 4, features, 3, padding=1),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, nclass, 1)
        )
    
    def forward(self, out_features, patch_h, patch_w):
        """
        前向传播
        """
        processed_features = []
        
        # 处理每个特征层
        for i, x in enumerate(out_features):
            B, L, C = x.shape
            
            # 重塑为空间特征图 [B, C, patch_h, patch_w]
            x = x.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
            
            # 1x1卷积投影
            x = self.projects[i](x)
            
            # 特征融合
            x = self.fuse_layers[i](x)
            processed_features.append(x)
        
        # 上采样所有特征到第一层的尺寸
        target_size = processed_features[0].shape[-2:]
        upsampled_features = []
        
        for i, feat in enumerate(processed_features):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            upsampled_features.append(feat)
        
        # 拼接特征
        fused = torch.cat(upsampled_features, dim=1)
        
        # 最终输出
        out = self.output_conv(fused)
        
        return out