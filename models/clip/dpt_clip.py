# dpt_clip_full.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip_feature_extractor import CLIPFeatureExtractor
from dpt_head import DPTHead

class DPT_CLIP_Full(nn.Module):
    """
    完整的DPT+CLIP模型
    结构与DINOv2版本完全一致
    """
    def __init__(
        self, 
        encoder_size='b',  # 为了与DINOv2参数名一致，但CLIP不使用这个参数
        nclass=1,
        features=128, 
        out_channels=[96, 192, 384, 768],  # 与DINOv2 DPT保持一致
        use_bn=False,
        clip_model=None,
        model_name="ViT-B/16",
        freeze_backbone: bool = True
    ):
        super(DPT_CLIP_Full, self).__init__()
        
        # 特征提取器
        self.feature_extractor = CLIPFeatureExtractor(clip_model, model_name)
        
        # 根据模型类型设置参数
        if "ViT-B" in model_name:
            embed_dim = 768
        elif "ViT-L" in model_name:
            embed_dim = 1024
        else:
            embed_dim = 768
        
        # DPT解码器头
        self.head = DPTHead(
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
        """冻结CLIP backbone的所有参数"""
        for param in self.feature_extractor.clip_visual.parameters():
            param.requires_grad = False
        
        # 设置eval模式
        for m in self.feature_extractor.clip_visual.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.Dropout, nn.Dropout2d)):
                m.eval()
        
        print("✅ CLIP backbone (encoder) has been frozen completely!")
    
    def unlock_backbone(self, lr: float = 1e-5):
        """解冻backbone"""
        for param in self.feature_extractor.clip_visual.parameters():
            param.requires_grad = True
        
        for m in self.feature_extractor.clip_visual.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.Dropout, nn.Dropout2d)):
                m.train()
        
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
        self.feature_extractor.remove_hooks()