import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PubMedCLIPDPT(nn.Module):
    """专门针对 PubMedCLIP 的 DPT 实现"""
    
    def __init__(self, clip_backbone, nclass=1, input_size=224, freeze_backbone=True):
        super().__init__()
        
        self.backbone = clip_backbone
        self.input_size = input_size
        self.nclass = nclass
        
        # 冻结骨干网络
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # PubMedCLIP ViT-B/32 的配置
        self.hidden_dim = 768
        self.num_tokens = 50  # 1 cls token + 49 spatial tokens
        self.grid_size = 7    # sqrt(49) = 7
        
        print(f"PubMedCLIP DPT 配置:")
        print(f"  Hidden dimension: {self.hidden_dim}")
        print(f"  Number of tokens: {self.num_tokens}")
        print(f"  Grid size: {self.grid_size}x{self.grid_size}")
        
        # 解码器通道
        decoder_channels = [256, 128, 64, 32]
        
        # 解码器网络
        self.decoder = nn.ModuleList()
        in_channels = self.hidden_dim
        
        for i, out_channels in enumerate(decoder_channels):
            decoder_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
            self.decoder.append(decoder_block)
            in_channels = out_channels
        
        # 最终分类层
        if nclass == 1:
            self.classifier = nn.Conv2d(decoder_channels[-1], 1, 1)
        else:
            self.classifier = nn.Conv2d(decoder_channels[-1], nclass, 1)
        
        # 上采样到原始尺寸
        self.final_upsample = nn.Upsample(
            size=(input_size, input_size), mode='bilinear', align_corners=False
        )
    
    def extract_features(self, x):
        """从 PubMedCLIP 中提取特征"""
        with torch.no_grad():
            if hasattr(self.backbone, 'model'):
                # PubMedCLIP
                outputs = self.backbone.model(x)
                if hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state
                else:
                    features = outputs
            else:
                # 标准 CLIP
                features = self.backbone(x)
        
        return features
    
    def reshape_to_2d(self, vit_features):
        """
        将 ViT 特征重塑为 2D 特征图
        输入: [B, 50, 768] (1 cls token + 49 spatial tokens)
        输出: [B, 768, 7, 7]
        """
        batch_size, num_tokens, channels = vit_features.shape
        
        # 移除 cls token (第一个 token)
        spatial_tokens = vit_features[:, 1:, :]  # [B, 49, 768]
        
        # 重塑为 2D
        cnn_features = spatial_tokens.permute(0, 2, 1).reshape(
            batch_size, channels, self.grid_size, self.grid_size
        )
        
        return cnn_features
    
    def forward(self, x):
        # 确保输入在与模型相同的设备上
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        
        # 提取特征
        vit_features = self.extract_features(x)
        
        # 重塑为 2D
        cnn_features = self.reshape_to_2d(vit_features)
        
        # 通过解码器
        features = cnn_features
        for decoder_block in self.decoder:
            features = decoder_block(features)
        
        # 分类
        output = self.classifier(features)
        
        # 上采样到原始尺寸
        if output.shape[-2:] != (self.input_size, self.input_size):
            output = self.final_upsample(output)
        
        return output


class SimpleDPTForPubMedCLIP(nn.Module):
    """更简单的 DPT 实现，专门为 PubMedCLIP"""
    
    def __init__(self, clip_backbone, nclass=1, input_size=224, freeze_backbone=True):
        super().__init__()
        
        self.backbone = clip_backbone
        self.input_size = input_size
        self.nclass = nclass
        
        # 冻结骨干网络
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # PubMedCLIP ViT-B/32 的配置
        self.hidden_dim = 768
        self.grid_size = 7
        
        print(f"Simple DPT for PubMedCLIP:")
        print(f"  Input size: {input_size}x{input_size}")
        print(f"  Hidden dimension: {self.hidden_dim}")
        print(f"  Grid size: {self.grid_size}x{self.grid_size}")
        
        # 特征提取和重塑
        self.feature_extractor = nn.Sequential(
            # 1x1 卷积调整通道
            nn.Conv2d(self.hidden_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 上采样路径
        self.upsample_path = nn.Sequential(
            # 7x7 -> 14x14
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 14x14 -> 28x28
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 28x28 -> 56x56
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 56x56 -> 112x112
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 112x112 -> 224x224
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
        # 分类头
        if nclass == 1:
            self.classifier = nn.Conv2d(8, 1, 1)
        else:
            self.classifier = nn.Conv2d(8, nclass, 1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 提取特征
        with torch.no_grad():
            if hasattr(self.backbone, 'model'):
                outputs = self.backbone.model(x)
                if hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state
                else:
                    features = outputs
            else:
                features = self.backbone(x)
        
        # 重塑为 2D: [B, 50, 768] -> [B, 768, 7, 7]
        # 移除 cls token
        spatial_features = features[:, 1:, :]  # [B, 49, 768]
        cnn_features = spatial_features.permute(0, 2, 1).reshape(
            batch_size, self.hidden_dim, self.grid_size, self.grid_size
        )
        
        # 特征提取
        x = self.feature_extractor(cnn_features)
        
        # 上采样路径
        x = self.upsample_path(x)
        
        # 分类
        x = self.classifier(x)
        
        return x


# 测试函数
def test_pubmed_clip_dpt():
    """测试 PubMedCLIP DPT"""
    import clip
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 创建测试输入
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    print(f"Test input shape: {test_input.shape}")
    
    # 加载标准 CLIP 作为骨干网络
    model, _ = clip.load("ViT-B/32", device=device)
    model = model.float()
    backbone = model.visual
    backbone.eval()
    
    # 创建 PubMedCLIP DPT
    dpt_model = PubMedCLIPDPT(
        clip_backbone=backbone,
        nclass=1,
        input_size=224
    ).to(device)
    
    # 测试前向传播
    with torch.no_grad():
        output = dpt_model(test_input)
        print(f"DPT output shape: {output.shape}")
        print(f"DPT output range: [{output.min():.4f}, {output.max():.4f}]")
    
    return dpt_model


if __name__ == "__main__":
    model = test_pubmed_clip_dpt()