# clip_feature_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPFeatureExtractor(nn.Module):
    """
    完整的CLIP特征提取器
    提取4个中间层特征，与DPT结构匹配
    """
    def __init__(self, clip_visual, model_name="ViT-B/16"):
        super(CLIPFeatureExtractor, self).__init__()
        self.clip_visual = clip_visual
        self.model_name = model_name
        
        # 根据模型类型设置参数
        if "ViT-B/16" in model_name:
            self.patch_size = 16
            self.embed_dim = 768
            self.num_layers = 12
            # 选择中间层索引，与DINOv2保持一致
            self.intermediate_layer_idx = [2, 5, 8, 11]
        elif "ViT-B/32" in model_name:
            self.patch_size = 32
            self.embed_dim = 768
            self.num_layers = 12
            self.intermediate_layer_idx = [2, 5, 8, 11]
        elif "ViT-L/14" in model_name or "ViT-L/14@336px" in model_name:
            self.patch_size = 14
            self.embed_dim = 1024
            self.num_layers = 24
            # 对于24层的模型，调整中间层索引
            self.intermediate_layer_idx = [4, 11, 17, 23]
        else:
            # 默认值
            self.patch_size = 16
            self.embed_dim = 768
            self.num_layers = 12
            self.intermediate_layer_idx = [2, 5, 8, 11]
        
        # 注册钩子来捕获中间层特征
        self.handles = []
        self.intermediate_features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册钩子以捕获中间层特征"""
        def make_hook(layer_idx):
            def hook(module, input, output):
                # 保存中间层特征
                self.intermediate_features[layer_idx] = output
            return hook
        
        # 为每个指定的层注册钩子
        for idx in self.intermediate_layer_idx:
            if idx < len(self.clip_visual.transformer.resblocks):
                handle = self.clip_visual.transformer.resblocks[idx].register_forward_hook(
                    make_hook(idx)
                )
                self.handles.append(handle)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def forward(self, x):
        """
        前向传播，提取中间层特征
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            intermediate_features: 4个中间层特征的列表，每个形状为 [B, L, C]
            patch_h: patch网格高度
            patch_w: patch网格宽度
        """
        # 清空前一次的特征
        self.intermediate_features = {}
        
        B, C, H, W = x.shape
        
        # 确保输入尺寸能被patch_size整除
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            new_H = ((H // self.patch_size) * self.patch_size)
            new_W = ((W // self.patch_size) * self.patch_size)
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=True)
            H, W = new_H, new_W
        
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        num_patches = patch_h * patch_w
        
        # CLIP的前向传播
        with torch.no_grad():
            # 1. 卷积提取patch特征
            x = self.clip_visual.conv1(x)  # [B, embed_dim, patch_h, patch_w]
            
            # 2. 重塑为序列
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, embed_dim, N]
            x = x.permute(0, 2, 1)  # [B, N, embed_dim]
            
            # 3. 添加位置编码
            pos_embed = self.clip_visual.positional_embedding  # [num_patches + 1, embed_dim]
            
            # 处理位置编码维度
            if pos_embed.dim() == 2:
                pos_embed = pos_embed.unsqueeze(0)  # [1, num_patches + 1, embed_dim]
            
            # 检查位置编码形状
            expected_pos_embed_shape = (1, num_patches + 1, self.embed_dim)
            if pos_embed.shape != torch.Size(expected_pos_embed_shape):
                # 插值位置编码
                if pos_embed.shape[1] == 1:
                    # 只有[CLS] token
                    cls_token = pos_embed[:, :1, :]
                    patch_tokens = torch.zeros(1, num_patches, self.embed_dim, 
                                              device=pos_embed.device, dtype=pos_embed.dtype)
                    pos_embed = torch.cat([cls_token, patch_tokens], dim=1)
                else:
                    # 插值patch tokens
                    cls_token = pos_embed[:, :1, :]
                    patch_tokens = pos_embed[:, 1:, :]
                    
                    # 计算原始patch网格
                    orig_patches = patch_tokens.shape[1]
                    orig_grid = int(orig_patches ** 0.5)
                    
                    # 重塑为2D
                    if orig_grid * orig_grid == orig_patches:
                        patch_tokens = patch_tokens.reshape(1, orig_grid, orig_grid, -1)
                        patch_tokens = patch_tokens.permute(0, 3, 1, 2)  # [1, C, H, W]
                        
                        # 插值
                        patch_tokens = F.interpolate(
                            patch_tokens, size=(patch_h, patch_w), 
                            mode='bicubic', align_corners=False
                        )
                        
                        # 重塑回序列
                        patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)
                        pos_embed = torch.cat([cls_token, patch_tokens], dim=1)
            
            x = x + pos_embed[:, 1:, :]  # 添加patch位置编码
            
            # 4. 添加[CLS] token
            cls_tokens = self.clip_visual.class_embedding.expand(B, 1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # 5. LayerNorm
            x = self.clip_visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            
            # 6. 通过transformer（会触发钩子）
            for block in self.clip_visual.transformer.resblocks:
                x = block(x)
            
            x = x.permute(1, 0, 2)  # LND -> NLD
        
        # 收集中间层特征
        intermediate_features = []
        for idx in self.intermediate_layer_idx:
            if idx in self.intermediate_features:
                feat = self.intermediate_features[idx]
                # 转换为 [B, L, C] 格式
                if feat.dim() == 3 and feat.shape[0] == num_patches + 1:
                    # 已经是 [L+1, B, C] 格式
                    feat = feat.permute(1, 0, 2)  # [B, L+1, C]
                    feat = feat[:, 1:, :]  # 去掉[CLS] token
                elif feat.dim() == 3 and feat.shape[1] == B:
                    # [L+1, B, C] 格式
                    feat = feat.permute(1, 0, 2)  # [B, L+1, C]
                    if feat.shape[1] == num_patches + 1:
                        feat = feat[:, 1:, :]  # 去掉[CLS] token
                intermediate_features.append(feat)
            else:
                # 如果特征不存在，使用最后一层特征
                feat = x[:, 1:, :]  # 去掉[CLS] token
                intermediate_features.append(feat)
        
        # 确保有4个特征
        while len(intermediate_features) < 4:
            intermediate_features.append(intermediate_features[-1] if intermediate_features else x[:, 1:, :])
        
        return intermediate_features, patch_h, patch_w