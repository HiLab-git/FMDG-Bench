# clip_feature_extractor_pubmed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CLIPFeatureExtractor(nn.Module):
    """
    支持 PubMedCLIP 的特征提取器
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
            self.intermediate_layer_idx = [4, 11, 17, 23]
        else:
            # 默认值
            self.patch_size = 16
            self.embed_dim = 768
            self.num_layers = 12
            self.intermediate_layer_idx = [2, 5, 8, 11]
        
        # 检查是否是 PubMedCLIP
        self.is_pubmed_clip = hasattr(self.clip_visual, 'model')
        
        # 注册钩子来捕获中间层特征
        self.handles = []
        self.intermediate_features = {}
        
        if not self.is_pubmed_clip:
            # 标准 CLIP
            self._register_hooks_standard()
        else:
            # PubMedCLIP
            self._register_hooks_pubmed()
    
    def _register_hooks_standard(self):
        """为标准 CLIP 注册钩子"""
        def make_hook(layer_idx):
            def hook(module, input, output):
                self.intermediate_features[layer_idx] = output
            return hook
        
        for idx in self.intermediate_layer_idx:
            if idx < len(self.clip_visual.transformer.resblocks):
                handle = self.clip_visual.transformer.resblocks[idx].register_forward_hook(
                    make_hook(idx)
                )
                self.handles.append(handle)
    
    def _register_hooks_pubmed(self):
        """为 PubMedCLIP 注册钩子"""
        def make_hook(layer_idx):
            def hook(module, input, output):
                # PubMedCLIP 的输出可能是元组或 BaseModelOutput
                # 确保我们获取的是张量
                if isinstance(output, tuple):
                    # 元组输出，通常第一个元素是隐藏状态
                    self.intermediate_features[layer_idx] = output[0]
                elif hasattr(output, 'last_hidden_state'):
                    # BaseModelOutputWithPooling
                    self.intermediate_features[layer_idx] = output.last_hidden_state
                else:
                    # 其他情况
                    self.intermediate_features[layer_idx] = output
            return hook
        
        # PubMedCLIP 的 transformer 层
        if hasattr(self.clip_visual.model, 'vision_model'):
            transformer_layers = self.clip_visual.model.vision_model.encoder.layers
        else:
            # 后备方案
            transformer_layers = self.clip_visual.model.encoder.layers
        
        for idx in self.intermediate_layer_idx:
            if idx < len(transformer_layers):
                handle = transformer_layers[idx].register_forward_hook(
                    make_hook(idx)
                )
                self.handles.append(handle)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def _forward_standard_clip(self, x, patch_h, patch_w):
        """标准 CLIP 前向传播"""
        B, C, H, W = x.shape
        
        # 1. 卷积提取patch特征
        x = self.clip_visual.conv1(x)
        
        # 2. 重塑为序列
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # 3. 添加位置编码
        pos_embed = self.clip_visual.positional_embedding
        
        if pos_embed.dim() == 2:
            pos_embed = pos_embed.unsqueeze(0)
        
        expected_pos_embed_shape = (1, patch_h * patch_w + 1, self.embed_dim)
        if pos_embed.shape != torch.Size(expected_pos_embed_shape):
            cls_token = pos_embed[:, :1, :]
            patch_tokens = F.interpolate(
                pos_embed[:, 1:, :].reshape(1, -1, int(math.sqrt(pos_embed.shape[1]-1)), int(math.sqrt(pos_embed.shape[1]-1))).permute(0, 3, 1, 2),
                size=(patch_h, patch_w),
                mode='bicubic',
                align_corners=False
            ).reshape(1, -1, patch_h * patch_w).permute(0, 2, 1)
            pos_embed = torch.cat([cls_token, patch_tokens], dim=1)
        
        x = x + pos_embed[:, 1:, :]
        
        # 4. 添加[CLS] token
        cls_tokens = self.clip_visual.class_embedding.expand(B, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 5. LayerNorm
        x = self.clip_visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        
        # 6. 通过transformer
        for block in self.clip_visual.transformer.resblocks:
            x = block(x)
        
        x = x.permute(1, 0, 2)
        
        return x
    
    def _forward_pubmed_clip(self, x, patch_h, patch_w):
        """PubMedCLIP 前向传播"""
        B, C, H, W = x.shape
        
        # PubMedCLIP 使用 CLIPVisionModel
        with torch.no_grad():
            outputs = self.clip_visual.model(x, output_hidden_states=True)
        
        # 获取最后一层输出
        x = outputs.last_hidden_state
        
        return x
    
    def forward(self, x):
        """
        前向传播，提取中间层特征
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
        
        # 执行前向传播
        if self.is_pubmed_clip:
            x = self._forward_pubmed_clip(x, patch_h, patch_w)
        else:
            x = self._forward_standard_clip(x, patch_h, patch_w)
        
        # 收集中间层特征
        intermediate_features = []
        
        for idx in self.intermediate_layer_idx:
            if idx in self.intermediate_features:
                feat = self.intermediate_features[idx]
                
                # 确保 feat 是张量，不是元组
                if isinstance(feat, tuple):
                    # 如果是元组，取第一个元素（通常是隐藏状态）
                    feat = feat[0]
                
                # 处理特征形状
                if hasattr(feat, 'dim') and feat.dim() == 3:  # 确保是张量且有 dim 属性
                    # [B, L, C] 格式
                    if feat.shape[1] > patch_h * patch_w:
                        # 包含 cls token
                        feat = feat[:, 1:, :]  # 去掉[CLS] token
                intermediate_features.append(feat)
            else:
                # 如果特征不存在，使用最后一层特征
                feat = x
                if hasattr(feat, 'dim') and feat.dim() == 3 and feat.shape[1] > patch_h * patch_w:
                    feat = feat[:, 1:, :]  # 去掉[CLS] token
                intermediate_features.append(feat)
        
        # 确保有4个特征
        while len(intermediate_features) < 4:
            if intermediate_features:
                intermediate_features.append(intermediate_features[-1])
            else:
                # 创建一个空特征
                intermediate_features.append(torch.zeros(B, patch_h * patch_w, self.embed_dim, device=x.device))
        
        return intermediate_features, patch_h, patch_w