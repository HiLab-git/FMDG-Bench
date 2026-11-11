# -*- coding: utf-8 -*-
"""
模型工厂：根据模型名称返回对应的模型实例。
所有模型统一接口: (B, in_channels, H, W) -> (B, out_channels, H, W)
训练模式下可能返回 list/tuple (deep supervision), 推理模式返回单一 Tensor。
"""

from .unetpp import UNetPlusPlusModel
from .attention_unet import AttentionUNetModel
from .nnunet import nnUNetModel
from .transunet import TransUNetModel


_MODEL_REGISTRY = {
    "unetpp":         UNetPlusPlusModel,
    "attention_unet": AttentionUNetModel,
    "nnunet":         nnUNetModel,
    "transunet":      TransUNetModel,
}


def get_model(model_name: str, in_channels: int = 3, out_channels: int = 2, **kwargs):
    """
    参数:
        model_name: "unetpp" | "attention_unet" | "nnunet"
        in_channels:  输入通道数 (默认 3)
        out_channels: 输出类别数 (2 or 3)
        **kwargs:     传递给具体模型构造函数的额外参数
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[model_name](
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def list_models():
    """返回所有可用模型名称"""
    return list(_MODEL_REGISTRY.keys())
