# -*- coding: utf-8 -*-
"""
UNet++ (Nested UNet) 实现
基于 segmentation_models_pytorch 库, 使用 ResNet34 作为 encoder。
训练时启用 deep supervision (返回 list of Tensors), 推理时返回单一 Tensor。

依赖: pip install segmentation-models-pytorch
"""

import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not installed. UNet++ will not be available.")


class UNetPlusPlusModel(nn.Module):
    """
    UNet++ with ResNet34 encoder and ImageNet pretrained weights.

    训练模式 (model.train()):
        forward() 返回 list [main_output, ds2, ds3, ds4, ds5]
        其中 main_output 是最高分辨率输出, ds* 是深层监督输出。
    推理模式 (model.eval()):
        forward() 返回单一 Tensor [B, out_channels, H, W]
    """

    def __init__(self, in_channels=3, out_channels=2, encoder_name="resnet34"):
        super().__init__()
        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation-models-pytorch is required for UNet++. "
                "Install with: pip install segmentation-models-pytorch"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
            activation=None,  # 输出原始 logits, 由训练脚本统一处理 softmax
        )

    def forward(self, image):
        """
        参数:
            image: [B, in_channels, H, W]
        返回:
            训练模式: list of [B, out_channels, H', W'] (不同分辨率)
            推理模式: [B, out_channels, H, W]
        """
        output = self.model(image)

        if self.training:
            # SMP UnetPlusPlus 在训练时返回 (main_output, [ds2, ds3, ds4, ds5])
            # 我们将它们统一为 list 供外部 loss 函数处理
            if isinstance(output, (list, tuple)):
                return list(output)
            else:
                # 如果 SMP 版本不同, 也可能在主输出外还有额外输出
                return [output]

        # 推理时只返回主输出
        if isinstance(output, (list, tuple)):
            return output[0]
        return output
