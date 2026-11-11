import torch

def dice_coefficient(pred, target):
    """
    计算单个前景类别的 Dice 系数

    Parameters:
    - pred (torch.Tensor): 模型预测的掩码，形状为 [B, H, W]，其中 B 是 batch size，H 和 W 是图像的高和宽。
    - target (torch.Tensor): 真实标签，形状与 pred 相同。

    Returns:
    - dice (torch.Tensor): 单个前景类别的 Dice 系数，形状为 [B]。
    """
    intersection = (pred * target).sum(dim=(1, 2))  # 对 H 和 W 维度求和
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))  # 对 H 和 W 维度求和
    dice = (2. * intersection + 1e-6) / (union + 1e-6)  # 加上小常数避免除以零
    return dice


def calculate_dice_for_foregrounds(pred, mask):
    """
    计算前景1和前景2的Dice系数

    Parameters:
    - pred (torch.Tensor): 模型预测的掩码，形状为 [B, 3, H, W]，其中 B 是批次大小，3 是类别数（背景、前景1、前景2），H 和 W 是图像的高和宽。
    - mask (torch.Tensor): 真实的掩码，形状与 pred 相同。

    Returns:
    - dice_fg1 (torch.Tensor): 前景1的 Dice 系数。
    - dice_fg2 (torch.Tensor): 前景2的 Dice 系数。
    """
    # 前景1：合并通道1和通道2作为前景1
    pred_fg1 = (pred[:, 1] + pred[:, 2]) > 0  # 前景1：预测中通道1和通道2的联合
    mask_fg1 = (mask[:, 1] + mask[:, 2]) > 0  # 前景1：目标中通道1和通道2的联合

    # 前景2：仅使用通道2
    pred_fg2 = pred[:, 2] > 0  # 前景2：预测中通道2
    mask_fg2 = mask[:, 2] > 0  # 前景2：目标中通道2

    print(pred_fg2.shape)
    # 计算前景1和前景2的 Dice 系数
    dice_fg1 = dice_coefficient(pred_fg1, mask_fg1)  # 计算前景1的 Dice 系数
    dice_fg2 = dice_coefficient(pred_fg2, mask_fg2)  # 计算前景2的 Dice 系数

    return dice_fg1, dice_fg2

# 示例使用
# 假设pred和mask的形状为 [B, 3, H, W]，B是批量大小，3是类别数（背景、前景1、前景2），H和W是图像的高和宽
pred = torch.randint(0, 3, (2, 3, 800, 800)).float()  # 2个样本，3个类别，800x800的图像
mask = torch.randint(0, 3, (2, 3, 800, 800)).float()

dice_fg1, dice_fg2 = calculate_dice_for_foregrounds(pred, mask)
print("Dice for Foreground 1:", dice_fg1)
print("Dice for Foreground 2:", dice_fg2)
