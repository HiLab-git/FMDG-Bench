import torch

def multi_class_dice(pred, target, num_classes):
    """
    计算每个前景类别的Dice系数。

    Parameters:
    - pred (torch.Tensor): 模型预测的掩码，形状为 [B, C, H, W]，其中 B 是 batch size，C 是类别数，H 和 W 是图像的高和宽。
    - target (torch.Tensor): 真实标签，形状与 pred 相同。
    - num_classes (int): 类别数，通常包括背景和前景类别。

    Returns:
    - dice_scores (torch.Tensor): 每个类别的 Dice 系数，形状为 [num_classes]。
    """
    dice_scores = []

    # 遍历每个类别，计算该类别的 Dice 系数
    for c in range(1, num_classes):  # 从 1 开始，跳过背景类别
        # 获取当前类别的预测值和真实值
        pred_class = pred[:, c, :, :]  # 获取预测中当前类别的通道
        target_class = target[:, c, :, :]  # 获取目标中当前类别的通道
        # print(c)
        # print(pred_class.shape)
        # print(target_class.shape)
        # 计算交集
        intersection = (pred_class * target_class).sum(dim=[1, 2])  # 对 height 和 width 维度求和
        union = pred_class.sum(dim=[1, 2]) + target_class.sum(dim=[1, 2])  # 对 height 和 width 维度求和
        
        # 计算 Dice 系数
        dice = (2. * intersection + 1e-6) / (union + 1e-6)  # 加上小常数避免除以零
        
        # 将当前类别的 Dice 系数添加到列表
        dice_scores.append(dice)

    # 将所有类别的 Dice 系数堆叠成一个 tensor，并返回
    return torch.stack(dice_scores)

# 示例使用
# 假设预测和目标都有 3 个类别（背景 + 前景1 + 前景2）
# pred 和 target 形状为 [batch_size, num_classes, height, width]

pred = torch.randint(0, 2, (3, 3, 800, 800)).float()  # 2 个样本，3 类，800x800 的图像
target = torch.randint(0, 2, (3, 3, 800, 800)).float()

num_classes = 3  # 背景 + 2 个前景类别
dice_scores = multi_class_dice(pred, target, num_classes)

print(dice_scores.shape)  # 输出每个类别的 Dice 系数