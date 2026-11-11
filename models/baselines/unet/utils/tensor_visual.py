import torch
import matplotlib.pyplot as plt

# 假设 gt2D 是一个四维张量
gt2D = torch.randn(1, 3, 256, 256)  # 形状 (1, 3, 256, 256)，模拟一个张量

# 选择第二个通道
channel_data = gt2D[0, 1, :, :]  # 索引 [0, 1, :, :] 获取第一批次和第二个通道的数据

# 转换为 NumPy 数组并可视化
plt.imshow(channel_data.numpy(), cmap='gray')  # 使用灰度图进行可视化
plt.colorbar()
plt.title("Visualization of gt2D[:, 1, :, :]")
plt.show()
