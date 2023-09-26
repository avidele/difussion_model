import torch

# 假设你的输入是一个名为image的PyTorch张量
image = torch.randn(32, 3, 64, 64)

# 沿着通道维度计算最大值
max_channel = torch.argmax(image, dim=1)
max_channel = max_channel.unsqueeze(1)
max_channel = max_channel.int()

# 显示颜色极大值通道
print(max_channel.shape)