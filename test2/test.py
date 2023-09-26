import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 加载和预处理输入图片
image_path = '../datasets/train/input/1_Hazy.jpg'  # 替换为你的图片路径
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.ToTensor()
])
image_tensor = transform(image)
image_tensor = torch.unsqueeze(image_tensor, 0)  # 添加批次维度

# 对图片进行功能处理
max_channel = torch.argmax(image_tensor[:, :3, :, :], dim=1)
max_channel = torch.unsqueeze(max_channel, 1)
max_channel = max_channel.int()
plt.figure(figsize=(8, 8))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
plt.title('Processed Image')
plt.imshow(max_channel[0, 0, :, :], cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()