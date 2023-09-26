import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import random

def dark_channel(image, window_size):
    min_pool = nn.MaxPool2d(window_size, stride=1, padding=window_size // 2)
    dark_channel = min_pool(-image)
    output = -torch.min(dark_channel, dim=1)[0].unsqueeze(1)
    return output

# 图像路径和文件名
image_dir = '../datasets/train/input'
image_filenames = ['1_Hazy.jpg', '2_Hazy.jpg', '3_Hazy.jpg', '4_Hazy.jpg', '5_Hazy.jpg',
                   '6_Hazy.jpg', '7_Hazy.jpg', '8_Hazy.jpg', '9_Hazy.jpg', '10_Hazy.jpg']

# 遍历图像列表
for filename in image_filenames:
    # 构建图像路径
    image_path = os.path.join(image_dir, filename)

    # 加载图像
    image = Image.open(image_path).convert('RGB')

    # 随机截取图像的一部分
    width, height = image.size
    crop_size = 64
    x = random.randint(0, width - crop_size)
    y = random.randint(0, height - crop_size)
    image = TF.crop(image, y, x, crop_size, crop_size)

    # 转换为Tensor并进行归一化
    image = TF.to_tensor(image)
    image = torch.clamp(image, 0, 1)  # 将图像数据限制在[0, 1]范围内

    # 计算暗通道
    window_size = 15
    dark_ch = dark_channel(image.unsqueeze(0), window_size)

    # 将图像数据从张量转换为NumPy数组，并进行适当的处理
    input_img = image.permute(1, 2, 0).cpu().numpy()
    input_img = torch.from_numpy(input_img)  # 将numpy.ndarray转换为Tensor
    input_img = torch.clamp(input_img, 0, 1)  # 将图像数据限制在[0, 1]范围内

    output_img = dark_ch[0].squeeze().cpu().numpy()

    # 绘制图像
    plt.subplot(1, 2, 1)
    plt.imshow(input_img)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_img, cmap='gray')
    plt.title('Dark Channel')
    plt.axis('off')

    plt.show()