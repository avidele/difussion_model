import cv2
import numpy as np


def maximum_contrast_enhancement(image, s=3, r=7):
    enhanced_images = []
    for i in range(image.shape[0]):
        img = image[i]
        enhanced_image = np.zeros_like(img, dtype=np.float32)
        padded_image = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                patch = padded_image[i:i + s, j:j + s]
                local_contrast_values = []
                for m in range(i, i + r):
                    for n in range(j, j + r):
                        local_patch = padded_image[m:m + s, n:n + s]
                        local_contrast_values.append(np.square(local_patch - patch).mean())
                max_contrast = max(local_contrast_values)
                enhanced_image[i, j] = max_contrast

        enhanced_images.append(enhanced_image)

    enhanced_images = np.array(enhanced_images)

    enhanced_images = (enhanced_images - np.min(enhanced_images)) / (np.max(enhanced_images) - np.min(enhanced_images))
    enhanced_images = (enhanced_images * 255).astype(np.uint8)

    return enhanced_images

# 在训练循环中
# 在训练循环中
def noise_estimation_loss(model, x0, t, e, b):
    # ...
    #dark_ch = DarkChannel.calculate_dark(x0[:, 3:, :, :])
    # x_input = x0[:, 3:, :, :].cpu().detach().numpy()
    # print(x_input)
    # dark_ch = darkchannel.calculate_dark(x_input)
    # dark_ch = torch.from_numpy(dark_ch)
    # 计算暗通道
    dark_ch = dark_channel(x0[:, :3, :, :], window_size=15)
    output = dark_ch
    print("output", output.shape)
    # max_channel = torch.argmax(x0[:, :3, :, :], dim=1)
    # max_channel = max_channel.unsqueeze(1)
    # max_channel = max_channel.int()
    # 将形状转换为[32, 3, 64, 64]
    # output = torch.squeeze(input, dim=2)
    # output = torch.nn.functional.interpolate(output, size=(64, 64), mode='bilinear', align_corners=False)
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)  # 通过扩散步数 t 计算权重信息
    x = x0[:, :3, :, :] * a.sqrt() + e * (1.0 - a).sqrt()  # 计算噪声估计值
    x0_np = x0[:, :3, :, :].cpu().numpy()
    print("x0_np.shape", x0_np.shape)
    enhanced_image = maximum_contrast_enhancement(x0_np)
    enhanced_images_tensor = []
    for i in range(enhanced_image.shape[0]):
        enhanced_img = enhanced_image[i].transpose(0,1,2)
        enhanced_img = torch.from_numpy(enhanced_img).to(x0.device)
        enhanced_img = torch.clamp(enhanced_img, 0, 255).byte()
        enhanced_images_tensor.append(enhanced_img)

    enhanced_images_tensor = torch.stack(enhanced_images_tensor, dim=0)
    print(enhanced_images_tensor.shape)
    plt.subplot(1, 3, 1)
    input_img = x0[0, :3].cpu().numpy().transpose(1, 2, 0)
    input_img = torch.from_numpy(input_img)  # 将numpy.ndarray转换为Tensor
    input_img = torch.clamp(input_img, 0, 1)  # 将图像数据限制在[0, 1]范围内
    plt.imshow(input_img)
    plt.title('Input Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    output_img = output[0].cpu().numpy()
    output_img = output_img.transpose(1, 2, 0)  # 重新排列维度
    output_img = torch.from_numpy(output_img)  # 将numpy.ndarray转换为Tensor
    output_img = torch.clamp(output_img, 0, 1)  # 将图像数据限制在[0, 1]范围内
    plt.imshow(output_img, cmap='gray')
    plt.title('Dark Channel')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    enhanced_img = enhanced_image[0].transpose(1,2,0)  # 重新排列维度
    enhanced_img = torch.from_numpy(enhanced_img)  # 将numpy.ndarray转换为Tensor
    enhanced_img = torch.clamp(enhanced_img, 0, 255).byte()  # 将图像数据限制在[0, 255]范围内，并转换为字节类型（uint8）
    plt.imshow(enhanced_img, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')
    plt.show()

    input_model = torch.cat([x0[:, :3, :, :], x, output,enhanced_images_tensor], dim=1)
    print("input_model", input_model.shape)
    output = model(input_model, t.float())  # 计算输出图像
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)