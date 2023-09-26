import numpy as np
import cv2


def maximum_contrast_enhancement(image, s=3, r=7):
    enhanced_image = np.zeros_like(image, dtype=np.float32)
    padded_image = cv2.copyMakeBorder(image, r, r, r, r, cv2.BORDER_REFLECT)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded_image[i:i + s, j:j + s]
            local_contrast_values = []
            for m in range(i, i + r):
                for n in range(j, j + r):
                    local_patch = padded_image[m:m + s, n:n + s]
                    local_contrast_values.append(np.square(local_patch - patch).mean())
            max_contrast = max(local_contrast_values)
            enhanced_image[i, j] = max_contrast

    # 归一化并将图像转换为uint8类型
    enhanced_image = (enhanced_image - np.min(enhanced_image)) / (np.max(enhanced_image) - np.min(enhanced_image))
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    return enhanced_image



image_path = '../datasets/train/input/4_Hazy.jpg'
image = cv2.imread(image_path)

enhanced_image = maximum_contrast_enhancement(image)

cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()