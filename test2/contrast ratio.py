import numpy as np
import cv2

def contrast_enhancement(image, alpha=1.2, beta=10):
    enhanced_image = np.zeros(image.shape, dtype=np.float32)
    for c in range(image.shape[2]):
        enhanced_image[:, :, c] = np.clip(alpha * image[:, :, c] + beta, 0, 255)
    return enhanced_image.astype(np.uint8)

image_path = '../datasets/train/input/5_Hazy.jpg'
image = cv2.imread(image_path)

# 对图片进行对比度增强
enhanced_image = contrast_enhancement(image)

cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()