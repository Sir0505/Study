import numpy as np
import cv2

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    手动为图像添加高斯噪声
    :param image: 输入图像 (numpy array)
    :param mean: 高斯噪声的均值
    :param sigma: 高斯噪声的标准差
    :return: 添加噪声后的图像副本
    """
    # 1. 创建副本，保护原图
    noisy_image = image.copy()
    
    # 2. 获取图像属性
    rows, cols = noisy_image.shape[:2]
    # 判断是彩色图像 (3通道) 还是灰度图像
    if len(noisy_image.shape) == 3:
        channels = noisy_image.shape[2]
    else:
        channels = 1
    
    # 3. 生成与图像形状相同的高斯噪声
    # 彩色图像：对每个通道独立生成噪声，保持空间独立性
    # 灰度图像：直接生成二维噪声
    if channels > 1:
        gaussian_noise = np.random.normal(mean, sigma, (rows, cols, channels))
    else:
        gaussian_noise = np.random.normal(mean, sigma, (rows, cols))
    
    # 4. 将噪声叠加到图像上 (先转为 float 防止溢出)
    noisy_image = noisy_image.astype(np.float32) + gaussian_noise
    
    # 5. 裁剪到 [0, 255] 并转回 uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

# === 运行示例 ===
img = cv2.imread('study/source/lena512.bmp') 
result_img = add_gaussian_noise(img, mean=0, sigma=50)
# cv2.imwrite("study/output/lena_gaussian_noise.bmp", result_img)
cv2.imshow('Noisy', result_img) 
cv2.waitKey(0)