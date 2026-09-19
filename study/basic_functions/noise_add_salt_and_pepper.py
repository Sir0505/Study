#https://edu.51cto.com/article/note/44858.html
# 椒盐噪声（Salt-and-Pepper Noise）是一种典型的脉冲噪声，其名称来源于图像中出现的黑白杂点。理解其物理意义是正确编写代码的前提。
# “椒”代表黑点：在灰度图或RGB通道中，像素值为0，模拟信号传输中的丢失或损坏。
# “盐”代表白点：在灰度图或RGB通道中，像素值为255（8位图像最大值），模拟信号中的突发干扰。
# 均匀分布原则：为了模拟真实的随机噪声，通常将噪声点总数均分为两组，一半赋值为0，一半赋值为255。
# 稀疏性特征：噪声比例通常较低（如1%-5%），以保持图像主体信息不被过度破坏。
# 通道一致性：对于彩色图像，噪声应独立作用于每个通道或整体像素位置，需确保不破坏RGB结构
import numpy as np
import cv2

def add_salt_and_pepper_noise(image, noise_ratio=0.05):
    """
    手动为图像添加椒盐噪声
    :param image: 输入图像 (numpy array)
    :param noise_ratio: 噪声比例 (0-1)
    :return: 添加噪声后的图像副本
    """
    # 1. 创建副本，保护原图
    noisy_image = image.copy()
    
    # 2. 获取图像属性
    rows, cols = noisy_image.shape[:2]
    # 计算总像素数 (如果是彩色图，需考虑通道，这里按像素位置处理)
    total_pixels = rows * cols
    
    # 3. 计算需要添加噪声的像素点数量
    num_noise = int(total_pixels * noise_ratio)
    
    # 4. 生成随机坐标 (行索引和列索引)
    # random_rows 和 random_cols 是一维数组，长度为 num_noise
    random_rows = np.random.randint(0, rows, num_noise)
    random_cols = np.random.randint(0, cols, num_noise)
    
    # 5. 分配椒(0)和盐(255)
    # 前一半为椒(0)，后一半为盐(255)
    half_num = num_noise // 2
    
    # 处理彩色图像 (3通道) 或灰度图像 (单通道/无第三维)
    if len(noisy_image.shape) == 3:
        # 彩色图像：对每个通道同时施加相同的坐标噪声，保持空间一致性
        noisy_image[random_rows[:half_num], random_cols[:half_num], :] = 0   # 椒: Black
        noisy_image[random_rows[half_num:], random_cols[half_num:], :] = 255 # 盐: White
    else:
        # 灰度图像直接赋值
        noisy_image[random_rows[:half_num], random_cols[:half_num]] = 0     # 椒: Black
        noisy_image[random_rows[half_num:], random_cols[half_num:]] = 255   # 盐: White
        
    return noisy_image

# === 运行示例 ===
img = cv2.imread('study/source/lena512color.tiff') 
result_img = add_salt_and_pepper_noise(img, noise_ratio=0.05)
# cv2.imwrite("study/output/lenacolor_salt_and_pepper_noise.bmp",result_img)
cv2.imshow('Noisy', result_img) 
cv2.waitKey(0)
