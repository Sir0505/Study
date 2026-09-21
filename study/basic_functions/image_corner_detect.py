#https://blog.csdn.net/R_Feynman_/article/details/157584387

import cv2
import numpy as np
import matplotlib.pyplot as plt



# 获取统一测试图像
test_img = img=cv2.imread("study/source/chessboard.png")
# 转为灰度图（两种算法均需要灰度输入）
test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

def shi_tomasi_corner_detect(img, gray, max_corners=150, quality_level=0.01, min_distance=10):
    # 调用 Shi-Tomasi 角点检测（OpenCV 官方接口：goodFeaturesToTrack）
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,       # 最大检测角点数量
        qualityLevel=quality_level,   # 角点质量阈值（0~1，仅保留高于该值的角点）
        minDistance=min_distance,     # 角点之间的最小欧氏距离
        useHarrisDetector=False       # 禁用 Harris 检测，使用纯 Shi-Tomasi 算法
    )
    # 复制原始图像用于绘制角点
    img_shi_tomasi = img.copy()
    if corners is not None:
        corners = np.int64(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img_shi_tomasi, (x, y), 4, (0, 255, 0), -1)
    return img_shi_tomasi, corners

def harris_corner_detect(img, gray, block_size=2, ksize=3, k=0.04, threshold=0.01):
    # 转换为 32 位浮点型（Harris 要求输入格式）
    gray_float = np.float32(gray)
    # 调用 Harris 角点检测
    harris_dst = cv2.cornerHarris(gray_float, blockSize=block_size, ksize=ksize, k=k)
    # 膨胀结果（增强角点显示效果，非必需）
    harris_dst = cv2.dilate(harris_dst, None)
    # 复制原始图像用于绘制角点
    img_harris = img.copy()
    # 标记角点（红色：BGR 格式 (0, 0, 255)）
    img_harris[harris_dst > threshold * harris_dst.max()] = (0, 0, 255)
    return img_harris, harris_dst


test_harris, _ = harris_corner_detect(test_img, test_gray)
test_shi_tomasi, _ = shi_tomasi_corner_detect(test_img, test_gray)

plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(test_harris, cv2.COLOR_BGR2RGB))
plt.title("test1.jpg - Harris 角点检测（红色标记）")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(test_shi_tomasi, cv2.COLOR_BGR2RGB))
plt.title("test1.jpg - Shi-Tomasi 角点检测（绿色标记）")
plt.axis("off")
plt.tight_layout()
plt.show()

