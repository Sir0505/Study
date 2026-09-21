#原文链接：https://blog.csdn.net/qq_40467656/article/details/109030455

import cv2
import numpy as np

img=cv2.imread("study/source/lena512.bmp")

rows, cols = img.shape[:2]

# 平移
# 构造移动矩阵H
# 在x轴方向移动多少距离，在y轴方向移动多少距离
# H = np.float32([[1, 0, 50], [0, 1, 25]])

# print(img.shape)
# print(rows, cols)
# res = cv2.warpAffine(img, H, (2*cols, 2*rows))

# 缩放
# res = cv2.resize(img, None, fx=2, fy=2, 
#                   interpolation=cv2.INTER_CUBIC)


# 旋转
# 参数1：旋转中心，参数2：旋转角度，参数3：缩放因子
# 参数3正为逆时针，负值为正时针
# M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
# print(M)
# 第三个参数是输出图像的尺寸
# res = cv2.warpAffine(img, M, (cols, rows))
# res = cv2.warpAffine(img, M, (cols,rows), borderValue=(255,255,255))

# 图像仿射变换
# pos1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pos2 = np.float32([[10, 100], [200, 50], [100, 250]])
# M = cv2.getAffineTransform(pos1, pos2)   #需要三对点
# print(M)
# res = cv2.warpAffine(img, M, (2*cols, 2*rows))


# 图像透视变换
# 设置图像透视变换矩阵
pos1 = np.float32([[114, 82], [287, 156],
                   [8, 100], [143, 177]])
pos2 = np.float32([[0, 0], [188, 0],
                   [0, 262], [188, 262]])
M = cv2.getPerspectiveTransform(pos1, pos2)
print(M)
res = cv2.warpPerspective(img, M, (2*cols,2*rows))



cv2.imshow('origin_picture', img)
cv2.imshow('new_picture', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 参数1：旋转中心，参数2：旋转角度，参数3：缩放因子
# 参数3正为逆时针，负值为正时针
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
print(M)
# 第三个参数是输出图像的尺寸中心
# dst = cv2.warpAffine(img, M, (cols, rows))
dst = cv2.warpAffine(img, M, (cols,rows), borderValue=(255,255,255))
