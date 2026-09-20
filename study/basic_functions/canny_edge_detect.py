# 1.使用高斯滤波器，平滑图像，滤除噪声
# 2.计算每个像素点的梯度强度与方向（sobel operator）
# 3.应用非极大值抑制，消除边缘检测带来的杂散响应
# 4.应用双阈值检测来确定真实的和潜在的边缘(梯度值大于maxval则处理为边界，介于maxval和minval若连有边界则保留，小于minval则舍弃)
# 5.通过抑制鼓励的弱边缘最终完成边缘检测
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("study/source/lena512.bmp")

img1=cv2.Canny(img,80,150)
img2=cv2.Canny(img,50,100)

res=np.hstack((img1,img2))
cv2.imwrite("study/output/lena_canny_edge_detect.bmp",res)


cv2.imshow("image",res)#“image”弹出窗口的命名
cv2.waitKey(0)#等待时间，ms，0表示任意键终止
cv2.destroyAllWindows()



