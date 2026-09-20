import cv2
import matplotlib
import numpy
import matplotlib.pyplot as plt

img0=cv2.imread("study/source/lena512.bmp")


imgy=cv2.Sobel(img0,-1,0,1,ksize=3)
imgy=cv2.convertScaleAbs(imgy)#梯度计算中容易出现负值，而负值在显示中会被当成0处理

imgx=cv2.Sobel(img0,-1,1,0,ksize=3)
imgx=cv2.convertScaleAbs(imgx)#梯度计算中容易出现负值，而负值在显示中会被当成0处理

imgxy=cv2.addWeighted(imgx,0.5,imgy,0.5,0)#图像求和


cv2.imwrite("study/output/lena_sobel_operator.bmp",imgxy)

cv2.imshow("image",imgxy)#“image”弹出窗口的命名
cv2.waitKey(0)#等待时间，ms，0表示任意键终止
cv2.destroyAllWindows()