import cv2
import numpy as np

# img=cv2.imread("study/source/lenacolor_salt_and_pepper_noise.bmp")
img=cv2.imread("study/source/lenacolor_gaussian_noise.bmp")
box=cv2.boxFilter(img,-1,(3,3),normalize=True)#方框滤波，基本为均值滤波，可选归一化
gussian=cv2.GaussianBlur(img,(5,5),1)#高斯滤波
median=cv2.medianBlur(img,5)

res = np.hstack((img,box,gussian,median))


# cv2.imwrite("study/output/lena_noise_salt_and_pepper_filter.bmp",res)
cv2.imwrite("study/output/lena_noise_gaussian_filter.bmp",res)

cv2.imshow("Various filter",res)#“image”弹出窗口的命名
cv2.waitKey(0)#等待时间，ms，0表示任意键终止
cv2.destroyAllWindows()