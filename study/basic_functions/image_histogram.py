import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("study/source/lena512.bmp",0)


# img=cv2.imread("study/source/lena512color.tiff")
# color=('b','g','r')

# for i ,col in enumerate(color):
#     histr=cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color=col)
#     plt.xlim([0,256])

equ=cv2.equalizeHist(img)
# 还可以用cv2.createCLAHE方法做自适应直方图均衡化
res=np.hstack((img,equ))

cv2.imwrite("study/output/lena_histogram_balance.bmp",equ)

cv2.imshow("image",res)#“image”弹出窗口的命名
cv2.waitKey(0)#等待时间，ms，0表示任意键终止
cv2.destroyAllWindows()

