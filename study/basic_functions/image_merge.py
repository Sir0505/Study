import cv2
import matplotlib
import numpy

img_lena=cv2.imread("study/source/lena512.bmp")
img_sci=cv2.imread("study/source/scientist.jpg")

# print(img_lena.shape)
# print(img_sci.shape)

img_sci2=cv2.resize(img_sci,(512,512))#对cv2函数自动对图像进行重采样，缩放
# print(img_sci2.shape)

img_merged=cv2.addWeighted(img_lena,0.6,img_sci2,0.4,0)#线性加权得到

cv2.imwrite("study/output/merge_img.bmp",img_merged)

cv2.imshow("image",img_merged)#“image”弹出窗口的命名
cv2.waitKey(0)#等待时间，ms，0表示任意键终止
cv2.destroyAllWindows()