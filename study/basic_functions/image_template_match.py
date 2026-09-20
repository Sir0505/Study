import cv2
import matplotlib
import numpy

img=cv2.imread("study/source/lena512.bmp")
template=cv2.imread("study/source/lena_head.bmp")

res=cv2.matchTemplate(img,template,cv2.TM_SQDIFF)
# TM_SQDIFF:计算平方不同，计算值越小越相关
# TM_CCORR:计算相关性，计算出来的值越大越相关
# TM_CCOEFF:计算相关系数，值越大越相关
# TM_SQDIFF_NORMED:计算归一化平方不同，计算出来的值越接近0，越相关
# TM_CCORR_NORMED:计算归一化相关性，越接近1越相关
# TM_CCOEFF_NORMED:计算归一化相关系数，越接近1，越相关
min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)

print(min_loc,max_loc)