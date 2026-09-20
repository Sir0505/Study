import cv2
import matplotlib
import numpy
import matplotlib.pyplot as plt

img=cv2.imread("study/source/lena512.bmp")
template=cv2.imread("study/source/lena_head.bmp")

# TM_SQDIFF:计算平方不同，计算值越小越相关
# TM_CCORR:计算相关性，计算出来的值越大越相关
# TM_CCOEFF:计算相关系数，值越大越相关
# TM_SQDIFF_NORMED:计算归一化平方不同，计算出来的值越接近0，越相关
# TM_CCORR_NORMED:计算归一化相关性，越接近1越相关
# TM_CCOEFF_NORMED:计算归一化相关系数，越接近1，越相关

h,w=template.shape[:2]
methods=["cv2.TM_SQDIFF","cv2.TM_CCORR","cv2.TM_CCOEFF","cv2.TM_SQDIFF_NORMED","cv2.TM_CCORR_NORMED","cv2.TM_CCOEFF_NORMED",]

for meth in methods:
    img2=img.copy()
    method=eval(meth)
    print(meth)

    res=cv2.matchTemplate(img,template,method)
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left=min_loc
    else:
        top_left=max_loc
    bottom_right=(top_left[0]+w,top_left[1]+h)

    cv2.rectangle(img2,top_left,bottom_right,255,2)

    plt.subplot(121),plt.imshow(res,cmap="gray")
    plt.xticks([]),plt.yticks([])#隐藏坐标轴
    plt.subplot(122),plt.imshow(img2,cmap="gray")
    plt.xticks([]),plt.yticks([])#隐藏坐标轴
    plt.suptitle(meth)
    plt.show()    

