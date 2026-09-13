import cv2
import matplotlib
import numpy

img=cv2.imread("study/source/lena512.bmp",cv2.IMREAD_GRAYSCALE)

lena_head =img[50:390,100:420]

cv2.imshow("image",lena_head)
cv2.waitKey(0)#等待时间，ms，0表示任意键终止
cv2.destroyAllWindows()