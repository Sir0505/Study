import cv2
import matplotlib
import numpy
import os

#print("当前工作目录 cwd :", os.getcwd()) #找到工作路径

#../为返回上级目录下 ./为本级目录下
#img=cv2.imread("study/source/lena512.bmp")
img=cv2.imread("study/source/lena512.bmp",cv2.IMREAD_GRAYSCALE)#读取灰度图

#print出来 有几个中括号代表有几维的数据
#[             ← 第 1 层：axis 0（有几"块"）
#[             ← 第 2 层：axis 1（每块有几行）
# [1 2 3]      ← 第 3 层：axis 2（每行有几个数）R,G,B

print(img)
print(img.size)
print(img.dtype)
cv2.imshow("image",img)#“image”弹出窗口的命名
cv2.waitKey(0)#等待时间，ms，0表示任意键终止
cv2.destroyAllWindows()
