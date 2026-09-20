import cv2
import matplotlib
import numpy
import matplotlib.pyplot as plt

img=cv2.imread("study/source/lena512.bmp")


sobelx=cv2.Sobel(img,-1,1,0,ksize=3)
sobely=cv2.Sobel(img,-1,0,1,ksize=3)
sobelx=cv2.convertScaleAbs(sobelx)#梯度计算中容易出现负值，而负值在显示中会被当成0处理,这里取绝对值
sobely=cv2.convertScaleAbs(sobely)
sobelxy=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

scharrx=cv2.Scharr(img,-1,1,0)
scharry=cv2.Scharr(img,-1,0,1)
scharrx=cv2.convertScaleAbs(scharrx)
scharry=cv2.convertScaleAbs(scharry)
scharrxy=cv2.addWeighted(scharrx,0.5,scharry,0.5,0)

laplacian=cv2.Laplacian(img,-1)
laplacian=cv2.convertScaleAbs(laplacian)

res=numpy.hstack((sobelxy,scharrxy,laplacian))
cv2.imwrite("study/output/lena_operator_process.bmp",res)

cv2.imshow("image",res)#“image”弹出窗口的命名
cv2.waitKey(0)#等待时间，ms，0表示任意键终止
cv2.destroyAllWindows()