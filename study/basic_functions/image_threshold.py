import cv2
import matplotlib
import numpy
import matplotlib.pyplot as plt

img0=img=cv2.imread("study/source/lena512.bmp")

ret,img1=cv2.threshold(img0,127,255,cv2.THRESH_BINARY)#超过127部分取255
ret,img2=cv2.threshold(img0,127,255,cv2.THRESH_BINARY_INV)#上面的反转
ret,img3=cv2.threshold(img0,127,255,cv2.THRESH_TRUNC)#大于阈值127部分设为127
ret,img4=cv2.threshold(img0,127,255,cv2.THRESH_TOZERO)#大于阈值127部分不变，其他改为0
ret,img5=cv2.threshold(img0,127,255,cv2.THRESH_TOZERO_INV)#上面的反转

titles=["Original Image","Binary","Binary_INV","Trunc","Tozero","Tozero_Inv"]
images=[img0,img1,img2,img3,img4,img5]

for i in range(6):
    plt.subplot(2,3,i+1) ,plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])


plt.savefig("study/output/threshold_result.png")
plt.show()#plt.show()后会关闭画窗，save要在show之前