import cv2
import matplotlib
import numpy
import matplotlib.pyplot as plt

img=cv2.imread("study/source/lena512color.tiff")

# cv2.imread 读到 img，它是 BGR
# plt.imshow，matplotlib 当成 RGB 来显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

top_size,bottom_size,left_size,right_size=(50,50,50,50)

replicate=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REPLICATE)
reflect  =cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REFLECT)
reflect101=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REFLECT_101)
wrap=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_WRAP)
constant=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_CONSTANT,value=0)#value为填充值

plt.subplot(231),plt.imshow(img,"gray"),plt.title("ORIGINAL")               
plt.subplot(232),plt.imshow(replicate,"gray"),plt.title("REPLICATE")#复制边缘像素
plt.subplot(233),plt.imshow(reflect,"gray"),plt.title("REFLECT")#fedcba|abcdefgh|hgfedcb
plt.subplot(234),plt.imshow(reflect101,"gray"),plt.title("REFLECT_101")#gfedcb|abcdefgh|gfedcba
plt.subplot(235),plt.imshow(wrap,"gray"),plt.title("WRAP")#cdefgh|abcdefgh|abcdefg
plt.subplot(236),plt.imshow(constant,"gray"),plt.title("CONSTANT")#常数值填充

# plt.savefig("study/output/border_result.png")
plt.show()