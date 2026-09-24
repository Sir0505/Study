import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


img=cv2.imread("study/source/lena512.bmp",0)

img_float32=np.float32(img)

#dft 时域->频域
dft=cv2.dft(img_float32,flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift=np.fft.fftshift(dft)

# 读取频谱图
# magnitude_spectrum=20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))



rows,cols =img.shape
crow,ccol=int(rows/2),int(cols/2)

#低通滤波器
mask=np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30,ccol-30:ccol+30]=1

#高通滤波器
# mask=np.ones((rows,cols,2),np.uint8)
# mask[crow-30:crow+30,ccol-30:ccol+30]=0




#idft 频域->时域
fshift=dft_shift*mask
f_ishift=np.fft.ifftshift(fshift)
img_back=cv2.idft(f_ishift)
img_back=cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
img_back = np.uint8(img_back)
cv2.imwrite("study/output/lena_low_pass.bmp",img_back)


plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title("Input Image"),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(img_back,cmap='gray')
plt.title("Result"),plt.xticks([]),plt.yticks([])
plt.show()