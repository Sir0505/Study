import cv2
import matplotlib
import numpy

img=cv2.imread("study/source/lena512color.tiff")

# B,G,R=cv2.split(img)
# print(B.shape)

# img=cv2.merge((B,G,R))
# print(img.shape)

R_img=img.copy()
R_img[:,:,0]=0#B分量归零
R_img[:,:,1]=0#G分量归零
cv2.imshow("R_image",R_img)
cv2.imwrite("study/output/lena_R.bmp",R_img)

G_img=img.copy()
G_img[:,:,0]=0
G_img[:,:,2]=0
cv2.imshow("G_image",G_img)
cv2.imwrite("study/output/lena_G.bmp",G_img)

B_img=img.copy()
B_img[:,:,1]=0
B_img[:,:,2]=0
cv2.imshow("B_image",B_img)
cv2.imwrite("study/output/lena_B.bmp",B_img)
cv2.waitKey(0)

# cv2.imshow("image",img)
# cv2.waitKey(0)
cv2.destroyAllWindows()