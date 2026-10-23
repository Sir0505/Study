from torchvision import transforms
import numpy as np
import cv2

img=cv2.imread("pytorch_study/source/hymenoptera_data/train/bees_image/16838648_415acd9e3f.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_array=np.array(img)

tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img_array)

print(tensor_img)



