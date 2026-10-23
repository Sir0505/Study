from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

writer=SummaryWriter("./pytorch_study/logs")
img=cv2.imread("pytorch_study/source/hymenoptera_data/train/bees_image/16838648_415acd9e3f.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_array=np.array(img)

writer.add_image("test",img_array,2,dataformats='HWC')

# #终端打开端口 tensorboard  --logdir="./pytorch_study/logs" --port=0505
# # writer.add_image()
# for i in range(100):
#     writer.add_scalar("y=2x",2*i,i)

writer.close()