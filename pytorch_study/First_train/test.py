import torchvision
from PIL import Image
import cv2
import numpy as np
from torch import nn
from torch.nn import Conv2d, MaxPool2d,Flatten,Linear,Sequential
import torch

target_class=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
img=cv2.imread("pytorch_study/First_train/dog.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#注意训练的时候如果用了GPU，那测试的时候模型也得用GPU

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),      # ndarray -> PIL  ★关键
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
])

img=transform(img)


class My_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2)  ,
            Conv2d(32,64,5,padding=2)  ,
            MaxPool2d(2)     ,
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self,x):
        x=self.model(x)
        return x

model=torch.load("./pytorch_study/First_train/1023_1430.pth",weights_only=False)

if torch.cuda.is_available():
    model=model.to(device)

img=torch.reshape(img,(1,3,32,32))
if torch.cuda.is_available():  
    img=img.to(device)
model.eval()
with torch.no_grad():
    output=model(img)


output_index=output.argmax(1)
final_infer=target_class[output_index]
print("最终预测结构:{}".format(final_infer))
