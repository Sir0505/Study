import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)#shuffle是否随机抽取

class My_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)#in_channels=3,out_channels=6时会生成两个卷积核

    def forward(self,x):
        x=self.conv1(x)
        return x

model=My_model()

#终端打开端口 tensorboard  --logdir="./pytorch_study/logs/test_nn_conv2" --port=0505
writer=SummaryWriter("./pytorch_study/logs/test_nn_conv2")
step=0
for data in dataloader:
    imgs,targets=data
    output=model(imgs)
    writer.add_images("input",imgs,step)
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step=step+1
