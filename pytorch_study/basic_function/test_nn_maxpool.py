import torch
from torch import nn
from torch.nn import MaxPool2d 
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

#池化操作通过减少特征图的维度和参数数量，降低了模型的复杂度，从而在一定程度上防止了过拟合现象的发生
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])

input=torch.reshape(input,(1,1,5,5))


class My_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool1(input)
        return output

model=My_model()
output=model(input)

#终端打开端口 tensorboard  --logdir="./pytorch_study/logs/test_nn_maxpool" --port=0505
writer=SummaryWriter("./pytorch_study/logs/test_nn_maxpool")
step=0
for data in dataloader:
    imgs,targets=data
    output=model(imgs)
    writer.add_images("input",imgs,step)
    writer.add_images("output",output,step)
    step=step+1