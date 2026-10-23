import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d,Flatten,Linear,Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision


dataset=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)#shuffle是否随机抽取

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

loss=nn.CrossEntropyLoss()
model=My_model()
optim =torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(20):
    running_loss=0.0
    for data in dataloader:
        imgs,targets=data
        outputs=model(imgs)
        result_loss=loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss=running_loss+result_loss
    print(running_loss)

