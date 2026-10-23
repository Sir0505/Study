import torch
from torch import nn
import torchvision
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)#shuffle是否随机抽取

class My_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=Linear(196608,10)

    def forward(self,input):
        output=self.linear1(input)
        return output

model=My_model()

#终端打开端口 tensorboard  --logdir="./pytorch_study/logs/test_nn_linear" --port=0505
writer=SummaryWriter("./pytorch_study/logs/test_nn_linear")
step=0
for data in dataloader:
    imgs,targets=data
    output=torch.flatten(imgs)
    output=model(output)
    step=step+1
