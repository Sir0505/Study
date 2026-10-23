import torch
from torch import nn
import torchvision
from torch.nn import ReLU
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=False)#shuffle是否随机抽取

class My_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1=ReLU()

    def forward(self,input):
        output=self.relu1(input)
        return output

model=My_model()

#终端打开端口 tensorboard  --logdir="./pytorch_study/logs/test_nn_relu" --port=0505
writer=SummaryWriter("./pytorch_study/logs/test_nn_relu")
step=0
for data in dataloader:
    imgs,targets=data
    output=model(imgs)
    writer.add_images("input",imgs,step)
    writer.add_images("output",output,step)
    step=step+1

    