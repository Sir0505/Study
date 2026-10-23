import torchvision
import os
from torch.utils.data import DataLoader
import torchvision
from torch import nn
os.environ['TORCH_HOME'] = 'D:/VsCodeProjects/pytorch_study/models'#把下载地址切换到当前目录

dataset=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=True,transform=torchvision.transforms.ToTensor(),download=True)

vgg16_true=torchvision.models.vgg16(weights="DEFAULT")

# vgg16_true.add_module("add_linear",nn.Linear(1000,10))在模型最末尾加一层神经网络
vgg16_true.classifier[6]=nn.Linear(4096,10)#替换classifier最后一层 
print(vgg16_true)
