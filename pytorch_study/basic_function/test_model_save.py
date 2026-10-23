import os
import torchvision
import torch
os.environ['TORCH_HOME'] = 'D:/VsCodeProjects/pytorch_study/models'#把下载地址切换到当前目录

vgg16_true=torchvision.models.vgg16(weights="DEFAULT")
#保存方式1 模型结构+模型参数
torch.save(vgg16_true,"D:/VsCodeProjects/pytorch_study/models/vgg16_method_trained1.pth")

#保存方式2 模型参数
torch.save(vgg16_true.state_dict(),"D:/VsCodeProjects/pytorch_study/models/vgg16_method_trained2.pth")
