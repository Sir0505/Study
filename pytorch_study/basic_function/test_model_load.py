import torch
import torchvision

#对应test_model_save中的方式1,加载模型
model=torch.load("D:/VsCodeProjects/pytorch_study/models/vgg16_method_trained1.pth",weights_only=False)

#对应test_model_save中的方式2,加载模型
vgg16=torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("D:/VsCodeProjects/pytorch_study/models/vgg16_method_trained2.pth"))

print(vgg16)