import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)#shuffle是否随机抽取

img,target=test_data[0]
print(img.shape)
print(target)

#终端打开端口 tensorboard  --logdir="./pytorch_study/logs/test_dataloader" --port=0505
writer=SummaryWriter("./pytorch_study/logs/test_dataloader")

for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,targets=data
        # print(imgs.shape)#torch.Size([4, 3, 32, 32]) 第一个4是batch_size的数值，原来的torch.Size([3, 32, 32])是也就是说会将数据融合在一起
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)#这里是images
        step=step+1

writer.close()