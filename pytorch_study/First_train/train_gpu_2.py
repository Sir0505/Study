import torchvision
import torch
from torch.utils.data import DataLoader
# from model import *
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import Conv2d, MaxPool2d,Flatten,Linear,Sequential
import time

writer=SummaryWriter("./pytorch_study/logs/First_train")

train_data=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#length长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

#定义训练的设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#利用Dataloader 加载数据集
train_dataloader=DataLoader(dataset=train_data,batch_size=64)
test_dataloader=DataLoader(dataset=test_data,batch_size=64)

#创建网络模型
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
model=My_model()
if torch.cuda.is_available():
    model=model.to(device)
#损失函数
loss_fn=nn.CrossEntropyLoss()
if torch.cuda.is_available():   
    loss_fn=loss_fn.to(device)
#优化器
learning_rate=1e-2
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#设置训练网络参数
total_train_step=0
total_test_step=0
#训练轮数
epoch=10
start_time=time.time()
for i in range(epoch):
    print("-----------第{}轮训练开始-----------".format(i+1))
    #训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs,targets=data
        if torch.cuda.is_available():  
            imgs=imgs.to(device)
            targets=targets.to(device)
        outputs=model(imgs)
        loss=loss_fn(outputs,targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step=total_train_step+1
        if total_train_step%100==0:
            print("训练次数：{},Loss:{}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始
    model.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():#避免对梯度进行调整
        for data in test_dataloader:
            imgs,targets=data
            if torch.cuda.is_available():  
                imgs=imgs.to(device)
                targets=targets.to(device)
            outputs=model(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    end_time=time.time()
    print("训练总计用时:{}s".format(end_time-start_time))
    writer.add_scalar("test_loss",total_test_loss,total_train_step)
    total_test_step=total_test_step+1

    # torch.save(model,"./pytorch_study/models/First_train_{}.pth".format(i))
    # print("模型保存完成")
torch.save(model,"./pytorch_study/First_train/1023_1445.pth")
print("模型保存完成")
#可以拆分终端一边跑一边看loss tensorboard  --logdir="./pytorch_study/logs/First_train" --port=0505 
writer.close()
