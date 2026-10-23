import torchvision
from torch.utils.data import DataLoader
from model import *
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("./pytorch_study/logs/First_train")

train_data=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#length长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

#利用Dataloader 加载数据集
train_dataloader=DataLoader(dataset=train_data,batch_size=64)
test_dataloader=DataLoader(dataset=test_data,batch_size=64)

#创建网络模型
model=My_model()

#损失函数
loss_fn=nn.CrossEntropyLoss()

#优化器
learning_rate=1e-2
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#设置训练网络参数
total_train_step=0
total_test_step=0
#训练轮数
epoch=10

for i in range(epoch):
    print("-----------第{}轮训练开始-----------".format(i+1))
    #训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs,targets=data
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
            outputs=model(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_train_step)
    total_test_step=total_test_step+1

    # torch.save(model,"./pytorch_study/models/First_train_{}.pth".format(i))
    # print("模型保存完成")

#可以拆分终端一边跑一边看loss tensorboard  --logdir="./pytorch_study/logs/First_train" --port=0505 
writer.close()
