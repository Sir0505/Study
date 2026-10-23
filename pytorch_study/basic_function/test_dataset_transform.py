import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="./pytorch_study/source/CIFAR10",train=False,transform=dataset_transform,download=True)

# print(test_set[0])
# print(test_set.classes)

# img,target=test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])

# #终端打开端口 tensorboard  --logdir="./pytorch_study/logs/test_dataset_transform" --port=0505
writer=SummaryWriter("./pytorch_study/logs/test_dataset_transform")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("test_set",img,i)

writer.close()