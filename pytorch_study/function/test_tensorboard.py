from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("./pytorch_study/logs")
#终端打开端口 tensorboard  --logdir="./pytorch_study/logs" --port=0505
# writer.add_image()
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()