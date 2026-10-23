from torch import nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output=input+1
        return output

my_model=Model()
x=torch.tensor(1.0)
output=my_model(x)
print(output)