import torch

x=torch.arange(12)

print(x)#tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
print(x.shape)#torch.Size([12])
print(x.numel())#12

X=x.reshape(3,4)

print(X)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

print(torch.zeros((2,3,4)))
# tensor([[[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]],

#         [[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]]])

x=torch.tensor([1.0,2,4,8])
y=torch.tensor([2,2,2,2])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**y)
#按元素计算
# tensor([ 3.,  4.,  6., 10.])
# tensor([-1.,  0.,  2.,  6.])
# tensor([ 2.,  4.,  8., 16.])
# tensor([0.5000, 1.0000, 2.0000, 4.0000])
# tensor([ 1.,  4., 16., 64.])

x=torch.arange(12,dtype=torch.float32).reshape((3,4))
y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((x,y),dim=0))
print(torch.cat((x,y),dim=1))
# dim=0 行拼接，dim=1列拼接
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [ 2.,  1.,  4.,  3.],
#         [ 1.,  2.,  3.,  4.],
#         [ 4.,  3.,  2.,  1.]])
# tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
#         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
#         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])

print(x.sum())
# tensor(66.)

a=torch.arange(3).reshape((3,1))
b=torch.arange(2).reshape((1,2))
print(a)
print(b)
# tensor([[0],
#         [1],
#         [2]])
# tensor([[0, 1]])
print(a+b)
#利用广播机制，维度数量一样的补全缺少的维度
# tensor([[0, 1],
#         [1, 2],
#         [2, 3]])
#相当于
# tensor([[0，0],     
#         [1，1],
#         [2，2]])   
#           +
# tensor([[0，1],     
#         [0，1],
#         [0，1]]) 

x=torch.arange(12,dtype=torch.float32).reshape((3,4))
print(x)
print(x[1])
print(x[1,2])
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])
# tensor([4., 5., 6., 7.])
# tensor(6.)

a=torch.tensor([3.5])
print(a,a.item(),float(a),int(a))
# tensor([3.5000]) 3.5 3.5 3