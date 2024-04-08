import torch

a=torch.ones([3,4])
b=torch.ones(4)
c=torch.sum(a,dim=0)
print(c)