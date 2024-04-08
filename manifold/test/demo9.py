import torch

a=torch.ones((10,15))
print(torch.mean(a,dim=1))