import torch

M=[]
r=1
if r == 1:
    temp = [0 for i in range(len([1,2,3,4,5,6]))]
    M.append(torch.Tensor(temp))
else:
    pass
print(M[0][0])