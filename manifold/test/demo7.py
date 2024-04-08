import torch
import numpy as np

a=torch.tensor(0.1)
b=torch.tensor(0.2)
c=torch.tensor(0.3)

CSI=[]
CSI.append(a)
CSI.append(b)
CSI.append(c)
# CSI=torch.tensor(CSI)
# CSI=CSI.tolist()
# print(CSI)
CSI=np.array(CSI)

n = 2
top_indices = np.argsort(CSI)[-n:][::-1]

print("Top", n, "indices:", top_indices)
