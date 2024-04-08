import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader


#准备数据
train_data=torchvision.datasets.CIFAR10("dataCI",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10("dataCI",train=False,transform=torchvision.transforms.ToTensor(),download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))




#准备dataloader
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

#创建模型
class mymodule(nn.Module):
    def __init__(self):
        super(mymodule, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


#实例化模型，损失函数，优化器
mymodel=mymodule()



local_weights = mymodel.state_dict()

print(type(local_weights))
print(local_weights.keys())