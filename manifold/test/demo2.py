import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
def create_datasets(data_path, dataset_name, num_clients, num_shards, iid):
    dataset_name=dataset_name.upper()
    if dataset_name=="MNIST":
        transform=torchvision.transforms.ToTensor()
    elif dataset_name=="CIFAR10":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),  # 转换为tensor
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
            ]
        )
    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)


    # prepare raw training & test datasets
    training_dataset = torchvision.datasets.__dict__[dataset_name](
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.__dict__[dataset_name](
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    # print(training_dataset.data.shape)#(60000,28,28)
    # print((training_dataset.targets.shape))#(60000)
    # unsqueeze channel dimension for grayscale image datasets
    if training_dataset.data.ndim == 3:  # convert to NxHxW -> NxHxWx1
        training_dataset.data.unsqueeze_(3)
        print(training_dataset.data.shape)  # (60000,28,28,1)
    num_categories = np.unique(training_dataset.targets).shape[0]

    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "list" not in str(type(training_dataset.targets)):
        training_dataset.targets = training_dataset.targets.tolist()
        # print((training_dataset.targets.shape))




    if iid:
        pass

    else:
        pass


