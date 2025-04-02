import os
import sys
import torch
import torch.utils
import torch.utils.data
import torchvision

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))


def get_dataloader(name, root, transform, batch_size=64, split="train", download=False):
    assert name in ["CelebA", "MNIST", "CIFAR10"], f"{name}:Unknown dataset name!"
    datasets = None
    if name=="CelebA":
        datasets = torchvision.datasets.CelebA(root, transform=transform, split=split, download=download)
    elif name=="MNIST":
        datasets = torchvision.datasets.MNIST(root, train=(split=="train"), transform=transform, download=download)
    elif name=="CIFAR10":
        datasets = torchvision.datasets.CIFAR10(root, train=(split=="train"), transform=transform, download=download)
    else:
        pass
    assert datasets is not None, "Empty Datasets!"
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader