import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

def get_raw_data():
    torch.manual_seed(42)

    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )

    return train_data,test_data

def get_loader_data(train_data,test_data):
    torch.manual_seed(42)
    print(f"Total Train Data : {len(train_data)}")
    print(f"Total Test Data : {len(test_data)}")

    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False
    )

    return train_loader,test_loader