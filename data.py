from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import DataLoader

def get_mnist_data(batch_size=64):
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')