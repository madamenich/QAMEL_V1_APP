import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def get_mnist_dataset(batch_size=100):
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data/',train=True,transform=transform,download=True)
    test_dataset = torchvision.datasets.MNIST(root='data/',train=False,transform=transform,download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader, test_loader

