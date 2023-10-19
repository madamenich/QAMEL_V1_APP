import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset


def get_mnist_dataset(batch_size=100, labels=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data/',train=True,transform=transform,download=True)
    test_dataset = torchvision.datasets.MNIST(root='data/',train=False,transform=transform,download=True)
    if labels is not None:
        # Filter the dataset for the specified labels
        train_indices = [i for i in range(len(train_dataset)) if train_dataset[i][1] in labels]
        test_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1] in labels]

        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader, test_loader



def get_cifar10_dataset(batch_size=4,labels=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
