from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch


def load_dataset(dataset_type='MNIST'):
    root = "./data"
    """
    Load a specific dataset based on the dataset_type.

    Parameters:
        dataset_type (str): The type of dataset to load. Default is 'MNIST'.
        root (str): Root directory to store the dataset.
        transform (callable, optional): A function/transform to apply to the dataset.

    Returns:
        dataset: The loaded dataset.
    """
    if dataset_type == 'MNIST':
        return datasets.MNIST(root=root, train=True, download=True,
                              transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_type == 'Caltech101':
        return datasets.Caltech101(root=root, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    # Add more elif clauses for other dataset types as needed


def preprocess1(dataset_type='MNIST', classes=[], n_samples=100, batch_size=1):
    # Add preprocessing steps here
    if dataset_type == 'MNIST':
        # Use pre-defined torchvision function to load MNIST train data
        X_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
        )

        # Filter out labels (originally 0-9), leaving only labels 0 and 1
        idx = np.append(
            np.where(X_train.targets == 0)[0][:n_samples], np.where(X_train.targets == 1)[0][:n_samples]
        )
        X_train.data = X_train.data[idx]
        X_train.targets = X_train.targets[idx]

        # Define torch dataloader with filtered data
        data_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

    return data_loader


def preprocess(dataset_type='MNIST', classes=[], n_samples=100, batch_size=1, fraction=0.7):
    # Add preprocessing steps here
    if dataset_type == 'MNIST':
        # Use pre-defined torchvision function to load MNIST train data
        X_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
        )

        # Filter out labels based on classes and n_samples
        idx = []
        for label in classes:
            idx_label = np.where(X_train.targets == label)[0][:n_samples]
            idx.extend(idx_label)

        # Create a new dataset with filtered labels
        filtered_dataset = torch.utils.data.Subset(X_train, idx)

        # Split the dataset into train and test based on the fraction
        train_size = int(fraction * len(filtered_dataset))
        test_size = len(filtered_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(filtered_dataset, [train_size, test_size])

        # Define torch dataloaders for train and test sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
