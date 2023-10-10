# Necessary imports
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss, MSELoss
from torch.optim import LBFGS

from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# Set seed for random generators
algorithm_globals.random_seed = 42

# Additional torch-related imports
import torch
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
)
import torch.nn.functional as F
from utils.qdatasets import load_dataset, preprocess

"""
Classical Neural Network
"""


class Net(Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 2)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc3 = Linear(1, 1)  # 1-dimensional output from QNN

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        return cat((x, 1 - x), -1)


def create_qnn():
    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    return qnn


def load_and_preprocess_data(dataset='MNIST', classes=[0, 1], batch_size=1, n_samples=100):
    # Use pre-defined torchvision function to load MNIST train data
    X_train = load_dataset(dataset)

    idx = np.append(
        np.where(X_train.targets == classes[0])[0][:n_samples],
        np.where(X_train.targets == classes[1])[0][:n_samples]
    )
    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]
    # Define torch dataloader with filtered data
    data_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    return data_loader


def plot_loss_convergence(loss_list):
    # Plot the loss convergence
    plt.plot(loss_list)
    plt.title("Hybrid NN Training Convergence")
    plt.xlabel("Training Iterations")
    plt.ylabel("Neg. Log Likelihood Loss")
    plt.savefig("plots/loss_convergence.png")


def train_model(model, train_loader, optimizer, loss_func, epochs):
    # Train the model
    model.train()  # Set model to training mode
    loss_list = []  # Store loss history

    for epoch in range(epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            output = model(data)  # Forward pass
            loss = loss_func(output, target)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights
            total_loss.append(loss.item())  # Store loss
        loss_list.append(sum(total_loss) / len(total_loss))
        print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))

    return loss_list


def evaluate_model(model, test_loader, loss_func):
    # Evaluate the model
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = loss_func(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    num_samples = len(test_loader.dataset)
    accuracy = 100.0 * correct / num_samples
    average_loss = total_loss / len(test_loader)

    print("Performance on test data:")
    print("\tLoss: {:.4f}".format(average_loss))
    print("\tAccuracy: {:.2f}%".format(accuracy))

    return average_loss, accuracy


def plot_predict_label(model, test_loader, n_samples_show=6, lr=0.001):
    # Plot predicted labels

    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

    model.eval()
    with no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if count == n_samples_show:
                break
            output = model(data[0:1])
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)

            pred = output.argmax(dim=1, keepdim=True)

            axes[count].imshow(data[0].numpy().squeeze(), cmap="gray")

            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title("Predicted {}".format(pred.item()))

            count += 1
    plt.tight_layout()
    plt.savefig("plots/predict_label.png")


def main(dataset='MNIST', n_samples=75, batch_size=1, classes=[0, 1], epochs=10, lr=0.001):

    train_loader, test_loader = preprocess(dataset_type='MNIST', classes=classes, n_samples=n_samples,
                                           batch_size=batch_size,
                                           fraction=0.7)

    model = Net(create_qnn())

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = NLLLoss()
    loss_list = train_model(model, train_loader, optimizer, loss_func, epochs=epochs)
    average_loss, accuracy = evaluate_model(model, test_loader, loss_func)
    plot_loss_convergence(loss_list)

def parse_labels(labels_str):
    # Split the input string on commas and convert to integers
    return [int(label) for label in labels_str.split(',')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default='MNIST')
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=0.7)
    parser.add_argument("--classes", type=str, default='0,1')

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    #Check if --labels argument is provided
    if args.classes:
        classes = parse_labels(args.classes)
        print('Parsed labels:', classes)
    manual_seed(args.seed)
    main(dataset=args.dataset, n_samples=args.n_samples, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    # caltech101 = datasets.Caltech101(root="./data", download=True)
    # print(caltech101.data.class_to_idx)
