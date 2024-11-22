import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dvc_model import CNN


def train_model(
    cnn: CNN,
    device: torch.device,
    train_data_loader: DataLoader,
    path_to_cnn_params: str,
    epochs: int = 10,
):
    """
    Function to train the CNN dvc_model
    :param cnn: object of class CNN that represents the CNN dvc_model
    :param device: torch device, can be either cpu or cuda
    :param train_data_loader: object of DataLoader that represents the training dvc_data
    :param path_to_cnn_params: path to parameters of the CNN dvc_model
    :param epochs: number of epochs to train the CNN dvc_model
    :return: nothing
    """
    if train_data_loader is None:
        return
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    try:
        cnn.load_state_dict(torch.load(path_to_cnn_params, weights_only=True))
    except FileNotFoundError:
        pass

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        print_every_n = 100
        num_of_samples = 200
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; dvc_data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_every_n == print_every_n - 1:  # print every 2000 mini-batches
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_every_n:.3f}"
                )
                running_loss = 0.0
                torch.save(cnn.state_dict(), path_to_cnn_params)
            if i >= num_of_samples:
                break

    print("Finished Training")
