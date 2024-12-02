import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dvc4_model import Transformer
import wandb
from dvc4_test import test_model
from dvc4_data import load_data


def train_model(
    cnn: Transformer,
    device: torch.device,
    train_data_loader: DataLoader,
    path_to_cnn_params: str,
    epochs: int = 10,
    test_data_loader: DataLoader | None = None,
    is_use_wandb: bool = False,
    refresh_train_data: bool = False,
    path_to_train_data: str | None = None,
    batch_size: int = 4,
):
    """
    Function to train the CNN model
    :param cnn: object of class CNN that represents the CNN model
    :param device: torch device, can be either cpu or cuda
    :param train_data_loader: object of DataLoader that represents the training data
    :param test_data_loader: object of DataLoader that represents the testing data
    :param path_to_cnn_params: path to parameters of the CNN model
    :param epochs: number of epochs to dvc4_train the CNN model
    :param is_use_wandb: whether to use wandb instead or not
    :param refresh_train_data: whether to refresh the training data or not
    :param path_to_train_data: path to training data
    :param batch_size: batch size for training
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
        total_loss = 0.0
        print_every_n = 24000 // batch_size
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; dvc4_data is a list of [inputs, labels]
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
            total_loss += loss.item()
        if True:  # print every n mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 24000:.3f}")
            torch.save(cnn.state_dict(), path_to_cnn_params)
        if test_data_loader is not None:
            accuracy = test_model(cnn, device, test_data_loader, path_to_cnn_params)
            if is_use_wandb:
                wandb.log({"loss": total_loss / print_every_n, "accuracy": accuracy})
        else:
            if is_use_wandb:
                wandb.log({"loss": total_loss / print_every_n})
        if refresh_train_data:
            train_data_loader, _ = load_data(path_to_train_data=path_to_train_data)

    print("Finished Training")
