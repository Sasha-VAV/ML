import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dvc4_model import ViT
import wandb
from dvc4_test import test_model
from dvc4_data import load_data


def train_model(
    vit: ViT,
    device: torch.device,
    train_data_loader: DataLoader,
    path_to_nn_params: str,
    epochs: int = 10,
    test_data_loader: DataLoader | None = None,
    is_use_wandb: bool = False,
    refresh_train_data: bool = False,
    path_to_train_data: str | None = None,
    batch_size: int = 4,
    save_n_times_per_epoch: int = 5,
    max_number_of_samples: int = 24000,
):
    """
    Function to train the CNN model
    :param vit: object of class CNN that represents ViT
    :param device: torch device, can be either cpu or cuda
    :param train_data_loader: object of DataLoader that represents the training data
    :param test_data_loader: object of DataLoader that represents the testing data
    :param path_to_nn_params: path to parameters of the NN
    :param epochs: number of epochs to dvc4_train the CNN model
    :param is_use_wandb: whether to use wandb instead or not
    :param refresh_train_data: whether to refresh the training data or not
    :param path_to_train_data: path to training data
    :param batch_size: batch size for training
    :param save_n_times_per_epoch: number of times to save the training data
    :param max_number_of_samples: maximum number of samples to load
    :return: nothing
    """
    if train_data_loader is None:
        return
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vit.parameters(), lr=0.001)
    try:
        vit.load_state_dict(torch.load(path_to_nn_params, weights_only=True))
    except FileNotFoundError:
        pass

    save_every_n_times_per_epoch = (
        max_number_of_samples // batch_size // save_n_times_per_epoch
    )

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; dvc4_data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = vit(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            if (
                i % save_every_n_times_per_epoch == save_every_n_times_per_epoch - 1
            ):  # print every n mini-batches
                print(
                    f"[{epoch + 1}, {(i + 1) * batch_size:5d}] "
                    f"loss: {running_loss / save_every_n_times_per_epoch / batch_size:.3f}"
                )
                running_loss = 0.0
                torch.save(vit.state_dict(), path_to_nn_params)

            if i * batch_size >= max_number_of_samples:
                break
        print(f"Total loss {total_loss / max_number_of_samples:.3f}")
        if test_data_loader is not None:
            accuracy = test_model(vit, device, test_data_loader, path_to_nn_params)
            if is_use_wandb:
                wandb.log(
                    {"loss": total_loss / max_number_of_samples, "accuracy": accuracy}
                )
        else:
            if is_use_wandb:
                wandb.log({"loss": total_loss / max_number_of_samples})
        if refresh_train_data:
            train_data_loader, _ = load_data(path_to_train_data=path_to_train_data)

    print("Finished Training")
