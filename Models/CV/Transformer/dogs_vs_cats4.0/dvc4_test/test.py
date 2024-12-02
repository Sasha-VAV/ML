import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dvc4_model import ViT


def test_model(
    nn: ViT,
    device: torch.device,
    test_data_loader: DataLoader,
    path_to_cnn_params: str,
) -> float:
    """
    Function to dvc4_test the dvc4_model
    :param nn: object of CNN class that will be used to dvc4_test the model
    :param device: torch device can be either cpu or cuda
    :param test_data_loader: object of DataLoader that represents the test data
    :param path_to_cnn_params: path to parameters of the CNN model
    :return: accuracy of the model in percent
    """
    if test_data_loader is None:
        return -1

    nn.load_state_dict(torch.load(path_to_cnn_params, weights_only=True))

    correct = 0
    total = 0
    for i, data in enumerate(test_data_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = nn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"{total} test samples. Accuracy: {correct * 100 / total}")
    return correct * 100 / total
