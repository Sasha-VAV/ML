import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dvc_model import CNN


def test_model(
    cnn: CNN,
    device: torch.device,
    test_data_loader: DataLoader,
    path_to_cnn_params: str,
) -> float:
    """
    Function to dvc_test the dvc_model
    :param cnn: object of CNN class that will be used to dvc_test the model
    :param device: torch device can be either cpu or cuda
    :param test_data_loader: object of DataLoader that represents the test data
    :param path_to_cnn_params: path to parameters of the CNN model
    :return: accuracy of the model in percent
    """
    if test_data_loader is None:
        return -1

    cnn.load_state_dict(torch.load(path_to_cnn_params, weights_only=True))

    correct = 0
    total = 0
    for i, data in enumerate(test_data_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"{total} test samples. Accuracy: {correct * 100 / total}")
    return correct * 100 / total
