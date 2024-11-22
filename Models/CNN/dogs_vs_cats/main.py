"""
Main class to work with CNN
"""
import torch
from dvc_model import CNN
from dvc_data import load_data
from train import train_model
from dvc_test import test_model


"""
PUT YOU CONSTANTS HERE
"""
# Params
path_to_cnn_params = "cnn.pth"

# TRAIN
# You should replace path_to_train_data with the folder that contains dogs and cats
# Get it here https://www.kaggle.com/c/dogs-vs-cats/data
# Leave None, if you do not want to train/dvc_test
# For example, path_to_train_data = None
path_to_train_data = "D:\\Backup\\Less Important\\My programs\\Git\\Dog_vs_Cats_neural_network_2.0\\Train"
train_batch_size = 4  # Number of samples per train batch

# TEST
path_to_test_data = (
    "D:\\Backup\\Less Important\\My programs\\Git\\Dog_vs_Cats_neural_network_2.0\\Test"
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = CNN().to(device)
train_data_loader, test_data_loader = load_data(path_to_train_data, path_to_test_data)

train_model(
    cnn=cnn,
    device=device,
    train_data_loader=train_data_loader,
    path_to_cnn_params=path_to_cnn_params,
    epochs=2,
)

test_model(
    cnn=cnn,
    device=device,
    test_data_loader=test_data_loader,
    path_to_cnn_params=path_to_cnn_params,
)
