"""
Config file to set model architecture
"""

import torch
from dogs_vs_cats4.dvc4_model import ViT
import torchvision.models as models

# Set computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model architecture
img_size = 224
num_classes = 100
is_use_torch_vit = False
if not is_use_torch_vit:
    # Edit your own model architecture
    vit = ViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        drop_rate=0.0,
    ).to(device)
else:  # OR
    # Choose a build in model
    vit = models.vit_b_16().to(device)

# Set where network is going to store its parameters
path_to_nn_params = "pretrained_configs/dvc.pth"

# Set train and test data paths
# If you want to use python datasets, then edit dvc4_test/test.py
# This model use dogs vs cats dataset from kaggle
# Get it here https://www.kaggle.com/c/dogs-vs-cats/data
# If you want to skip training or testing, then leave this field with incorrect path or None
# For example, path_to_train_data = None, or path_to_test_data = "invalid_path"
path_to_train_data = "D:\\Backup\\Less Important\\My programs\\Git\\Dog_vs_Cats_neural_network_2.0\\small_train"
is_use_build_in_train = path_to_train_data is None
path_to_test_data = "D:\\Backup\\Less Important\\My programs\\Git\\Dog_vs_Cats_neural_network_2.0\\small_test"
is_use_build_in_test = path_to_test_data is None
num_of_workers = 0
# Set number of processes that are going to load the data, or leave 0

# Training config
train_batch_size = 10  # Number of samples per batch
num_of_train_epochs = 1000
# Leave -1, if you want to use all the samples from train dataset
max_number_of_train_samples = 100
is_refresh_train_data = False  # Set True, if you want to refresh data after every epoch
is_aug = False  # Set True, if you want to use data augmentation to improve results
is_shuffle_train_data = False  # Set True, if you want to shuffle train data

# Validation and test config
number_of_validation_samples = 10  # Leave 0, if you do not want to validate data
max_number_of_test_samples = 10
# Leave -1, you want to use all the samples from test dataset

# Logging
# Console logging
print_n_times_per_epoch = 0
# How much times you will print into console information about epoch


# This project supports wandb
# Wandb logging
is_use_wandb = False  # Set True if you want to log data in wandb
wandb_config = {
    "learning_rate": 0.001,
    "architecture": "VIT",
    "dataset": "Dog_vs_Cats_Kaggle",
    "epochs": num_of_train_epochs,
}

# Using nn for tasks
# Put paths to images, that you want to classify here
list_of_images_paths = [
    "img/cat1.jpg",
    "img/dog1.jpg",
    "img/dog2.jpg",
    "img/corgi.jpg",
    "img/corgi1.jpg",
    "img/cat2.jpg",
    "img/cat3.jpg",
    "img/samoed.jpg",
]
# Set classes in order of nn's training
classes = ("cat", "dog")

# Good job!
# You are set up
# Now you should launch __main__.py