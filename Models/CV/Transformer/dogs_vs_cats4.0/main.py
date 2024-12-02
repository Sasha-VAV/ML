"""
Main class to work with CNN
"""

import torch
from dvc4_model import ViT
from dvc4_data import load_data
from dvc4_train import train_model
from dvc4_test import test_model
import wandb
from PIL import Image
from torchvision.transforms import transforms
import torchvision

"""
PUT YOUR CONSTANTS HERE
"""
# Params
path_to_nn_params = "pretrained_configs/dvc.pth"

# TRAIN
# You should replace path_to_train_data with the folder that contains dogs and cats
# Get it here https://www.kaggle.com/c/dogs-vs-cats/data
# Leave None, if you do not want to train/test
# For example, path_to_train_data = None
path_to_train_data = "D:\\Backup\\Less Important\\My programs\\Git\\Dog_vs_Cats_neural_network_2.0\\Train"
train_batch_size = 50  # Number of samples per train batch
epochs = 80  # Number of learning epochs
save_n_times_per_epoch = (
    5  # How much times per one learning epoch you do want to save data and log it
)
max_number_of_samples = (
    24000  # How much training samples are going to be used for training
)
is_aug = False  # Set True, if you want to use data augmentation to improve results
is_shuffle_train_data = True  # Set True, if you want to shuffle train data
num_of_workers = 0  # Set number of processes that are going to load the data

# WANDB
# Replace with True if you do want to use wandb
# Also check wandb_init method
is_use_wandb = True

# TEST
path_to_test_data = (
    "D:\\Backup\\Less Important\\My programs\\Git\\Dog_vs_Cats_neural_network_2.0\\Test"
)

# Predict an answer
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


def wandb_init(is_init: bool = False):
    """
    WANDB
    :param is_init: set by default ot False if you do not want to use wandb
    :return:
    """
    if is_init:
        wandb.init(
            # set the wandb project where this run will be logged
            project="dogs_vs_cats4",
            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.001,
                "architecture": "VIT",
                "dataset": "Dog_vs_Cats_Kaggle",
                "epochs": epochs,
            },
        )
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vit = ViT().to(device)
train_data_loader, test_data_loader = load_data(
    path_to_train_data,
    path_to_test_data,
    train_batch_size=train_batch_size,
    test_batch_size=train_batch_size,
    is_augmentation=is_aug,
    is_shuffle_train=is_shuffle_train_data,
    num_workers=num_of_workers,
)

if train_data_loader is not None:
    wandb_init(is_use_wandb)

train_model(
    vit=vit,
    device=device,
    train_data_loader=train_data_loader,
    path_to_nn_params=path_to_nn_params,
    epochs=epochs,
    test_data_loader=test_data_loader,
    is_use_wandb=is_use_wandb,
    refresh_train_data=True,
    path_to_train_data=path_to_train_data,
    batch_size=train_batch_size,
    save_n_times_per_epoch=save_n_times_per_epoch,
    max_number_of_samples=max_number_of_samples,
)

test_model(
    nn=vit,
    device=device,
    test_data_loader=test_data_loader,
    path_to_cnn_params=path_to_nn_params,
)

if list_of_images_paths is None:
    exit(0)

print("Now let's see your photos")


def load_image(path: str):
    transform = transforms.Compose(
        [
            transforms.Resize(224),  # Optional: Resize the image
            transforms.CenterCrop(224),  # Optional: Center crop the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Load the image
    img = Image.open(path)

    # Apply the transform to the image
    img_tensor = transform(img)

    # Add a batch dimension (since the model expects a batch of images)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


classes = ("cat", "dog")
vit.load_state_dict(torch.load(path_to_nn_params, weights_only=True))
for s in list_of_images_paths:
    img_tensor = load_image(s)
    img_tensor = img_tensor.to(device=device)
    output = vit(img_tensor)
    _, predicted = torch.max(output, 1)
    try:
        print("Predicted: ", " ".join(f"{classes[predicted]:5s}"))
    except IndexError:
        print(f"Predicted: class with number {predicted + 1}, which is wrong, sorry")
    # print(f"Tensor: {output}")
