"""
Main class to work with CNN
"""

import torch
from dvc_model import CNN
from dvc_data import load_data
from dvc_train import train_model
from dvc_test import test_model
import wandb
from PIL import Image
from torchvision.transforms import transforms
import torchvision

"""
PUT YOUR CONSTANTS HERE
"""
# Params
path_to_cnn_params = "pretrained_configs/alexnet_dvc_aug_95.6.pth"

# TRAIN
# You should replace path_to_train_data with the folder that contains dogs and cats
# Get it here https://www.kaggle.com/c/dogs-vs-cats/data
# Leave None, if you do not want to train/test
# For example, path_to_train_data = None
path_to_train_data = "D:\\Backup\\Less Important\\My programs\\Git\\Dog_vs_Cats_neural_network_2.0\\T1rain"
train_batch_size = 4  # Number of samples per dvc_train batch
epochs = 80

# WANDB
# Replace with True if you do want to use wandb
# Also check wandb_init method
is_use_wandb = False

# TEST
path_to_test_data = "D:\\Backup\\Less Important\\My programs\\Git\\Dog_vs_Cats_neural_network_2.0\\1Test"

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
            project="dogs_vs_cats",
            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.001,
                "architecture": "CNN",
                "dataset": "Dog_vs_Cats_Kaggle",
                "epochs": epochs,
            },
        )
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = CNN().to(device)
train_data_loader, test_data_loader = load_data(path_to_train_data, path_to_test_data)

if train_data_loader is not None:
    wandb_init(is_use_wandb)

train_model(
    cnn=cnn,
    device=device,
    train_data_loader=train_data_loader,
    path_to_cnn_params=path_to_cnn_params,
    epochs=epochs,
    test_data_loader=test_data_loader,
    is_use_wandb=is_use_wandb,
    refresh_train_data=True,
    path_to_train_data=path_to_train_data,
)

test_model(
    cnn=cnn,
    device=device,
    test_data_loader=test_data_loader,
    path_to_cnn_params=path_to_cnn_params,
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
cnn.load_state_dict(torch.load(path_to_cnn_params, weights_only=True))
for s in list_of_images_paths:
    img_tensor = load_image(s)
    img_tensor = img_tensor.to(device=device)
    output = cnn(img_tensor)
    _, predicted = torch.max(output, 1)
    try:
        print("Predicted: ", " ".join(f"{classes[predicted]:5s}"))
    except IndexError:
        print(f"Predicted: class with number {predicted + 1}, which is wrong, sorry")
    # print(f"Tensor: {output}")
