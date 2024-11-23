"""
Main class to work with CNN
"""
import torch
from dvc_model import CNN
from dvc_data import load_data
from train import train_model
from dvc_test import test_model
import wandb
from PIL import Image
from torchvision.transforms import transforms

"""
PUT YOUR CONSTANTS HERE
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
epochs = 200

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

# WANDB
# Comment, if you do not want to use it
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = CNN().to(device)
train_data_loader, test_data_loader = load_data(path_to_train_data, path_to_test_data)

train_model(
    cnn=cnn,
    device=device,
    train_data_loader=train_data_loader,
    path_to_cnn_params=path_to_cnn_params,
    epochs=epochs,
    test_data_loader=test_data_loader,
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
for s in list_of_images_paths:

    def load_image(path: str):
        transform = transforms.Compose(
            [
                transforms.Resize(34),  # Optional: Resize the image
                transforms.CenterCrop(32),  # Optional: Center crop the image
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
    img_tensor = load_image(s)
    img_tensor = img_tensor.to(device=device)
    output = cnn(img_tensor)
    _, predicted = torch.max(output, 1)

    print("Predicted: ", " ".join(f"{classes[predicted]:5s}"))
    print(f"Tensor: {output}")
