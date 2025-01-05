from data import train_data_loader, train_data
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def display_random_image(image, label):
    plt.imshow(
        torchvision.utils.make_grid(image.unsqueeze(0), normalize=True).permute(1, 2, 0)
    )
    plt.title(f"Label: {train_data.classes[label]}")
    plt.axis("off")
    plt.show()


for i, data in enumerate(train_data_loader, 0):
    images, labels = data
    print(i)
    display_random_image(images[0], labels[0])
