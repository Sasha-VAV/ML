from torch import nn
from data import train_data_loader, test_data_loader
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    """
    CNN model that can be used to determine if it is a dog or a cat picture.
    It is based on LeNet model.
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(3, 6, 5, device=self.device)
        self.pool1 = nn.AvgPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 5, device=self.device)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, device=self.device)
        self.fc2 = nn.Linear(120, 84, device=self.device)
        self.fc3 = nn.Linear(84, 10, device=self.device)
        self.fc4 = nn.Linear(10, 2, device=self.device)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


cnn = CNN()
"""images, labels = next(iter(train_data_loader))
ans = cnn(images[0])
print(ans)
a = 3"""
