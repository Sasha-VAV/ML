# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
#
# path_to_data = (
#     "D:\\Backup\\Less Important\\My programs\\Git\\Dog_vs_Cats_neural_network_2.0\\"
# )
# batch_size = 1
# data_transform = transforms.Compose(
#     [
#         transforms.Resize(64),
#         transforms.CenterCrop(60),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )
#
# train_data = ImageFolder(root=path_to_data + "train", transform=data_transform)
# test_data = ImageFolder(root=path_to_data + "test", transform=data_transform)
#
# train_data_loader = DataLoader(
#     train_data, batch_size=batch_size, shuffle=True, num_workers=0
# )
# test_data_loader = DataLoader(
#     test_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
# )
#
# images, labels = next(iter(train_data_loader))


import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4

train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2
)

test_data = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_data_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
)
# drop_last = true!!!

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
