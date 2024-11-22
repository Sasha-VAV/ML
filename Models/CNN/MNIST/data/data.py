import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

batch_size = 4

train_data = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2
)

test_data = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_data_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
)
# drop_last = true!!!
