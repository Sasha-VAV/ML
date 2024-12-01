import torch
from mnist_model import cnn
import torch.nn as nn
from mnist_data import test_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
cnn.to(device=device)
cnn.load_state_dict(torch.load("cnn.pth", weights_only=True))


def main():
    correct = 0
    total = 0
    for i, data in enumerate(test_data_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"accuracy: {correct * 100 / total}")


if __name__ == "__main__":
    main()
