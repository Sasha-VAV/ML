import torch
import torch.optim as optim
import torch.nn as nn
from model import cnn
from data import train_data_loader, test_data_loader
import numpy as np


def get_ans(x):
    return list(round(x[i]) for i in range(len(x)))


def is_good(x, y):
    for a, b in zip(x, y):
        if a != b:
            return 1
    return 0


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn.to(device)
if not torch.cuda.is_available():
    raise ValueError('Device must be "cuda"')
cnn.load_state_dict(
    torch.load(
        "cnn.pth",
        map_location=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )
)
for epoch in range(0, 1000):
    running_loss = 0.0
    i = -1
    for data in train_data_loader:
        i += 1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels_pos = data

        labels = torch.tensor([0.0, 0.0])
        labels[labels_pos] = 1.0

        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 0 and i != 0:
            torch.save(cnn.state_dict(), "cnn.pth")
            print(f"Epoch {epoch}: {i} avg loss: {running_loss / i}")
    if epoch % 1 == 0:
        max_loss = 0.0
        err = 0
        for data in test_data_loader:
            inputs, labels_pos = data
            labels = torch.tensor([0.0, 0.0])
            labels[labels_pos] = 1.0
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            max_loss += loss.item()
            ans = outputs.cpu().detach().numpy()
            ans = get_ans(ans)
            err += is_good(ans, labels.cpu().numpy())
        print(
            f"Epoch: {epoch}, accuracy: {1 - err / len(test_data_loader.dataset)}, avg loss: {max_loss / len(test_data_loader.dataset)}"
        )
        torch.save(cnn.state_dict(), "cnn.pth")

print("Finished Training")

for data in test_data_loader:
    inputs, labels_pos = data
    labels = torch.tensor([0.0, 0.0])
    labels[labels_pos] = 1.0
    outputs = cnn(inputs)
    print(outputs, labels)
