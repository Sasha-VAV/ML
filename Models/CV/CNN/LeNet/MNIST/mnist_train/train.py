import torch
import torch.nn as nn
import torch.optim as optim

from mnist_data import train_data_loader
from mnist_model import cnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
cnn.to(device=device)
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

cnn.load_state_dict(torch.load("cnn.pth", weights_only=True))


def main():
    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; mnist_data is a list of [inputs, labels]
            inputs, labels = data
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
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
                torch.save(cnn.state_dict(), "cnn.pth")

    print("Finished Training")


if __name__ == "__main__":
    main()
