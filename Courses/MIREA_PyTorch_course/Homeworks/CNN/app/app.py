"""
Here you can test my CNN with your images
"""

import torch
from PIL import Image
from torchvision.transforms import transforms

from model import cnn
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
cnn.to(device=device)
cnn.load_state_dict(torch.load("cnn.pth", weights_only=True))
print("Type <exit\\> to perform exit")
s = input("Enter the image path: ")


def load_image(path):
    transform = transforms.Compose(
        [
            transforms.Resize(34),  # Optional: Resize the image
            transforms.CenterCrop(32),  # Optional: Center crop the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the image
    img = Image.open(path)

    # Apply the transform to the image
    img_tensor = transform(img)

    # Add a batch dimension (since the model expects a batch of images)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


s = "TBM930.jpg"
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
while s != "exit":
    img_tensor = load_image(s)
    img_tensor = img_tensor.to(device=device)
    output = cnn(img_tensor)
    _, predicted = torch.max(output, 1)

    print("Predicted: ", " ".join(f"{classes[predicted]:5s}"))
    print(f"Tensor: {output}")
    s = input("Enter the image path: ")
