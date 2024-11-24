import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def load_data(
    path_to_train_data: str | None = None,
    path_to_test_data: str | None = None,
    train_batch_size: int = 4,
    test_batch_size: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Function to load the training and testing dvc_data
    :param path_to_train_data: string path to train dvc_data
    :param path_to_test_data: string path to dvc_test dvc_data
    :param train_batch_size: number of samples per train batch
    :param test_batch_size: number of samples per dvc_test batch
    :return: tuple[DataLoader, DataLoader]
    """
    data_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    try:
        train_data = ImageFolder(root=path_to_train_data, transform=data_transform)
        train_data_loader = DataLoader(
            train_data, batch_size=train_batch_size, shuffle=True, num_workers=0
        )
    except FileNotFoundError:
        train_data_loader = None
    try:
        test_data = ImageFolder(root=path_to_test_data, transform=data_transform)
        test_data_loader = DataLoader(
            test_data,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
    except FileNotFoundError:
        test_data_loader = None

    return train_data_loader, test_data_loader
