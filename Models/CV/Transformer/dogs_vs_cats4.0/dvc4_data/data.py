import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def load_data(
    path_to_train_data: str | None = None,
    path_to_test_data: str | None = None,
    train_batch_size: int = 4,
    test_batch_size: int = 4,
    is_augmentation: bool = False,
    is_shuffle_train: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Function to load the training and testing data
    :param path_to_train_data: string path to train data
    :param path_to_test_data: string path to test data
    :param train_batch_size: number of samples per train batch
    :param test_batch_size: number of samples per test batch
    :param is_augmentation: whether to use data augmentation or not
    :param is_shuffle_train: whether to shuffle train data or not
    :param num_workers: number of workers for data loading
    :return: tuple[DataLoader, DataLoader]
    """
    augmented_data_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(179),
            transforms.RandomVerticalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    default_data_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if path_to_train_data is not None:
        try:
            if is_augmentation:
                train_data = ImageFolder(
                    root=path_to_train_data, transform=augmented_data_transform
                )
                train_data_loader = DataLoader(
                    train_data,
                    batch_size=train_batch_size,
                    shuffle=is_shuffle_train,
                    num_workers=num_workers,
                )
            else:
                train_data = ImageFolder(
                    root=path_to_train_data, transform=default_data_transform
                )
                train_data_loader = DataLoader(
                    train_data,
                    batch_size=train_batch_size,
                    shuffle=is_shuffle_train,
                    num_workers=num_workers,
                )
        except FileNotFoundError:
            train_data_loader = None
    else:
        train_data_loader = None
    if path_to_test_data is not None:
        try:
            test_data = ImageFolder(
                root=path_to_test_data, transform=default_data_transform
            )
            test_data_loader = DataLoader(
                test_data,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True,
            )
        except FileNotFoundError:
            test_data_loader = None
    else:
        test_data_loader = None
    return train_data_loader, test_data_loader
