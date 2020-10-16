from torchvision import datasets, transforms
from torch.utils import data
from torch import load


def get_dataset(batch_size, dataset_type="mnist"):
    if dataset_type == "mnist":
        train_dl = data.DataLoader(
            data.Subset(
                datasets.MNIST(root="./data/", train=True, download=False, transform=transforms.ToTensor()),
                load("./data/mnist_train_indices.pt")
            ),
            batch_size=batch_size,
            shuffle=True
        )
        val_dl = data.DataLoader(
            data.Subset(
                datasets.MNIST(root="./data/", train=True, download=False, transform=transforms.ToTensor()),
                load("./data/mnist_val_indices.pt")
            ),
            batch_size=batch_size,
            shuffle=False
        )

        test_dl = data.DataLoader(
            datasets.MNIST(root="./data/", train=False, transform=transforms.ToTensor(), download=False),
            batch_size=batch_size,
            shuffle=False
        )
    elif dataset_type == 'cifar10':
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                # transforms.RandomErasing()
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]
        )
        train_dl = data.DataLoader(
            data.Subset(
                datasets.CIFAR10(root="./data/", train=True, download=False, transform=train_transform),
                load("./data/cifar10_train_indices.pt")
            ),
            batch_size=batch_size,
            shuffle=True
        )

        val_dl = data.DataLoader(
            data.Subset(
                datasets.CIFAR10(root="./data/", train=True, download=False, transform=test_transform),
                load("./data/cifar10_val_indices.pt")
            ),

            batch_size=batch_size,
            shuffle=False
        )

        test_dl = data.DataLoader(
            datasets.CIFAR10(root="./data/", train=False, download=False, transform=test_transform),
            batch_size=batch_size,
            shuffle=False
        )
    else:
        raise ValueError(f"unknown dataset {dataset_type}")

    return train_dl, val_dl, test_dl
