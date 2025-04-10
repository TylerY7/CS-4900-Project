import os
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
# subset_sampler or subset_random_sampler incase random split doesn't work

DATASET_PATH = './data/cifar-100-python/'

def is_dataset_downloaded():
    # Check to see if dataset is downloaded already
    return os.path.exists(DATASET_PATH)

def download_train_dataset():
    #Download if not already downloaded
    if is_dataset_downloaded():
        print(f"Dataset already exists at: {DATASET_PATH}")
    else:
        print("Downloading CIFAR-100 dataset...")

    # Download full CIFAR-100 training dataset
    full_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=not is_dataset_downloaded(),
        transform=ToTensor()
    )
    return full_dataset

def download_test_dataset():
        #Download if not already downloaded
    if is_dataset_downloaded():
        print(f"Dataset already exists at: {DATASET_PATH}")
    else:
        print("Downloading CIFAR-100 dataset...")
    
    test_data = datasets.CIFAR100(
        root='./data',
        train=False,
        download=not is_dataset_downloaded(),
        transform=ToTensor()
    )
    return test_data


def split_data(full_dataset):
    # get sizes for training set and validation set
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # split dataset between training and validation
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    return train_dataset, val_dataset

def get_data_loader(train_dataset, batch_size):
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    dataset = download_train_dataset()
    train_dataset, val_dataset = split_data(dataset)

    train_dataloader = get_data_loader(train_dataset, batch_size=32)
    val_dataloader = get_data_loader(val_dataset, batch_size=32)
