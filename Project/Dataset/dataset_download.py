from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
# subset_sampler or subset_random_sampler incase random split doesnt work

def download_dataset():
    # Dowload full CIFAR-100 training dataset
    full_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=ToTensor()
    )
    return full_dataset

    '''
    test_data = datasets.CIFAR100(
        root='./data',
        traing=False,
        Download=True,
        transform=ToTensor()
    )
    '''


def split_data(full_dataset):
    # get sizes for training set and validation set
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))

    # split dataset betwee training and validation
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    return train_dataset, val_dataset
    # create dataloaders for training and validation

def get_data_loaders(train_dataset, val_dataset, batch_size):
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

if __name__ == '__main__':
    dataset = download_dataset()
    train_dataloader, val_dataloader = split_data(dataset)
