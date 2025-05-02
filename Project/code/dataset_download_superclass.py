"""
Used for downloading CIFAR100 dataset with fine/coarse labels, splitting training and validation set,
and returning dataloaders in train.py. 
"""
from torchvision.datasets import CIFAR100
import os
import pickle
from torch.utils.data import random_split, DataLoader

class CIFAR100Custom(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, label_type='fine'):
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

        filename = self.train_list[0][0] if train else self.test_list[0][0]
        filepath = os.path.join(self.root, self.base_folder, filename)

        with open(filepath, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')

        self.fine_labels_all = entry['fine_labels']
        self.coarse_labels_all = entry['coarse_labels']

        if label_type == 'fine':
            self.targets = self.fine_labels_all
        elif label_type == 'coarse':
            self.targets = self.coarse_labels_all
        else:
            raise ValueError("label_type must be 'fine' or 'coarse'")

    def split_data(full_dataset):
        """
        Function to split CIFAR dataset into training and validation set.

        Args:
            full_dataset (CIFAR100Custom): full CIFAR100 dataset with fine and coarse labels

        Returns:
            train_dataset (CIFAR100Custom):
        """
        # get sizes for training set and validation set
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        # split dataset between training and validation
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        return train_dataset, val_dataset

    def get_data_loader(train_dataset, batch_size):
        """
        Function used in split_data() for obtaining DataLoader for training.

        Args:
            train_dataset (CIFAR100Custom): training dataset
            batch_size (int): batch size of training data loader
        
        Returns:
            DataLoader (DataLoader)
        """
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def get_validation_data_loader(val_dataset, batch_size):
        """
        Function used in split_data() for obtaining DataLoader for validation.

        Args:
            val_dataset (CIFAR100Custom): validation dataset
            batch_size (int): batch size of validation data loader

        Returns:
            DataLoader (DataLoader)
        """
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
