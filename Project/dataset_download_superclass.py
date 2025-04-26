from torchvision.datasets import CIFAR100
import os
import pickle

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
