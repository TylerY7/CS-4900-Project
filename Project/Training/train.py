# Import necessary modules
import argparse
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image
from torch.nn import functional as F
import torch.optim as optim
from torch import nn
from datetime import datetime
from model_cnn import Net
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Import out modules
import sys
# Append folder to path so python can find the module to import
sys.path.append('../Dataset/')
import dataset_download


def train(epochs, batch_size, lr, dataset, path):
    # split dataset
    train_dataset, val_dataset = dataset_download.split_data(dataset)

    # Create training set data loader
    train_loader = dataset_download.get_data_loader(train_dataset, 1)
    size = len(train_loader.dataset)

    # Instantiate model
    net = Net(len(get_classes(dataset)))
    writer = SummaryWriter('runs/CIFAR100' + timestamp)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            batch = i
            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(data[0])
                print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

        print(f'[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}')
        writer.add_scalar('training loss', running_loss / len(trainloader), epoch)
    print('Finished Training')
    writer.close()
    torch.save(net.state_dict(), path)

def get_classes(dataset):
    return dataset.classes

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a CNN model on CIFAR-100")

    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--model', type=str, default="cnn", choices=["cnn"], help="Model type to use")

    args = parser.parse_args()

    timestamp = str(datetime.now().timestamp())

    train_dataset = dataset_download.download_train_dataset()
    test_dataset = dataset_download.download_test_dataset()

    # Path for saving/loading model
    PATH = f'./models/{args.model}_model_{timestamp}.pt'

    train(args.epochs, args.batch_size, args.lr, train_dataset, PATH)
