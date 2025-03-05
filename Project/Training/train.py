# Import necessary modules
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
    wrtier = SummaryWriter('runs/CIFAR100' + timestamp)
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
    timestamp = str(datetime.now().timestamp())

    train_dataset = dataset_download.download_train_dataset()
    test_dataset = dataset_download.download_test_dataset()

    # Path for saving/loading model
    PATH = './models/model_' + timestamp + '.pt'

    # Make these command line arguments later
    epochs = 1 
    batch_size = 32
    learning_rate = 0.005
    train(epochs, batch_size, learning_rate, train_dataset, PATH)
