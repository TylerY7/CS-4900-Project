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
import argparse

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

    # ensure user has cuda, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device: {device}')

    '''
    ###
    Choose which style of CL arguments we prefer
    ###

    if len(sys.argv) != 4:
        print('Script usage: python(3) train.py [epochs:int] [batch_size:int] [learning_rate:float]\n')
        print('epochs: Number of training epochs')
        print('batch_size: Batch size for training')
        print('learning rate: Learning rate for optimization\n')
        print('Example usage: python(3) train.py 10 32 (0.005/5e-3)')
        sys.exit()
    else:
        epochs = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        learning_rate = float(sys.argv[3])
    '''
    # create parser object
    parser = argparse.ArgumentParser(description='Hyperparameters for training model.')

    # option to use positional arguments (must be entered in order when running script)
    parser.add_argument('epochs', nargs='?', type=int, help='(int) Number of training epochs')
    parser.add_argument('batch_size', nargs='?', type=int, help='(int) Batch size for training')
    parser.add_argument('learning_rate', nargs='?', type=float, help='(float) Learning rate for optimization')

    # option to specify arguments (does not need to be ordered)
    parser.add_argument('--epochs', dest='epochs_flag', type=int, help='(int) Number of training epochs')
    parser.add_argument('--batch_size', dest='batch_flag', type=int, help='(int) Batch size for training')
    parser.add_argument('--learning_rate', dest='lr_flag', type=float, help='(float) Learning rate for optimization')

    # get command line arguments
    args = parser.parse_args()

    # Set variables to positional arguments if used, otherwise use flag specified arguments
    epochs = args.epochs if args.epochs is not None else args.epochs_flag
    batch_size = args.batch_size if args.batch_size is not None else args.batch_flag
    learning_rate = args.learning_rate if args.learning_rate is not None else args.lr_flag

    # print out args for debugging
    print(f'epochs={epochs}\nbatch_size={batch_size}\nlr={learning_rate}')

    if epochs is None or batch_size is None or learning_rate is None:
        print('\nError: Must provide epochs, batch_size, and learning_rate as either positional or flagged arguments')
        print('\nExample script usage:')
        print('\tPositional arguments: python(3) train.py 5 32 0.005')
        print('\tFlagged arguments: python(3) train.py --epochs 5 --batch_size 32 --learning_rate 0.005\n')
        sys.exit(1)

    # generate timestamp for filename when saving
    timestamp = str(datetime.now().timestamp())

    # Downloades datasets if not already installed
    train_dataset = dataset_download.download_train_dataset()
    test_dataset = dataset_download.download_test_dataset()

    # Path for saving/loading model
    PATH = './models/model_' + timestamp + '.pt'

    # runs train function
    train(epochs, batch_size, learning_rate, train_dataset, PATH)
