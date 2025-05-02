# Import necessary modules
import argparse
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image
from torchvision import transforms
from torch.nn import functional as F
import torch.optim as optim
from torch import nn
from datetime import datetime
from model_cnn import Net
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
import sys
import os

# Import for the linear model for training
from linear_model import LinearModel

# Append folder to path so python can find the module to import

base_dir = os.path.dirname(os.path.abspath(__file__))
runs_dir = os.path.join(base_dir, '../runs/CIFAR100')
models_dir = os.path.join(base_dir, '../models')

from dataset_download_superclass import CIFAR100Custom

# Created a dictionary to add more models
MODEL_MAP = {
    "Net": Net,
    'LinearModel': LinearModel
}

def train(epochs, batch_size, lr, dataset, path, model_name, output_classes):
    """
    Function for training the model with given arguments. 
    Prints information on training loss and validation loss throughout along with accuracy.
    Saves models to models folder for future loading of model and saves to runs folder for Tensorboard graphing.

    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate for optimization
        dataset (CIFAR100Custom): CIFAR100 dataset
        path (string): path to save trained model
        model_name (string): type of model (Net for CNN or LinearModel for Linear)
        output_classes (int): 20 (super classes) or 100 (classes)
    """
    # split dataset
    train_dataset, val_dataset = CIFAR100Custom.split_data(dataset)

    # Create training set data loader
    train_loader = CIFAR100Custom.get_data_loader(train_dataset, batch_size)
    size = len(train_loader.dataset)

    # Create validation set data loader
    val_loader = CIFAR100Custom.get_validation_data_loader(val_dataset, batch_size)

    # (Dynamically) Instantiate model
    model_class = MODEL_MAP[model_name]
    model = model_class(output_classes)
    model.to(device)
    writer = SummaryWriter(runs_dir + str(epochs) + 'e-' + str(batch_size) + 'bs-' + str(lr) + 'lr-' + str(output_classes) + 'cls_' + timestamp)

    criterion = nn.CrossEntropyLoss()
    print(f'learning_rate={lr}')
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    # training loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train(True)

        for i, data in enumerate(train_loader):
            # Loads data into device being used for training
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if epoch == 0 and i == 0:
                img_grid = torchvision.utils.make_grid(inputs)
                writer.add_image('input_images', img_grid)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 0:
                print(f'loss: {loss.item():>7f} [{i * batch_size:>5d}/{size:>5d}]')
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * (correct / total)
        print(f'[{epoch + 1}] loss: {avg_loss:.3f} | accuracy: {accuracy:.2f}%')

        # function that performs validation
        validate(epoch, model, writer, val_loader, avg_loss)

        writer.add_scalar('Training loss', avg_loss, epoch)
        writer.add_scalar('Training accuracy', accuracy, epoch)
        # get_last_lr() returns list of len=1 containing LR used for this epoch
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        # ensure collected data is written to disc after each epoch
        writer.flush()

    print('Finished Training')
    writer.close()

    # Saves model with its label type(Fine/Course) and number of classes (Fine = 100, coarse = 20)
    torch.save({
        'model_state': model.state_dict(),
        'label_type': label_type,
        'num_classes': output_classes
        }, path)

def validate(epoch, model, writer, val_loader, avg_loss):
    """
    Measures model's training performance. 
    Validation is performed each epoch during training method and uses same arguments provided for training method.
    Prints average training loss compared to average validation loss. Records and graphs training vs validation loss.

    Args:
        epochs (int): Number of training epochs
        model (string): Net for CNN or LinearModel for linear model
        writer (SummaryWriter): For saving data in a tensor
        val_loader (DataLoader): Loads datasets for CIFAR100
        avg_loss (float): Average loss during training
    """
    # sets model to evalution mode
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_v_loss = 0.0

    # performs validation
    with torch.no_grad():
        for i, v_data in enumerate(val_loader):
            v_inputs, v_labels = v_data
            v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
            v_outputs = model(v_inputs)
            v_loss = criterion(v_outputs, v_labels)
            running_v_loss += v_loss.item()
    
    avg_v_loss = running_v_loss / len(val_loader)
        
    # displays/compares with average training loss
    print(f"Avg training loss: {avg_loss:>5f}   vs   Avg val loss: {avg_v_loss:>5f}")

    # records and graphs training vs validation loss
    writer.add_scalars("Training vs. validation loss", {"Training": avg_loss, "Validation": avg_v_loss}, epoch)


if __name__ == '__main__':

    # ensure user has cuda, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device: {device}')

    # create parser object
    parser = argparse.ArgumentParser(description='Hyperparameters for training model.')

    # option to use positional arguments (must be entered in order when running script)
    parser.add_argument('epochs', nargs='?', type=int, help='(int) Number of training epochs')
    parser.add_argument('batch_size', nargs='?', type=int, help='(int) Batch size for training')
    parser.add_argument('learning_rate', nargs='?', type=float, help='(float) Learning rate for optimization')
    parser.add_argument('model', nargs='?', type=str, choices=MODEL_MAP.keys(), help='(str) type of model (Net for CNN or LinearModel for Linear)')
    parser.add_argument('output_classes', nargs='?', type=int, choices=[20, 100], help='(int) 20 (super classes) or 100 (classes)')

    # option to specify arguments (does not need to be ordered)
    parser.add_argument('--epochs', dest='epochs_flag', type=int, help='(int) Number of training epochs')
    parser.add_argument('--batch_size', dest='batch_flag', type=int, help='(int) Batch size for training')
    parser.add_argument('--learning_rate', dest='lr_flag', type=float, help='(float) Learning rate for optimization')

    # New flags for model and output classes
    parser.add_argument('--model', dest='model_flag', type=str, default='Net', choices=MODEL_MAP.keys(), help='(str) Model name (Net or LinearModel, default: Net)')
    parser.add_argument('--output_classes', dest='output_classes_flag', type=int, choices=[20,100], default=100, help='(int) Number of output classes (20 or 100)')

    # get command line arguments
    args = parser.parse_args()

    # Set variables to positional arguments if used, otherwise use flag specified arguments
    epochs = args.epochs if args.epochs is not None else args.epochs_flag
    batch_size = args.batch_size if args.batch_size is not None else args.batch_flag
    learning_rate = args.learning_rate if args.learning_rate is not None else args.lr_flag
    model_name = args.model if args.model is not None else args.model_flag
    num_classes = args.output_classes if args.output_classes is not None else args.output_classes_flag

    # print out args for debugging
    print(f'epochs={epochs}\nbatch_size={batch_size}\nlr={learning_rate}\nmodel=P{model_name}\noutput_classes={num_classes}')

    if epochs is None or batch_size is None or learning_rate is None:
        print('\nError: Must provide epochs, batch_size, and learning_rate as either positional or flagged arguments')
        print('\nExample script usage:')
        print('\tPositional arguments: python(3) train.py 5 32 0.005 Net 20')
        print('\tFlagged arguments: python(3) train.py --epochs 5 --batch_size 32 --learning_rate 0.005 --model Net --output_classes 20\n')
        sys.exit(1)

    # generate timestamp for filename when saving
    timestamp = str(datetime.now().timestamp())

    label_type = 'fine' if num_classes == 100 else 'coarse'

    # Downloads datasets if not already installed
    train_dataset = CIFAR100Custom(root='./data', train=True, download=True,
                               transform=transforms.ToTensor(), label_type=label_type)
    test_dataset = CIFAR100Custom(root='./data', train=False, download=True,
                               transform=transforms.ToTensor(), label_type=label_type)

    # Path for saving/loading model
    PATH = models_dir + '/model_' + model_name + '_' + str(epochs) + 'e-' + str(batch_size) + 'bs-' + str(learning_rate) + 'lr-' + 'cls_' + timestamp + '.pt'

    # runs train function
    train(epochs, batch_size, learning_rate, train_dataset, PATH, model_name, num_classes)
