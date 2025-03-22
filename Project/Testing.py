import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from model_cnn import Net
import sys
from datetime import datetime
import dataset_download
import argparse

# Function to get class names
def get_classes(dataset):
    return dataset.classes

# Function to compute accuracy per class
def compute_metrics(correct_per_class, total_per_class, classes):
    print("Per-Class Accuracy:")
    for i, cls in enumerate(classes):
        if total_per_class[i] > 0:
            acc = 100 * correct_per_class[i] / total_per_class[i]
        else:
            acc = 0.0
        print(f'{cls}: {acc:.2f}%')


def test(model_path, batch_size):
    """
    Function to test the trained model on the test dataset.
    """
    # Define transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load test dataset
    test_dataset = dataset_download.download_test_dataset(transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    classes = get_classes(test_dataset)

    # Load the trained model
    net = Net(len(classes))
    net.load_state_dict(torch.load(model_path))
    net.eval()
    
    correct = 0
    total = 0
    correct_per_class = [0] * len(classes)
    total_per_class = [0] * len(classes)
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                total_per_class[label] += 1
                if predicted[i].item() == label:
                    correct_per_class[label] += 1
    
    # Compute overall accuracy
    accuracy = 100 * correct / total
    print(f'Overall Accuracy: {accuracy:.2f}%')
    
    # Compute per-class accuracy
    compute_metrics(correct_per_class, total_per_class, classes)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained CNN model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    
    args = parser.parse_args()
    test(args.model_path, args.batch_size)
