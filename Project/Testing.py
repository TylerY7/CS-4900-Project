import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from Training.model_cnn import Net
import sys
from datetime import datetime
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# Append folder to path so python can find the module to import

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'Dataset')
sys.path.append(dataset_path)

import dataset_download
import argparse

class_to_superclass = {
        'apple': 'fruit', 'orange': 'fruit', 'banana': 'fruit', 'pineapple': 'fruit', 'grape': 'fruit',
        'peach': 'fruit', 'strawberry': 'fruit', 'cherry': 'fruit', 'lemon': 'fruit', 'pear': 'fruit',
        
        'aquarium_fish': 'fish', 'flatfish': 'fish', 'ray': 'fish', 'shark': 'fish', 'trout': 'fish',
        
        'orchid': 'flower', 'poppy': 'flower', 'rose': 'flower', 'sunflower': 'flower', 'tulip': 'flower',
        
        'bee': 'insect', 'beetle': 'insect', 'butterfly': 'insect', 'cockroach': 'insect', 'dragonfly': 'insect',
        
        'bear': 'mammal', 'cat': 'mammal', 'cow': 'mammal', 'dog': 'mammal', 'horse': 'mammal', 'elephant': 'mammal',
        'kangaroo': 'mammal', 'lion': 'mammal', 'tiger': 'mammal', 'wolf': 'mammal',
        
        'rocket': 'vehicle', 'airliner': 'vehicle', 'cab': 'vehicle', 'ambulance': 'vehicle', 'fire_engine': 'vehicle'
    }


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

def compute_precision(all_labels, all_predictions, classes):
    """
    Function to display precision score per class.
    Shows class names along with precision score associated with the class.
    """
    print("---------------------\n\nPer-Class Precision:")
    precision_matrix = precision_score(all_labels, all_predictions, average = None, zero_division=0)
    for i, cls in enumerate(classes):
        precision = precision_matrix[i]
        print(f'{cls}: {precision:.4f}')


# Computes macro percisions
def compute_macro_percision(all_labels, all_predictions):
    precision_matrix = precision_score(all_labels, all_predictions, average = 'macro')
    print(f"Percision Recall: {precision_matrix:.4f}")


def compute_recall(all_labels, all_predictions, classes):
    """
    Function to display recall score per class.
    Shows class names along with recall score associated with the class.
    """
    print("---------------------\n\nPer-Class Recall:")
    recall_matrix = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    for i, cls in enumerate(classes):
        recall = recall_matrix[i]
        print(f'{cls}: {recall:.4f}')

# Computes macro recall
def compute_macro_recall(all_labels, all_predictions):
    recall_matrix = recall_score(all_labels, all_predictions, average='macro')
    print(f"Macro Recall: {recall_matrix:.4f}")


def compute_f1(all_labels, all_predictions, classes):
    """
    Function to display f1 score per class.
    Shows class names along with f1 score associated with the class.
    """
    print("---------------------\n\nPer-Class f1:")
    f1_matrix = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    for i, cls in enumerate(classes):
        f1 = f1_matrix[i]
        print(f'{cls}: {f1:.4f}')


# Computes macro F1-scores 
def com_macro(all_labels, all_predictions):
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")


def compute_mean_accuracy_per_superclass(correct_per_class, total_per_class, classes):
    superclass_correct = {}
    superclass_total = {}


    for i, cls in enumerate(classes):
        superclass = classes
        superclass = class_to_superclass.get(cls, None)
        if superclass:
            if superclass not in superclass_correct:
                superclass_correct[superclass] = 0
                superclass_total[superclass] = 0
            superclass_correct[superclass] += correct_per_class[i]
            superclass_total[superclass] += total_per_class[i]

    print("\nMean Accuracy per Superclass:")
    for superclass, correct in superclass_correct.items():
        total = superclass_total[superclass]
        mean_accuracy = 100 * correct / total if total > 0 else 0
        print(f'{superclass}: {mean_accuracy:.2f}%')


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

    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
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

    # computes precision score per class
    compute_precision(all_labels, all_predictions, classes)

    # computes macro precision 
    compute_macro_percision(all_labels, all_predictions)

    # computes recall score per class
    compute_recall(all_labels, all_predictions, classes)

    # computes macro recall 
    compute_macro_recall(all_labels, all_predictions)

    # computes f1 score per class
    compute_f1(all_labels, all_predictions, classes)

    # Computes macro f1 score
    com_macro(all_labels, all_predictions)

    # Function to compute mean accuracy per superclass
    compute_mean_accuracy_per_superclass(correct_per_class, total_per_class, classes)
    
    # Computes metrics for each super class



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained CNN model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    
    args = parser.parse_args()
    test(args.model_path, args.batch_size)
