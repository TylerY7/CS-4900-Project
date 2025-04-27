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
    precision_matrix = precision_score(all_labels, all_predictions, average = 'macro', zero_division=0)
    print(f"Macro Percision: {precision_matrix:.4f}")


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
def compute_macro_f1(all_labels, all_predictions):
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")


def compute_per_class_accuracy_per_superclass(correct_per_class, total_per_class, classes):
    """
    Function to compute per-class accuracy grouped by superclass.
    """
    superclass_to_classes = {}

    # First, build a mapping from superclass to the list of classes
    for i, cls in enumerate(classes):
        superclass = class_to_superclass.get(cls, None)
        if superclass:
            if superclass not in superclass_to_classes:
                superclass_to_classes[superclass] = []
            superclass_to_classes[superclass].append(i)  # store the index, not the name

    # Now, compute and print per-class accuracy under each superclass
    print("\nPer-Class Accuracy within Each Superclass:")
    for superclass, indices in superclass_to_classes.items():
        print(f"\nSuperclass: {superclass}")
        for idx in indices:
            cls = classes[idx]
            if total_per_class[idx] > 0:
                acc = 100 * correct_per_class[idx] / total_per_class[idx]
            else:
                acc = 0.0
            print(f"  {cls}: {acc:.2f}%")

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

def compute_precision_for_superclass(all_labels, all_predictions, classes):
    """
    Function to compute precision score per superclass.
    """
    # Group all the labels and predictions by superclass
    superclass_labels = {}
    superclass_predictions = {}

    for i, label in enumerate(all_labels):
        superclass = class_to_superclass.get(classes[label], None)
        if superclass:
            if superclass not in superclass_labels:
                superclass_labels[superclass] = []
                superclass_predictions[superclass] = []
            superclass_labels[superclass].append(label)
            superclass_predictions[superclass].append(all_predictions[i])

    # Now, compute and print precision for each superclass
    print("\nPrecision per Superclass:")
    for superclass in superclass_labels:
        precision = precision_score(superclass_labels[superclass], superclass_predictions[superclass], average='macro', zero_division=0)
        print(f"{superclass}: {precision:.4f}")


def compute_recall_for_superclass(all_labels, all_predictions, classes):
    """
    Function to compute recall score per superclass.
    """
    # Group all the labels and predictions by superclass
    superclass_labels = {}
    superclass_predictions = {}

    for i, label in enumerate(all_labels):
        superclass = class_to_superclass.get(classes[label], None)
        if superclass:
            if superclass not in superclass_labels:
                superclass_labels[superclass] = []
                superclass_predictions[superclass] = []
            superclass_labels[superclass].append(label)
            superclass_predictions[superclass].append(all_predictions[i])

    # Now, compute and print recall for each superclass
    print("\nRecall per Superclass:")
    for superclass in superclass_labels:
        recall = recall_score(superclass_labels[superclass], superclass_predictions[superclass], average='macro', zero_division=0)
        print(f"{superclass}: {recall:.4f}")


def compute_f1_for_superclass(all_labels, all_predictions, classes):
    """
    Function to compute f1 score per superclass.
    """
    # Group all the labels and predictions by superclass
    superclass_labels = {}
    superclass_predictions = {}

    for i, label in enumerate(all_labels):
        superclass = class_to_superclass.get(classes[label], None)
        if superclass:
            if superclass not in superclass_labels:
                superclass_labels[superclass] = []
                superclass_predictions[superclass] = []
            superclass_labels[superclass].append(label)
            superclass_predictions[superclass].append(all_predictions[i])

    # Now, compute and print f1 score for each superclass
    print("\nF1 Score per Superclass:")
    for superclass in superclass_labels:
        f1 = f1_score(superclass_labels[superclass], superclass_predictions[superclass], average='macro')
        print(f"{superclass}: {f1:.4f}")

def compute_macro_precision_for_superclass(all_labels, all_predictions, classes):
    # Group all the labels and predictions by superclass
    superclass_labels = {}
    superclass_predictions = {}

    for i, label in enumerate(all_labels):
        superclass = class_to_superclass.get(classes[label], None)
        if superclass:
            if superclass not in superclass_labels:
                superclass_labels[superclass] = []
                superclass_predictions[superclass] = []
            superclass_labels[superclass].append(label)
            superclass_predictions[superclass].append(all_predictions[i])

    # Now, compute and print macro precision for each superclass
    print("\nMacro Precision per Superclass:")
    for superclass in superclass_labels:
        precision = precision_score(superclass_labels[superclass], superclass_predictions[superclass], average='macro', zero_division=0)
        print(f"{superclass}: {precision:.4f}")

# Computes macro recall for each superclass
def compute_macro_recall_for_superclass(all_labels, all_predictions, classes):
    # Group all the labels and predictions by superclass
    superclass_labels = {}
    superclass_predictions = {}

    for i, label in enumerate(all_labels):
        superclass = class_to_superclass.get(classes[label], None)
        if superclass:
            if superclass not in superclass_labels:
                superclass_labels[superclass] = []
                superclass_predictions[superclass] = []
            superclass_labels[superclass].append(label)
            superclass_predictions[superclass].append(all_predictions[i])

    # Now, compute and print macro recall for each superclass
    print("\nMacro Recall per Superclass:")
    for superclass in superclass_labels:
        recall = recall_score(superclass_labels[superclass], superclass_predictions[superclass], average='macro', zero_division=0)
        print(f"{superclass}: {recall:.4f}")

# Computes macro F1 for each superclass
def compute_macro_f1_for_superclass(all_labels, all_predictions, classes):
    # Group all the labels and predictions by superclass
    superclass_labels = {}
    superclass_predictions = {}

    for i, label in enumerate(all_labels):
        superclass = class_to_superclass.get(classes[label], None)
        if superclass:
            if superclass not in superclass_labels:
                superclass_labels[superclass] = []
                superclass_predictions[superclass] = []
            superclass_labels[superclass].append(label)
            superclass_predictions[superclass].append(all_predictions[i])

    # Now, compute and print macro F1 for each superclass
    print("\nMacro F1 Score per Superclass:")
    for superclass in superclass_labels:
        f1 = f1_score(superclass_labels[superclass], superclass_predictions[superclass], average='macro', zero_division=0)
        print(f"{superclass}: {f1:.4f}")


def test(model_path, batch_size, evaluate_only_super):
    """
    Function to test the trained model on the test dataset.

    Args:
        model_path (string): model path for the trained model
        batch_size (int): batch size for testing
        evaluate_only_super (string): Chooses between evaluating only on super class metrics (if model only trained on super class),
                         or both super class and class metrics (if model was trained with classes as ground truths) (Choices = y or n)
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
    
    if(evaluate_only_super == 'n'):
        # Compute overall accuracy
        accuracy = 100 * correct / total
        print(f'Overall Accuracy: {accuracy:.2f}%')

        # Compute per-class accuracy
        compute_metrics(correct_per_class, total_per_class, classes)

        # computes precision score per class
        compute_precision(all_labels, all_predictions, classes)

        # computes recall score per class
        compute_recall(all_labels, all_predictions, classes)

        # computes f1 score per class
        compute_f1(all_labels, all_predictions, classes)

        # computes macro precision 
        compute_macro_percision(all_labels, all_predictions)

        # computes macro recall 
        compute_macro_recall(all_labels, all_predictions)

        # Computes macro f1 score
        compute_macro_f1(all_labels, all_predictions)

    # Computes the per-class accuracy over the whole test set for each super class 
    compute_per_class_accuracy_per_superclass(correct_per_class, total_per_class, classes)

    # Function to compute mean accuracy per superclass
    compute_mean_accuracy_per_superclass(correct_per_class, total_per_class, classes)
    
    # Computes precision for each superclass
    compute_precision_for_superclass(all_labels, all_predictions, classes)

    # Computes recall for each superclass
    compute_recall_for_superclass(all_labels, all_predictions, classes)

    # Computes F1-scores for each superclass
    compute_f1_for_superclass(all_labels, all_predictions, classes)

    # Computes macro precision for each superclass
    compute_macro_precision_for_superclass(all_labels, all_predictions, classes)

    # Computes macro recall for each superclass
    compute_macro_recall_for_superclass(all_labels, all_predictions, classes)

    # Computes macro F1-scores for each superclass
    compute_macro_f1_for_superclass(all_labels, all_predictions, classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained CNN model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--evaluate_only_super', type=str, required=True, choices=['y', 'n'],
                         help='(str) Evaluates only on super class metrics or both class and super class metrics')
    
    args = parser.parse_args()
    test(args.model_path, args.batch_size, args.evaluate_only_super)

# models\model_1742768151.282855.pt