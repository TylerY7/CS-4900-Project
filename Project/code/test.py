"""
File for testing saved models from training. Models must be in models directory under the Project directory.
To run, navigate to the directory of test.py and include command line arguments
for the model path to the saved model as a string, desired batch size
for testing as an integer (default is 32), and whether to only display super class metrics
or class metrics when the model has finished training ('y' or 'n').

Command line arguments example:
python test.py --model_path models\\model_Net_250e-64bs-0.005lr-100cls_1745839241.032571.pt --batch_size 32 --evaluate_only_super y
Note: the string provided as model path must start with models\\ so that the correct directory will be searched.
"""
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model_cnn import Net
from linear_model import LinearModel
import sys
from datetime import datetime
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# Append folder to path so python can find the module to import

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

import argparse


# CIFAR-100 class labels (index = class id)
classes = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

# Superclass labels
superclasses = [
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture", "insects", "large_carnivores", "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes", "large_omnivores_and_herbivores", "medium_mammals", "non-insect_invertebrates",
    "people", "reptiles", "small_mammals", "trees", "vehicles_1", "vehicles_2"
]

# Coarse labels per class (index = fine class ID, value = coarse class ID)
coarse_labels = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13
]

# Final mapping: class name → superclass name
class_to_superclass = {
    classes[i]: superclasses[coarse_labels[i]]
    for i in range(len(classes))
}


# Function to get class names
def get_classes(dataset):
    """
    Function to get class names.

    Args:
        dataset (CIFAR100Custom): CIFAR100 dataset
    
    Returns:
        list: returns list of class names in the dataset
    """
    return dataset.classes

# Function to compute accuracy per class
def compute_metrics(correct_per_class, total_per_class, classes):
    """
    Function using during testing (if class metrics are evaluated) to compute per class accuracy and prints name of each class with its accuracy.
    Uses arguments given from testing.

    Args:
        correct_per_class (list): list of values from 0 onwards representing how many correct predictions the model made per class
        total_per_class (list): list of values from 0 onwards representing how many images from each class were used
        classes (list): list of class names from dataset
    """
    print("Per-Class Accuracy:")
    for i, cls in enumerate(classes):
        if total_per_class[i] > 0:
            acc = 100 * correct_per_class[i] / total_per_class[i]
        else:
            acc = 0.0
        print(f'{cls}: {acc:.2f}%')

def compute_precision(all_labels, all_predictions, classes):
    """
    Function using during testing (if class metrics are evaluated) to compute per class precision and prints name of each class with its precision.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
        classes (list): list of class names from dataset
    """
    print("---------------------\n\nPer-Class Precision:")
    precision_matrix = precision_score(all_labels, all_predictions, average = None, zero_division=0)
    for i, cls in enumerate(classes):
        precision = precision_matrix[i]
        print(f'{cls}: {precision:.4f}')


# Computes macro percisions
def compute_macro_percision(all_labels, all_predictions):
    """
    Function using during testing (if class metrics are evaluated) to compute and print macro precision.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
    """
    precision_matrix = precision_score(all_labels, all_predictions, average = 'macro', zero_division=0)
    print(f"Macro Percision: {precision_matrix:.4f}")


def compute_recall(all_labels, all_predictions, classes):
    """
    Function using during testing (if class metrics are evaluated) to compute per class recall score and prints name of each class with its recall score.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
        classes (list): list of class names from dataset
    """
    print("---------------------\n\nPer-Class Recall:")
    recall_matrix = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    for i, cls in enumerate(classes):
        recall = recall_matrix[i]
        print(f'{cls}: {recall:.4f}')

# Computes macro recall
def compute_macro_recall(all_labels, all_predictions):
    """
    Function using during testing (if class metrics are evaluated) to compute and print macro recall.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
    """
    recall_matrix = recall_score(all_labels, all_predictions, average='macro')
    print(f"Macro Recall: {recall_matrix:.4f}")


def compute_f1(all_labels, all_predictions, classes):
    """
    Function using during testing (if class metrics are evaluated) to compute per class f1 score and prints name of each class with its f1 score.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
        classes (list): list of class names from dataset
    """
    print("---------------------\n\nPer-Class f1:")
    f1_matrix = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    for i, cls in enumerate(classes):
        f1 = f1_matrix[i]
        print(f'{cls}: {f1:.4f}')


# Computes macro F1-scores 
def compute_macro_f1(all_labels, all_predictions):
    """
    Function using during testing to compute and print macro f1 score if class metrics are evaluated.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
    """
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")


def compute_per_class_accuracy_per_superclass(correct_per_class, total_per_class, classes):
    """
    Function using during testing to compute per class accuracy grouped by super class and prints per class accuracy under each super class.
    Uses arguments given from testing.

    Args:
        correct_per_class (list): list of values from 0 onwards representing how many correct predictions the model made per class
        total_per_class (list): list of values from 0 onwards representing how many images from each class were used
        classes (list): list of class names from dataset
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
    """
    Function using during testing to compute and print mean accuracy per super class.
    Uses arguments given from testing.

    Args:
        correct_per_class (list): list of values from 0 onwards representing how many correct predictions the model made per class
        total_per_class (list): list of values from 0 onwards representing how many images from each class were used
        classes (list): list of class names from dataset
    """
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
    Function using during testing to compute and print precision scores per super class.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
        classes (list): list of class names from dataset
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
    Function using during testing to compute and print recall score per superclass.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
        classes (list): list of class names from dataset
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
    Function using during testing to compute and print f1 score per superclass.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
        classes (list): list of class names from dataset
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
    """
    Function using during testing to compute and print macro precision scores per superclass.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
        classes (list): list of class names from dataset
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

    # Now, compute and print macro precision for each superclass
    print("\nMacro Precision per Superclass:")
    for superclass in superclass_labels:
        precision = precision_score(superclass_labels[superclass], superclass_predictions[superclass], average='macro', zero_division=0)
        print(f"{superclass}: {precision:.4f}")

# Computes macro recall for each superclass
def compute_macro_recall_for_superclass(all_labels, all_predictions, classes):
    """
    Function using during testing to compute and print macro recall scores per superclass.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
        classes (list): list of class names from dataset
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

    # Now, compute and print macro recall for each superclass
    print("\nMacro Recall per Superclass:")
    for superclass in superclass_labels:
        recall = recall_score(superclass_labels[superclass], superclass_predictions[superclass], average='macro', zero_division=0)
        print(f"{superclass}: {recall:.4f}")

# Computes macro F1 for each superclass
def compute_macro_f1_for_superclass(all_labels, all_predictions, classes):
    """
    Function using during testing to compute and print macro f1 scores per superclass.
    Uses arguments given from testing.

    Args:
        all_labels (list): list of numpy.int64 values representing all labels in each testing batch
        all_predictions (list): list of numpy.int64 values representing all predictions the model made during testing
        classes (list): list of class names from dataset
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

    # Now, compute and print macro F1 for each superclass
    print("\nMacro F1 Score per Superclass:")
    for superclass in superclass_labels:
        f1 = f1_score(superclass_labels[superclass], superclass_predictions[superclass], average='macro', zero_division=0)
        print(f"{superclass}: {f1:.4f}")


def test(model_path, batch_size, evaluate_only_super):
    """
    Function to test the trained model on the test dataset. Loads model from given model_path.
    If evaluate_only_super is 'n', class metrics will be printed first followed by super class metrics.
    If 'y', only super class metrics will be printed.

    Args:
        model_path (string): model path for the trained model
        batch_size (int): batch size for testing
        evaluate_only_super (string): Chooses between evaluating only on super class metrics (if model only trained on super class), or both super class and class metrics (if model was trained with classes as ground truths) (Choices = y or n)
    """
    # Define transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load test dataset
    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    classes = get_classes(test_dataset)

    # Load the trained model
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    #label_type = checkpoint['label_type']
    num_classes = checkpoint['num_classes']

    # check if model is linear or cnn
    model = None
    if("Net" in model_path):
        print("Starting testing of CNN model")
        model = Net(num_classes)
    else:
        print("Starting testing of linear model")
        model = LinearModel(num_classes)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    correct = 0
    total = 0
    correct_per_class = [0] * len(classes)
    total_per_class = [0] * len(classes)

    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
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
    
    # determines whether or not to show only super class metrics if 'y' or both super class and class metrics if 'n'
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
                         help='(str) Displays only super class metrics or both class and super class metrics')
    
    args = parser.parse_args()

    # finds model in models folder
    models_model_path = os.path.join(os.path.sep,parent_dir,args.model_path)

    test(models_model_path, args.batch_size, args.evaluate_only_super)

# for testing (deleter later): models\model_LinearModel_5e-32bs-0.005lr-cls_1746209107.30725.pt
# Project\models\model_Net_250e-64bs-0.005lr-100cls_1745839241.032571.pt
# Project\models\super_scale\model_Net_300e-64bs-0.008lr-20cls_1745849542.993408.pt
