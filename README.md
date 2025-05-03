![Project Banner](assets/banner.jpg)

# CIFAR-100 Image Classifier GUI

A Python-based image classification app that uses a custom-trained convolutional neural network (CNN) to classify images from the CIFAR-100 dataset. The application includes a Tkinter GUI, allowing users to select an image and view both the predicted class and super-class.

This project fulfills the A-grade requirements for the CS: 4900 Capstone course at Valdosta State University.

## ✨ Features

- Custom-trained CNN using PyTorch (no pre-trained models)
- Additional linear model implementation (no convolution layers)
- Tkinter GUI for user-friendly interaction
- Image loading and prediction display
- Model selection via command-line arguments
- Class and super-class prediction support
- Modular and well-documented codebase
- TensorBoard logging for loss tracking

---

## 🧠 Dataset

- **Dataset**: CIFAR-100  
- **Source**: https://www.cs.toronto.edu/~kriz/cifar.html  
- **Classes**: 100 fine-grained classes grouped into 20 super-classes  
- **Data Split**:
  - 80% training
  - 20% validation
  - Separate test set for final evaluation

---

## 🏗️ Project Structure

```
Project/
├── code/                                 # Core training and testing logic, graphing, and GUI
|   └── dataset_download_superclass.py    # Script for handling dataset downloading for training
│   └── generate_cifar_images.py          # Script for generating png files for GUI testing
|   └── gui.py                            # Tkinter GUI for loading images and showing predictions
|   └── linear_model.py                   # Script containing linear model
│   └── model_cnn.py                      # Script containing CNN model
│   └── tensorboard_graphing.py           # Model graphing based on saved information in runs
|   └── test.py                           # Model testing script
│   └── train.py                          # Model training script
├── models/                               # Contains saved models
├── runs/                                 # TensorBoard logs and model checkpoints
└── README.md                             # Project documentation
```
## 🚀 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/CS-4900-Project.git
   cd CS-4900-Project
   ```

2. **Install dependencies manually**  
   Make sure you have Python 3.10+ installed and install the necessary packages:

   ```bash
   pip install torch torchvision matplotlib tqdm
   pip install scikit-learn tensorboard
   pip install Pillow
   ```

3. **Download CIFAR-100 dataset**  
   The dataset will be automatically downloaded the first time you run your training script, using:

   ```python
   torchvision.datasets.CIFAR100(root='./Dataset', train=True, download=True)
   ```

## 🧪 Training

To train a model from scratch, run the training script from the `code/` directory:

```bash
python code/train.py --model=Net --epochs=50 --learning_rate=0.005 --batch_size=64 --output_classes=100
```

or:

```bash
python code/train.py 5 32 0.005 Net 20
```

### Available Arguments

- `--model`: Choose between `Net` for CNN or `LinearModel` for LinearModel
- `--output_classes`: Set to `100` (fine labels) or `20` (super-class labels)
- `--learning_rate`: Learning rate (e.g. `0.005`)
- `--batch_size`: Number of samples per batch
- `--epochs`: Total number of training epochs

All training logs are saved and compatible with **TensorBoard** for visualization.

You can launch TensorBoard with:

```bash
tensorboard --logdir=runs
```

or by running the tensorboard script from the `code/` directory:

```bash
python code/tensorboard_graphing.py
```

## 📈 Evaluation

To evaluate a trained model, run the test script from the `code/` directory once you have the file path
of the model:

```bash
python code/test.py --model_path models\\model_Net_250e-64bs-0.005lr-100cls_1745839241.032571.pt --batch_size 32 --evaluate_only_super y
```

Note: the string provided as model path must start with models\\ so that the correct directory will be searched.

### Available Arguments

- `--evaluate_only_super`: either `y` (shows only super class metrics) or `n` (shows both super class and class metrics)
- `--batch_size`: batch size as an integer (defaults to 32)
- `--model_path`: Path to the trained model


### Evaluation Metrics

The following metrics are computed:

- **Per-class accuracy**
- **Mean accuracy** across all classes
- **Precision**, **Recall**, and **F1-score** (Macro-averaged)
- Optionally: Super-class versions of the above metrics (if applicable)

## 🖼️ GUI Usage

To launch the GUI, run the GUI script from the `code/` directory:

```bash
python code/gui.py
```

### Features

- Open a file dialog to select an image from disk
- Display the selected image within the app
- Use a trained model to classify the image
- Display both the predicted **class** and **super-class**
- Designed for non-technical users — no command-line interaction required

## 🧠 Models

### CNN (Convolutional Neural Network)

A custom CNN trained from scratch using the CIFAR-100 dataset. It consists of multiple convolutional layers followed by ReLU activations, max pooling, and fully connected layers.

**Training configuration:**

- Optimizer: Stochastic Gradient Descent (SGD)
- Loss Function: CrossEntropyLoss
- Batch Size: 64
- Epochs: up to 50
- Output Classes: `100` (class labels) or `20` (super-class labels)

---

### Linear Model

A simpler model architecture using only fully connected (linear) layers with no convolutional components. This model was implemented to satisfy the A-grade requirements and is used as a comparison baseline.

- Useful for demonstrating the performance gap between deep CNNs and shallow models
- Also trained on CIFAR-100 with the same evaluation pipeline

## 🎓 Grading Goals

This project satisfies the **A-level criteria** outlined in the CS: 4900 Capstone documentation:

- Command-line argument support for:
  - Number of epochs
  - Learning rate
  - Batch size
  - Model type (`Net` or `LinearModel`)
  - Output type (`100` or `20`)
- Implementation of both a CNN and a linear (fully connected) model
- Training and testing workflows for both class and super-class label configurations
- Modular code structure using `if __name__ == "__main__"` blocks
- Clear function-based organization with docstrings and inline comments
- Evaluation includes per-class and super-class metrics
- Visualizations of training performance using TensorBoard

## 📚 References

- [PyTorch](https://pytorch.org)  
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
