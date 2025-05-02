![Project Banner](assets/banner.png)

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
├── Dataset/             # Scripts or files for loading and preprocessing the CIFAR-100 dataset
├── GUI/                 # Tkinter GUI for loading images and showing predictions
│   └── gui.py
├── Code/                # Core training and testing logic
│   ├── Training/        # Training scripts
│   │   └── train.py     # CNN and linear model training script
│   └── Testing.py       # Model evaluation script
├── models/              # Contains CNN and linear model definitions
│   └── model_defs.py
├── runs/                # TensorBoard logs and model checkpoints
└── README.md            # Project documentation
```
## 🚀 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/CS-4900-Project.git
   cd CS-4900-Project

2. **Install dependencies manually**  
   Make sure you have Python 3.10+ installed and install the necessary packages:

   ```bash
   pip install torch torchvision matplotlib tqdm
   pip install scikit-learn tensorboard
   ```

3. **Download CIFAR-100 dataset**  
   The dataset will be automatically downloaded the first time you run your training script, using:

   ```python
   torchvision.datasets.CIFAR100(root='./Dataset', train=True, download=True)
   ```

## 🧪 Training

To train a model from scratch, run the training script from the `Code/Training/` directory:

```bash
python Code/Training/train.py --model=cnn --num-epochs=50 --lr=0.005 --batch-size=64 --num-classes=100
```

### Available Arguments

- `--model`: Choose between `cnn` or `linear`
- `--num-classes`: Set to `100` (fine labels) or `20` (super-class labels)
- `--lr`: Learning rate (e.g. `0.005`)
- `--batch-size`: Number of samples per batch
- `--num-epochs`: Total number of training epochs

All training logs are saved and compatible with **TensorBoard** for visualization.

You can launch TensorBoard with:

```bash
tensorboard --logdir=runs
```
## 📈 Evaluation

To evaluate a trained model, run the test script from the `Code/` directory:

```bash
python Code/Testing.py --model=cnn --num-classes=100 --weights=models/cnn_class_weights.pth
```

### Available Arguments

- `--model`: Model type to evaluate (`cnn` or `linear`)
- `--num-classes`: Either `100` (class labels) or `20` (super-class labels)
- `--weights`: Path to the trained model `.pth` file

### Evaluation Metrics

The following metrics are computed:

- **Per-class accuracy**
- **Mean accuracy** across all classes
- **Precision**, **Recall**, and **F1-score** (Macro-averaged)
- Optionally: Super-class versions of the above metrics (if applicable)

## 🖼️ GUI Usage

To launch the GUI, run the following command:

```bash
python GUI/gui.py
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
  - Model type (`cnn` or `linear`)
  - Output type (`class` or `super-class`)
- Implementation of both a CNN and a linear (fully connected) model
- Training and testing workflows for both class and super-class label configurations
- Modular code structure using `if __name__ == "__main__"` blocks
- Clear function-based organization with docstrings and inline comments
- Evaluation includes per-class and super-class metrics
- Visualizations of training performance using TensorBoard

## 📚 References

- [PyTorch](https://pytorch.org)  
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
