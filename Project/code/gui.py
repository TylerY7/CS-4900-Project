import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, Label, Button
import os
import sys

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

# Set up paths for model
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, '..', 'Dataset')
net_path = os.path.join(base_dir, '..', 'Training')
model_path = os.path.join(base_dir, '..', 'models', 'model_Net_150e-128bs-0.01lr-100cls_1745798565.021512.pt')
sys.path.append(dataset_path)
sys.path.append(net_path)

from model_cnn import Net
from dataset_download_superclass import CIFAR100Custom

# Load checkpoint
checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
label_type = checkpoint['label_type']
num_classes = checkpoint['num_classes']

# Load labels based on label type
label_dataset = CIFAR100Custom(root='./data', train=False, download=True,
                               transform=transforms.ToTensor(), label_type=label_type)
label_names = label_dataset.classes

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(num_classes)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

# Make Prediction
def predict_image(path, top_k=5):
    image = Image.open(path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        #pred_class = probs.argmax(dim=1).item()
        top_probs, top_indices = torch.topk(probs, top_k)

    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()

    top_predictions = []
    for idx, prob in zip(top_indices, top_probs):
        fine_label = label_names[idx]
        coarse_label = superclasses[coarse_labels[idx]]
        top_predictions.append((fine_label, coarse_label, prob))

    return top_predictions




# GUI Code
from PIL import ImageTk  # needed for displaying image

root = tk.Tk()
root.state('zoomed')

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

# Set Title as Image Loader
root.title("Image Loader")

# Set the resolution of window
root.geometry("600x400")

# Allow Window to be resizeable
root.resizable(width=True, height=True)

var = tk.StringVar()
stringVar = tk.StringVar()
numberVar = tk.StringVar()

label = tk.Label(root, textvariable=var)
label.grid(row=40, columnspan=4)

panel = None

def openfilename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title='Open')
    return filename

def open_img():
    global panel
    # Select the Image name from a folder
    x = openfilename()

    if not x:
        return

    # opens the image
    img = Image.open(x)

    # resize the image and apply a high-quality down sampling filter
    img = img.resize((250, 250))

    # PhotoImage class is used to add image to widgets, icons, etc
    img = ImageTk.PhotoImage(img)

    # create a label
    if panel:
        panel.configure(image=img)
        panel.image = img
        panel.grid(row=2, column=1)

    else:
        panel = tk.Label(root, image=img)
        panel.image = img
        panel.grid(row=2, column=1)

    predictions = predict_image(x)
    output_lines = ["Top Predictions:"]
    for i, (fine, coarse, prob) in enumerate(predictions):
        output_lines.append(f'{i+1}. {fine} ({coarse}) - {prob*100:.2f}%)')

    chosen_fine, chosen_coarse, _ = predictions[0]
    output_lines.append(f"\nFinal Prediction:\nClass - {chosen_fine}\nsuper class - {chosen_coarse}")

    var.set("\n".join(output_lines))

    #var.set(f"Prediction:\nClass: {fine_label}\nSuper Class: {coarse_label}")

    label = tk.Label(root, textvariable=var)
    label.grid(row=40, columnspan=4)

# Create a button and place it into the window using grid layout

root.grid_columnconfigure(1, weight=1)

btn = tk.Button(root, text='open image', command=open_img)
btn.grid(row=1, column=1, pady=10)

root.mainloop()

