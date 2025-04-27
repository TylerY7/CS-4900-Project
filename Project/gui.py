import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, Label, Button
import os
import sys

# Set up paths for model
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'Dataset')
net_path = os.path.join(base_dir, 'Training')
model_path = os.path.join(base_dir, 'models', 'model_Net_1745781891.582031.pt')
sys.path.append(dataset_path)
sys.path.append(net_path)

from model_cnn import Net
from dataset_download_superclass import CIFAR100Custom

'''
# Remove this line later when we train models that save their label type
label_type = 'fine'

# Load labels
label_dataset = CIFAR100Custom(root='./data', train=False, download=True,
                                transform=transforms.ToTensor(), label_type=label_type)
label_names = label_dataset.classes

'''
#Use this block with new models that save label type
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
def predict_image(path):
    image = Image.open(path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()

    return label_names[pred_class]



# GUI Code
from PIL import ImageTk  # needed for displaying image

root = tk.Tk()

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

    fine_label = predict_image(x)
    var.set(f"Prediction:\n {label_type.title()} Label: {fine_label}")

    label = tk.Label(root, textvariable=var)
    label.grid(row=40, columnspan=4)

# Create a button and place it into the window using grid layout

root.grid_columnconfigure(1, weight=1)

btn = tk.Button(root, text='open image', command=open_img)
btn.grid(row=1, column=1, pady=10)

root.mainloop()

