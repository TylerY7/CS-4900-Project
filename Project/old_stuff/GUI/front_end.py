from tkinter import *

import random

#loading Python imaging Library
from PIL import ImageTk, Image


# To get the dialog box to open when required
from tkinter import filedialog


root = Tk()

# Set Title as Image Loader
root.title("Image Loader")

# Set the resolution of window
root.geometry("600x400")

# Allow Window to be resizeable
root.resizable(width = True, height = True)

var = StringVar()

stringVar = StringVar()
numberVar = StringVar()
label = Label(root, textvariable=var).grid(row=40, columnspan=4)


def openfilename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title ='Open')
    return  filename

def open_img():
    # Select the Image name from a folder
    x = openfilename()

    # opens the image
    img = Image.open(x)

    # resize the image and apply a high-quality down sampling filter
    img = img.resize ((250, 250))

    # PhotoImage class is used to add image to widgets, icons, etc
    img = ImageTk.PhotoImage(img)


    # create a label
    panel = Label(root, image = img)

    # set the image as img
    panel.image = img
    panel.grid(row = 2)

    label = Label(root, textvariable=var ).grid(row = 40, columnspan = 4)


    var.set("Image Displayed")

# Create a button and place it into the window using grid layout
btn = Button(root, text ='open image', command = open_img).grid(row = 1, columnspan = 4)

#label.pack()
root.mainloop()