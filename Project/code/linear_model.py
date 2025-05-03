"""
Linear model for use in image classification. Model takes an input image, flattens it into a 1D tensor, and passes it 
through fully connected layers to make predictions on classification.
"""
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, num_classes):  
        """
        Creates linear model with given output classes. Defines fully connected layers and mapping.

        Args:
            num_classes (int): number of output classes for classification
        """
        super(LinearModel, self).__init__()
        # Flatten the input image (3x32x32) to 1D (3072)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)           
        self.fc3 = nn.Linear(512, 256)            
        self.fc4 = nn.Linear(256, 128)            
        self.fc5 = nn.Linear(128, num_classes)    

    def forward(self, x):
        """
        Performs forward pass and feeds input through model layers.

        Args:
            x (torch.Tensor): input tensor with image

        Returns:
            x (torch.Tensor): output tensor containing logits of shape
        """
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
