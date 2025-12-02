"""
MRI Dataset Loading and Preprocessing Module

PURPOSE: This file handles the loading and preprocessing of MRI brain scan images for training a VAE.
It creates a custom PyTorch Dataset class that can load PNG images from directories, apply 
transforms (resizing, normalization), and create DataLoaders for efficient batch processing.

WHY IT'S NEEDED: 
- Raw MRI images need to be preprocessed (resized to consistent dimensions, normalized)
- PyTorch requires a custom Dataset class to work with our file structure
- DataLoaders enable efficient batch processing during training
- Visualisation helps verify the data is loaded correctly
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# -----------------------------
# Custom Dataset Class for MRI Images
# -----------------------------
class MRIDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading MRI brain scan images.
    
    This class inherits from PyTorch's Dataset and implements the required methods
    to load PNG images from a directory, apply transforms, and return them as tensors.
    """
    
    def __init__(self, folder, transform=None):
        """
        Initialize the dataset with a folder containing PNG images.
        
        Args:
            folder (str): Path to directory containing PNG images
            transform (callable, optional): Transform to apply to each image
        """
        self.folder = folder
        self.transform = transform
        # Get all PNG files from the folder and sort them for consistent ordering
        self.images = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")])
        
        # Ensure we have images to work with
        if len(self.images) == 0:
            raise ValueError(f"No PNG images found in folder: {folder}")

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Load and return a single image at the given index.
        
        Args:
            idx (int): Index of the image to load
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image and convert to grayscale (single channel)
        img = Image.open(self.images[idx]).convert('L')  # 'L' mode = grayscale
        
        # Apply transforms if provided (resize, normalize, etc.)
        if self.transform:
            img = self.transform(img)
        return img

# -----------------------------
# Data Path Configuration
# -----------------------------
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))  # folder of this script

# Navigate to the data directory (two levels up from scripts folder)
data_base = os.path.abspath(os.path.join(script_dir, "../../keras_png_slices_data"))

# Define paths to train, validation, and test data folders
train_folder = os.path.join(data_base, "keras_png_slices_train")
validate_folder = os.path.join(data_base, "keras_png_slices_validate")
test_folder = os.path.join(data_base, "keras_png_slices_test")

# Verify the data paths exist
print("Train folder:", train_folder)
print("Exists?", os.path.exists(train_folder))

# -----------------------------
# Image Preprocessing Pipeline
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64,64)),        # Resize all images to 64x64 pixels for consistency
    transforms.ToTensor(),             # Convert PIL image to PyTorch tensor (0-1 range)
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range (better for VAE training)
])

# -----------------------------
# Dataset Creation
# -----------------------------
# Create dataset objects for train, validation, and test sets
train_dataset = MRIDataset(train_folder, transform=transform)
val_dataset = MRIDataset(validate_folder, transform=transform)
test_dataset = MRIDataset(test_folder, transform=transform)

# Display dataset sizes
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# -----------------------------
# Data Visualization
# -----------------------------
# Display a sample of training images to verify data loading
plt.figure(figsize=(8,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    # .squeeze() removes the channel dimension for display (1,64,64) -> (64,64)
    plt.imshow(train_dataset[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()

# -----------------------------
# DataLoader Creation
# -----------------------------
# Create DataLoaders for efficient batch processing during training
# DataLoaders handle batching, shuffling, and parallel data loading
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)   # Shuffle for training
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)      # No shuffle for validation
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
