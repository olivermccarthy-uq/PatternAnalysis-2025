import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# -----------------------------
# Dataset Class
# -----------------------------
class MRIDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")])
        if len(self.images) == 0:
            raise ValueError(f"No PNG images found in folder: {folder}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')  # grayscale
        if self.transform:
            img = self.transform(img)
        return img

# -----------------------------
# Resolve paths relative to script
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))  # folder of this script
data_base = os.path.abspath(os.path.join(script_dir, "../../keras_png_slices_data"))

train_folder = os.path.join(data_base, "keras_png_slices_train")
validate_folder = os.path.join(data_base, "keras_png_slices_validate")
test_folder = os.path.join(data_base, "keras_png_slices_test")

print("Train folder:", train_folder)
print("Exists?", os.path.exists(train_folder))

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# Load Datasets
# -----------------------------
train_dataset = MRIDataset(train_folder, transform=transform)
val_dataset = MRIDataset(validate_folder, transform=transform)
test_dataset = MRIDataset(test_folder, transform=transform)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# -----------------------------
# Quick Visualization
# -----------------------------
plt.figure(figsize=(8,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(train_dataset[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()

# -----------------------------
# Create DataLoaders (for next commit)
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
