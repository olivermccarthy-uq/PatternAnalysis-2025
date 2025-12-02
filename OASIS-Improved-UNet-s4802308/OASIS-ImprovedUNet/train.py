import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from modules import ImprovedUNet
from dataset import load_oasis_data

# CONFIG
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
EPOCHS = 35
LR = 1e-3

# Data loading: update paths as needed
train_images, train_labels = load_oasis_data(
    '/home/groups/comp3710/OASIS/keras_png_slices_train',
    '/home/groups/comp3710/OASIS/keras_png_slices_seg_train'
)
val_images, val_labels = load_oasis_data(
    '/home/groups/comp3710/OASIS/keras_png_slices_validate',
    '/home/groups/comp3710/OASIS/keras_png_slices_seg_validate'
)

class OasisDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)/255.0
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)/255.0

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_ds = OasisDataset(train_images, train_labels)
val_ds = OasisDataset(val_images, val_labels)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = ImprovedUNet(n_channels=1, n_classes=1).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, masks in train_dl:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    avg_train_loss = running_loss / len(train_dl.dataset)
    train_losses.append(avg_train_loss)

    model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        for vimgs, vmasks in val_dl:
            vimgs, vmasks = vimgs.to(device), vmasks.to(device)
            voutputs = model(vimgs)
            vloss = loss_fn(voutputs, vmasks)
            running_vloss += vloss.item() * vimgs.size(0)
    avg_val_loss = running_vloss / len(val_dl.dataset)
    val_losses.append(avg_val_loss)
    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Save checkpoint
    torch.save(model.state_dict(), f"unet_epoch{epoch+1}.pth")

# Plot losses
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()
