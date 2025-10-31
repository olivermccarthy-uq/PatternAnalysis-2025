import torch
import matplotlib.pyplot as plt
from modules import ImprovedUNet
from dataset import load_oasis_data
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedUNet(n_channels=1, n_classes=1).to(device)
model.load_state_dict(torch.load('unet_epoch15.pth', map_location=device))
model.eval()

test_images, test_labels = load_oasis_data(
    '/home/groups/comp3710/OASIS/keras_png_slices_test',
    '/home/groups/comp3710/OASIS/keras_png_slices_seg_test'
)

images_tensor = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)/255.0
labels_tensor = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)/255.0

# Use DataLoader for batching
test_ds = TensorDataset(images_tensor, labels_tensor)
test_dl = DataLoader(test_ds, batch_size=8, shuffle=False)
preds = []

with torch.no_grad():
    for xb, _ in test_dl:
        out = model(xb.to(device))
        out = torch.sigmoid(out) > 0.5
        preds.append(out.cpu())
preds = torch.cat(preds, dim=0)

# Visualize 3 sample predictions
for i in range(3):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(test_images[i], cmap='gray')
    axs[0].set_title('Input')
    axs[1].imshow(test_labels[i], cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(preds[i][0], cmap='gray')
    axs[2].set_title('Prediction')
    plt.savefig(f'prediction_{i}.png')
    plt.close(fig)

