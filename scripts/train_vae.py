"""
VAE Training Script with Validation and Reconstruction Visualization

PURPOSE:
- Trains a Variational Autoencoder on MRI brain scans
- Tracks training, validation, and test loss
- Visualizes reconstructed images for qualitative evaluation
- Saves model weights after training
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from load_dataset import train_loader, val_loader, test_loader  # import all loaders
from vae_model import VAE

# -----------------------------
# Device Configuration
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Model Initialization
# -----------------------------
latent_dim = 16
model = VAE(latent_dim=latent_dim).to(device)

# -----------------------------
# VAE Loss Function
# -----------------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# -----------------------------
# Optimizer Setup
# -----------------------------
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Training Parameters
# -----------------------------
epochs = 15

# -----------------------------
# Initialize lists to store losses
# -----------------------------
train_losses = []
val_losses = []

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # -----------------------------
    # Validation Loop
    # -----------------------------
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# -----------------------------
# Plot Epoch vs Loss
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training and Validation Loss per Epoch")
plt.legend()
plt.show()

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("../models", exist_ok=True)
model_path = "../models/vae_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# -----------------------------
# Final Test Evaluation
# -----------------------------
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        recon, mu, logvar = model(batch)
        loss = vae_loss(recon, batch, mu, logvar)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader.dataset)
print(f"Final Test Loss: {avg_test_loss:.4f}")

# -----------------------------
# Final Visualization (after training)
# -----------------------------
with torch.no_grad():
    sample_batch = next(iter(test_loader)).to(device)
    recon_batch, _, _ = model(sample_batch)
    plt.figure(figsize=(8,4))
    for i in range(4):
        # Original
        plt.subplot(2,4,i+1)
        plt.imshow(sample_batch[i].cpu().squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        # Reconstructed
        plt.subplot(2,4,i+5)
        plt.imshow(recon_batch[i].cpu().squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()
