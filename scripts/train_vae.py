import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from load_dataset import train_loader, val_loader  # your DataLoader from commit 1
from vae_model import VAE

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Model
# -----------------------------
latent_dim = 16
model = VAE(latent_dim=latent_dim).to(device)

# -----------------------------
# Loss function
# -----------------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# -----------------------------
# Optimizer
# -----------------------------
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Training loop
# -----------------------------
epochs = 5  # you can increase later

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
    print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}")

# -----------------------------
# Save model
# -----------------------------
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/vae_model.pth")
print("Model saved to ../models/vae_model.pth")
