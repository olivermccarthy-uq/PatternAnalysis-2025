import os
import torch
import matplotlib.pyplot as plt
from vae_model import VAE

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load trained model
# -----------------------------
latent_dim = 16  # same as used in training
model = VAE(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load("../models/vae_model.pth", map_location=device))
model.eval()

# -----------------------------
# Sample latent vectors
# -----------------------------
num_samples = 16
with torch.no_grad():
    z = torch.randn(num_samples, latent_dim).to(device)
    generated = model.decoder(model.fc_decode(z))

# -----------------------------
# Display generated images
# -----------------------------
plt.figure(figsize=(8,8))
for i in range(num_samples):
    plt.subplot(4,4,i+1)
    plt.imshow(generated[i].squeeze().cpu(), cmap='gray')
    plt.axis('off')
plt.suptitle("Generated MRI Samples from VAE")
plt.show()
