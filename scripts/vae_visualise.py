"""
VAE Visualization and Generation Script

PURPOSE: This script loads a trained VAE model and generates new MRI-like images by sampling
from the learned latent space. It demonstrates the generative capabilities of the VAE by
creating synthetic brain scan images that follow the patterns learned during training.

WHY IT'S NEEDED:
- Demonstrates the generative power of the trained VAE model
- Shows how the model learned to map random noise to realistic MRI images
- Provides visual verification that training was successful
- Generates new synthetic data that could be used for data augmentation
- Illustrates the continuous nature of the learned latent space
"""

import os
import torch
import matplotlib.pyplot as plt
from vae_model import VAE  # Import our VAE model architecture

# -----------------------------
# Device Configuration
# -----------------------------
# Use the same device configuration as training (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Model Loading
# -----------------------------
# Initialize the VAE model with the same architecture as training
latent_dim = 16  # Must match the latent dimension used during training
model = VAE(latent_dim=latent_dim).to(device)

# Load the trained model weights from the saved checkpoint
model_path = "../models/vae_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded model from {model_path}")
else:
    print(f"Error: Model file not found at {model_path}")
    print("Please run train_vae.py first to train and save the model.")
    exit(1)

# Set model to evaluation mode (disables dropout, batch norm uses running stats)
model.eval()

# -----------------------------
# Image Generation
# -----------------------------
# Number of images to generate
num_samples = 16

print(f"Generating {num_samples} new MRI images...")

# Generate new images by sampling from the latent space
with torch.no_grad():  # Disable gradient computation for inference
    # Sample random latent vectors from standard normal distribution N(0,1)
    # This is the key insight: we can generate new images by sampling from the
    # learned latent space distribution
    z = torch.randn(num_samples, latent_dim).to(device)
    
    # Pass the random latent vectors through the decoder to generate images
    # We bypass the encoder and directly use the decoder part of the VAE
    generated = model.decoder(model.fc_decode(z))

print("Image generation completed!")

# -----------------------------
# Visualization
# -----------------------------
# Create a grid layout to display all generated images
plt.figure(figsize=(8, 8))
plt.suptitle("Generated MRI Samples from VAE", fontsize=16, fontweight='bold')

# Display each generated image in a 4x4 grid
for i in range(num_samples):
    plt.subplot(4, 4, i + 1)
    
    # Convert tensor to numpy and remove channel dimension for display
    # .squeeze() removes the channel dimension: (1, 64, 64) -> (64, 64)
    # .cpu() moves tensor from GPU to CPU for matplotlib
    img = generated[i].squeeze().cpu()
    
    # Display the image in grayscale
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Remove axes for cleaner display

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()

print("Visualization completed!")
print("\nWhat you're seeing:")
print("- Each image is generated from a random 16-dimensional vector")
print("- The VAE learned to map these random vectors to realistic MRI-like images")
print("- The diversity shows the model learned a rich representation of brain anatomy")
print("- These are completely synthetic images, not reconstructions of training data")
