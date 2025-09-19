"""
Variational Autoencoder (VAE) Model Architecture

PURPOSE: This file defines the neural network architecture for a Variational Autoencoder (VAE)
designed to work with MRI brain scan images. The VAE consists of an encoder that compresses
images into a latent space representation, and a decoder that reconstructs images from
latent vectors.

WHY IT'S NEEDED:
- VAE learns a probabilistic representation of the data in a lower-dimensional latent space
- The encoder maps images to mean and variance parameters of a Gaussian distribution
- The decoder generates new images by sampling from the latent space
- This enables both reconstruction and generation of new MRI-like images
- The reparameterization trick allows for differentiable sampling during training
"""

import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    Variational Autoencoder for MRI brain scan images.
    
    This VAE architecture uses convolutional layers for the encoder and transpose
    convolutional layers for the decoder, making it well-suited for image data.
    """
    
    def __init__(self, latent_dim=16):
        """
        Initialize the VAE model.
        
        Args:
            latent_dim (int): Dimension of the latent space representation
        """
        super(VAE, self).__init__()
        
        # -----------------------------
        # Encoder Network
        # -----------------------------
        # The encoder compresses 64x64 grayscale images into latent representations
        # Each conv layer reduces spatial dimensions by half while increasing channels
        self.encoder = nn.Sequential(
            # Input: 1 channel (grayscale), Output: 32 channels
            # Kernel size 4, stride 2, padding 1: 64x64 -> 32x32
            nn.Conv2d(1, 32, 4, 2, 1),  
            nn.ReLU(),
            
            # Input: 32 channels, Output: 64 channels  
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 4, 2, 1), 
            nn.ReLU(),
            
            # Input: 64 channels, Output: 128 channels
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.ReLU(),
            
            # Flatten the 128x8x8 feature map into a vector
            nn.Flatten()
        )
        
        # -----------------------------
        # Latent Space Mapping
        # -----------------------------
        # Map the flattened features to mean and log-variance of Gaussian distribution
        # This is the key difference from regular autoencoders - we learn a distribution
        self.fc_mu = nn.Linear(128*8*8, latent_dim)      # Mean of latent distribution
        self.fc_logvar = nn.Linear(128*8*8, latent_dim) # Log-variance of latent distribution

        # -----------------------------
        # Decoder Network
        # -----------------------------
        # The decoder reconstructs images from latent vectors
        self.fc_decode = nn.Linear(latent_dim, 128*8*8)  # Expand latent vector back to feature map size
        
        self.decoder = nn.Sequential(
            # Reshape the vector back to 128x8x8 feature map
            nn.Unflatten(1, (128, 8, 8)),
            
            # Transpose convolutions to upsample and reduce channels
            # 8x8 -> 16x16, 128 channels -> 64 channels
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  
            nn.ReLU(),
            
            # 16x16 -> 32x32, 64 channels -> 32 channels
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   
            nn.ReLU(),
            
            # 32x32 -> 64x64, 32 channels -> 1 channel (grayscale)
            nn.ConvTranspose2d(32, 1, 4, 2, 1),    
            
            # Tanh activation outputs values in [-1, 1] range (matching our normalization)
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE training.
        
        Instead of sampling directly from N(mu, sigma), we sample from N(0,1) and transform:
        z = mu + sigma * epsilon, where epsilon ~ N(0,1)
        
        This makes the sampling process differentiable, allowing gradients to flow through.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log-variance of the latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        # Convert log-variance to standard deviation
        std = torch.exp(0.5 * logvar)
        
        # Sample random noise from standard normal distribution
        eps = torch.randn_like(std)
        
        # Apply reparameterization: z = mu + std * epsilon
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 1, 64, 64)
            
        Returns:
            tuple: (reconstructed_images, mu, logvar)
                - reconstructed_images: Generated images
                - mu: Mean of latent distribution
                - logvar: Log-variance of latent distribution
        """
        # -----------------------------
        # Encoding: Image -> Latent Distribution
        # -----------------------------
        # Pass through encoder to get feature representation
        enc = self.encoder(x)
        
        # Map features to latent distribution parameters
        mu = self.fc_mu(enc)        # Mean of latent Gaussian
        logvar = self.fc_logvar(enc) # Log-variance of latent Gaussian
        
        # -----------------------------
        # Sampling: Latent Distribution -> Latent Vector
        # -----------------------------
        # Sample from the learned distribution using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # -----------------------------
        # Decoding: Latent Vector -> Reconstructed Image
        # -----------------------------
        # Expand latent vector and pass through decoder
        out = self.decoder(self.fc_decode(z))
        
        return out, mu, logvar

# -----------------------------
# Model Testing and Verification
# -----------------------------
if __name__ == "__main__":
    # Test the model with random input to verify architecture
    model = VAE(latent_dim=16)
    x = torch.randn(2, 1, 64, 64)  # batch of 2 grayscale images
    
    # Forward pass
    recon, mu, logvar = model(x)
    
    # Print shapes to verify everything works correctly
    print("Input shape:", x.shape)
    print("Reconstructed shape:", recon.shape)
    print("Latent mu shape:", mu.shape)
