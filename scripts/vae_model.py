import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128*8*8, latent_dim)
        self.fc_logvar = nn.Linear(128*8*8, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 128*8*8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),    # 32->64
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        mu, logvar = self.fc_mu(enc), self.fc_logvar(enc)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(self.fc_decode(z))
        return out, mu, logvar

# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    model = VAE(latent_dim=16)
    x = torch.randn(2, 1, 64, 64)  # batch of 2 grayscale images
    recon, mu, logvar = model(x)
    print("Input shape:", x.shape)
    print("Reconstructed shape:", recon.shape)
    print("Latent mu shape:", mu.shape)
