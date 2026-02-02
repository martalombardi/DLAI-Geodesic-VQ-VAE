"""
LatticeVAE

This module defines a VAE whose latent representation is a 2D lattice:
    z in R^{latent_dim * grid_res * grid_res}

This design matches image structure and makes tokenization into a sequence of grid tokens natural

Two strided conv downsamples by a factor 4 overall
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGridEncoder(nn.Module):
    """
    Convolutional encoder producing mean/log-variance as feature maps:
        mu, logvar: (B, latent_dim, H', W')
    """
    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # Downsample ×2 twice: (img → img/2 → img/4), then map to 2*latent_dim channels.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 2 * c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 2 * latent_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor):
        h = self.net(x)  # (B, 2*latent_dim, H', W')
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


class ConvGridDecoder(nn.Module):
    """
    Convolutional decoder mapping the latent lattice to an image.
    Input:
        z_grid: (B, latent_dim, H', W')
    Output:
        recon: (B, out_channels, H, W)
    """
    def __init__(self, out_channels: int, latent_dim: int, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # Upsample ×2 twice: (grid → 2*grid → 4*grid)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 2 * c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * c, c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Suitable for inputs normalized to [0, 1] and BCE loss
        )

    def forward(self, z_grid: torch.Tensor):
        return self.net(z_grid)


class LatticeVAE(nn.Module):
    """
    Variational Autoencoder with a 2D latent lattice (grid).

    Forward returns:
        recon, mu, logvar

    where:
        mu and logvar have shape (B, latent_dim, H', W').
    """
    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvGridEncoder(in_channels, latent_dim, base_channels)
        self.decoder = ConvGridDecoder(in_channels, latent_dim, base_channels)

    def forward(self, x: torch.Tensor):
        # Encode input into mean and log-variance maps
        mu, logvar = self.encoder(x)

        # Reparameterization trick (inlined)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Decode latent lattice into reconstructed image
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss_bce_kl(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float = 1.0,
    reduction: str = "sum",   # "sum" or "mean"
):
    """
    VAE objective: BCE(recon, x) + beta * KL(q(z|x) || p(z)).

    reduction:
      - "sum": sums over all elements in the batch (classic VAE implementation).
      - "mean": averages per batch element (more stable across batch sizes/datasets).

    Important:
      We keep BCE and KL on a consistent scale:
      - "sum": BCE is summed, KL is summed
      - "mean": BCE is averaged over batch, KL is averaged over batch
    """
    if reduction not in ("sum", "mean"):
        raise ValueError("reduction must be 'sum' or 'mean'")

    # Reconstruction term
    bce = F.binary_cross_entropy(recon, x, reduction=reduction)

    # KL term (compute per-sample, then reduce consistently)
    # kl_per_sample shape: (B,)
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3))

    if reduction == "sum":
        kl = kl_per_sample.sum()
    else:  # "mean"
        kl = kl_per_sample.mean()

    loss = bce + beta_kl * kl
    return loss, bce.detach(), kl.detach()