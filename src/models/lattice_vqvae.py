"""
Lattice VQ-VAE (Vector-Quantized Variational Autoencoder) for image datasets like MNIST/Fashion-MNIST.

This module implements:
- A convolutional encoder that maps an image to a latent grid (B, D, H, W)
- A vector quantizer (VQ) with straight-through gradients
- A convolutional decoder that reconstructs the image from the quantized latent grid
- A simple reconstruction + VQ loss helper (BCE by default)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------

class ConvGridEncoder(nn.Module):
    """
    Convolutional encoder that outputs a latent *grid* (B, latent_dim, H, W).

    With the default two stride-2 convolutions:
      - 28x28 -> 14x14 -> 7x7
    """
    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c, 2 * c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(2 * c, latent_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvGridDecoder(nn.Module):
    """
    Convolutional decoder that maps a latent grid (B, latent_dim, H, W)
    back to an image (B, out_channels, H_img, W_img).
    """
    def __init__(self, out_channels: int, latent_dim: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 2 * c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(2 * c, c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # assume inputs are normalized to [0,1]
        )

    def forward(self, z_grid: torch.Tensor) -> torch.Tensor:
        return self.net(z_grid)


# ---------------------------------------------------------------------
# Vector Quantizer
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class VQOutput:
    """Structured output for the VQ layer."""
    z_q: torch.Tensor                 # (B, D, H, W) quantized latent, straight-through
    vq_loss: torch.Tensor             # scalar
    indices: torch.Tensor             # (B, H, W) code indices
    perplexity: torch.Tensor          # scalar
    usage_probs: torch.Tensor         # (K,) average usage probability over the batch


class VectorQuantizer(nn.Module):
    """
    Standard VQ layer with straight-through estimator (VQ-VAE v1 style).

    Parameters
    ----------
    n_codes:
        Vocabulary size K (number of codebook vectors).
    code_dim:
        Embedding dimensionality D (must match encoder latent_dim).
    beta_commit:
        Weight of the commitment term.
    init_scale:
        Initialization range for codebook weights (uniform in [-init_scale, init_scale]).
    """
    def __init__(
        self,
        n_codes: int,
        code_dim: int,
        beta_commit: float = 0.25,
        init_scale: float | None = None,
    ) -> None:
        super().__init__()
        if n_codes <= 0:
            raise ValueError("n_codes must be > 0")
        if code_dim <= 0:
            raise ValueError("code_dim must be > 0")

        self.n_codes = int(n_codes)
        self.code_dim = int(code_dim)
        self.beta_commit = float(beta_commit)

        self.codebook = nn.Embedding(self.n_codes, self.code_dim)

        # A simple, stable default init
        if init_scale is None:
            init_scale = 1.0 / self.n_codes
        nn.init.uniform_(self.codebook.weight, -init_scale, init_scale)

    @staticmethod
    def _flatten_latents(z_e: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Convert z_e (B, D, H, W) -> z (N, D), where N = B*H*W.
        """
        if z_e.dim() != 4:
            raise ValueError(f"Expected z_e to be 4D (B,D,H,W), got shape {tuple(z_e.shape)}")

        B, D, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)  # (N, D)
        return z, B, D, H, W

    def forward(self, z_e: torch.Tensor) -> VQOutput:
        """
        Quantize encoder latents.

        Inputs
        ------
        z_e: torch.Tensor
            Encoder output with shape (B, D, H, W).

        Returns
        -------
        VQOutput
            Contains straight-through quantized latents, losses, indices and perplexity.
        """
        z, B, D, H, W = self._flatten_latents(z_e)            # (N, D)
        e = self.codebook.weight                              # (K, D)

        # Squared L2 distance: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        z_sq = torch.sum(z ** 2, dim=1, keepdim=True)         # (N, 1)
        e_sq = torch.sum(e ** 2, dim=1).unsqueeze(0)          # (1, K)
        ze = 2.0 * (z @ e.t())                                # (N, K)
        dist = z_sq + e_sq - ze                               # (N, K)

        indices_flat = torch.argmin(dist, dim=1)              # (N,)
        z_q_flat = self.codebook(indices_flat)                # (N, D)

        # Reshape back to grid
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)

        # Losses (VQ-VAE v1)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta_commit * commitment_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Usage stats + perplexity
        one_hot = F.one_hot(indices_flat, num_classes=self.n_codes).float()  # (N, K)
        usage_probs = one_hot.mean(dim=0)                                    # (K,)
        perplexity = torch.exp(-torch.sum(usage_probs * torch.log(usage_probs + 1e-10)))

        indices_grid = indices_flat.view(B, H, W)

        return VQOutput(
            z_q=z_q_st,
            vq_loss=vq_loss,
            indices=indices_grid,
            perplexity=perplexity,
            usage_probs=usage_probs,
        )


# ---------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------

class LatticeVQVAE(nn.Module):
    """
    Lattice VQ-VAE: Encoder -> VectorQuantizer -> Decoder.

    Returns reconstructions plus VQ diagnostics useful for:
    - monitoring token usage / perplexity
    - training an autoregressive prior over indices
    """
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        n_codes: int,
        base_channels: int = 32,
        beta_commit: float = 0.25,
    ) -> None:
        super().__init__()

        self.encoder = ConvGridEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
        )
        self.vq = VectorQuantizer(
            n_codes=n_codes,
            code_dim=latent_dim,
            beta_commit=beta_commit,
        )
        self.decoder = ConvGridDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x: (B, C, H, W)
            Input image tensor.

        Returns
        -------
        recon: (B, C, H, W)
        vq_loss: scalar
        indices: (B, H_lat, W_lat)
        perplexity: scalar
        """
        z_e = self.encoder(x)
        vq_out = self.vq(z_e)
        recon = self.decoder(vq_out.z_q)
        return recon, vq_out.vq_loss, vq_out.indices, vq_out.perplexity


# ---------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------

def vqvae_loss_bce(
    recon: torch.Tensor,
    x: torch.Tensor,
    vq_loss: torch.Tensor,
    loss_reduction: str = "sum",
    vq_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BCE reconstruction loss + weighted VQ loss.

    Parameters
    ----------
    recon, x:
        Tensors of shape (B, C, H, W). Inputs assumed in [0, 1].
    vq_loss:
        Scalar tensor produced by VectorQuantizer.
    loss_reduction:
        "sum" or "mean". Keep consistent with your training config and logging.
    vq_weight:
        Multiplier for the VQ loss term.

    Returns
    -------
    total_loss, recon_loss_detached, vq_loss_detached
    """
    if loss_reduction not in ("sum", "mean"):
        raise ValueError("loss_reduction must be 'sum' or 'mean'")

    recon_loss = F.binary_cross_entropy(recon, x, reduction=loss_reduction)
    total = recon_loss + vq_weight * vq_loss
    return total, recon_loss.detach(), vq_loss.detach()
