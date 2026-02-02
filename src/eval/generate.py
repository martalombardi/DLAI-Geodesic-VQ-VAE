"""
Sampling / generation utilities for the Transformer prior.

This module generates discrete token sequences autoregressively (multinomial sampling),
maps tokens to latent vectors via the codebook, and decodes images with the VAE decoder.

Reproducibility:
- Sampling uses a local torch.Generator with a fixed seed.
- This makes the sampled token sequences deterministic given the same model weights.

Note:
- Full bitwise determinism on GPU can still be affected by low-level CUDA libraries.
  In practice, using a local Generator is the correct approach for reproducible sampling.
"""

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


@torch.no_grad()
def generate_images_multinomial(
    transformer,
    vae_decoder,
    codebook,
    n_samples: int = 16,
    start_token: int = 128,
    grid_res: int = 7,
    device=None,
    seed: int = 123,
):
    """
    Generate images by multinomial sampling from an autoregressive Transformer.

    Args:
        transformer: autoregressive model returning logits (B, T, n_codes)
        vae_decoder: VAE decoder mapping (B, D, H, W) -> (B, C, H_img, W_img)
        codebook: (n_codes, latent_dim) numpy array or torch tensor
        n_samples: number of images to generate
        start_token: integer index used as sequence start (usually == n_codes)
        grid_res: grid resolution (sequence length = grid_res * grid_res)
        device: torch.device or None (infer from transformer)
        seed: sampling seed for reproducible multinomial draws

    Returns:
        imgs: (n_samples, C, H, W) tensor on CPU
        tokens: (n_samples, L) tensor on CPU, where L = grid_res*grid_res
    """
    if device is None:
        device = next(transformer.parameters()).device

    transformer.eval()
    vae_decoder.eval()

    # Codebook -> tensor on device
    if isinstance(codebook, torch.Tensor):
        cb = codebook.to(device).float()
    else:
        cb = torch.from_numpy(codebook).to(device).float()

    seq_len = grid_res * grid_res
    n_codes = cb.size(0)

    # Local generator for deterministic multinomial sampling
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Start sequence: (B, 1)
    tokens = torch.full((n_samples, 1), start_token, dtype=torch.long, device=device)

    # Autoregressive sampling
    for _ in range(seq_len):
        logits = transformer(tokens)          # (B, t, n_codes)
        next_logits = logits[:, -1, :]        # (B, n_codes)
        probs = F.softmax(next_logits, dim=-1)

        # Deterministic multinomial draw (given fixed seed + fixed probs)
        next_token = torch.multinomial(probs, num_samples=1, generator=g)  # (B, 1)
        tokens = torch.cat([tokens, next_token], dim=1)

    # Drop START token
    tokens_L = tokens[:, 1:]  # (B, L)

    # Tokens -> latent grid
    z = cb[tokens_L]  # (B, L, D)
    z_grid = z.view(n_samples, grid_res, grid_res, -1).permute(0, 3, 1, 2).contiguous()

    # Decode to images
    imgs = vae_decoder(z_grid)  # (B, C, H, W)
    return imgs.detach().cpu(), tokens_L.detach().cpu()


def save_generated_grid(
    imgs: torch.Tensor,
    out_path: Path,
    nrow: int = 4,
    cmap: str = "gray",
):
    """
    Save a grid image from a batch tensor.
    Assumes imgs shape: (N, C, H, W). For C=1 uses grayscale plotting.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    N, C, H, W = imgs.shape
    ncol = int(np.ceil(N / nrow))

    fig = plt.figure(figsize=(ncol * 1.5, nrow * 1.5))
    for i in range(N):
        ax = plt.subplot(nrow, ncol, i + 1)
        if C == 1:
            ax.imshow(imgs[i, 0], cmap=cmap)
        else:
            # For RGB-like data, move channels last
            ax.imshow(imgs[i].permute(1, 2, 0))
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_and_save(
    run_dir: Path,
    transformer,
    vae_decoder,
    codebook,
    n_samples: int,
    start_token: int,
    grid_res: int,
    device,
    seed: int,
    tag: str = None,
):
    """
    Convenience wrapper: generate samples and save (png + tokens + meta).

    Files saved under:
      run_dir/samples/
        - samples_<tag>_seedXXX.png
        - tokens_<tag>_seedXXX.pt
        - meta_<tag>_seedXXX.json
    """
    run_dir = Path(run_dir)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    tag = tag or "gen"
    png_path = samples_dir / f"samples_{tag}_seed{seed}.png"
    tok_path = samples_dir / f"tokens_{tag}_seed{seed}.pt"
    meta_path = samples_dir / f"meta_{tag}_seed{seed}.json"

    imgs, tokens = generate_images_multinomial(
        transformer=transformer,
        vae_decoder=vae_decoder,
        codebook=codebook,
        n_samples=n_samples,
        start_token=start_token,
        grid_res=grid_res,
        device=device,
        seed=seed,
    )

    save_generated_grid(imgs, png_path, nrow=int(np.sqrt(n_samples)))

    torch.save({"tokens": tokens}, tok_path)

    meta = {
        "n_samples": int(n_samples),
        "start_token": int(start_token),
        "grid_res": int(grid_res),
        "seed": int(seed),
        "png": str(png_path),
        "tokens_pt": str(tok_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return imgs, tokens
