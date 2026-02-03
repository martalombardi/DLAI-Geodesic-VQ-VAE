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

def generate_images_multinomial_refined(
    run_dir: Path,
    transformer,
    vae_decoder,
    codebook,
    n_samples: int,
    start_token: int,
    grid_res: int,
    device,
    seed: int,
    metric_tag: str,                 # <-- NEW: "riemann" / "euclidean"
    tag: str = "refined",
    temperature: float = 0.8,
    top_p: float = 0.9,
    codebook_scale: float = 1.15
):
    """
    Refined version of `generate_images_multinomial`.

    This function improves the basic multinomial sampling strategy by:
    - Applying Temperature scaling to control the sharpness of the token distribution.
    - Using Top-p (Nucleus) sampling to remove low-probability tokens and
      stabilize generation.
    - Optionally scaling the codebook vectors before decoding to increase
      latent contrast (heuristic inspired by Method A–style refinements).
    """
    run_dir = Path(run_dir)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # 1) Generation (token sampling)
    # ---------------------------
    transformer.eval()
    vae_decoder.eval()

    if isinstance(codebook, torch.Tensor):
        cb = codebook.to(device).float() * codebook_scale
    else:
        cb = torch.from_numpy(codebook).to(device).float() * codebook_scale

    seq_len = grid_res * grid_res

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    tokens = torch.full((n_samples, 1), start_token, dtype=torch.long, device=device)

    for _ in range(seq_len):
        logits = transformer(tokens)[:, -1, :]
        logits = logits / max(temperature, 1e-6)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("inf"))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=g)
        tokens = torch.cat([tokens, next_token], dim=1)

    tokens_L = tokens[:, 1:]

    z_grid = (
        cb[tokens_L]
        .view(n_samples, grid_res, grid_res, -1)
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    imgs = vae_decoder(z_grid)

    imgs_cpu = imgs.detach().cpu()
    tokens_cpu = tokens_L.detach().cpu()

    # ---------------------------
    # 3) Save artifacts (REFINED saving)
    # ---------------------------
    # 3.1 Save PNG grid with parameter-encoded filename
    png_path = save_generated_grid_refined(
        imgs=imgs_cpu,
        out_dir=samples_dir,
        metric_tag=metric_tag,
        seed=seed,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        codebook_scale=codebook_scale,
        tag=tag,
    )

    # 3.2 Save tokens with aligned naming
    tok_path = samples_dir / png_path.name.replace("samples_", "tokens_").replace(".png", ".pt")
    torch.save({"tokens": tokens_cpu}, tok_path)

    # 3.3 Save metadata JSON with aligned naming
    meta_path = samples_dir / png_path.name.replace("samples_", "meta_").replace(".png", ".json")
    meta = {
        "metric_tag": str(metric_tag),
        "tag": str(tag),
        "n_samples": int(n_samples),
        "start_token": int(start_token),
        "grid_res": int(grid_res),
        "seed": int(seed),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "codebook_scale": float(codebook_scale),
        "png": str(png_path),
        "tokens_pt": str(tok_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return imgs_cpu, tokens_cpu

    # ---------------------------
    # 3) Save artifacts (same convention as the base pipeline)
    # ---------------------------
    # 3.1 Save a PNG grid
    save_generated_grid(imgs_cpu, png_path, nrow=int(np.sqrt(n_samples)))

    # 3.2 Save tokens
    torch.save({"tokens": tokens_cpu}, tok_path)

    # 3.3 Save metadata JSON (for reproducibility)
    meta = {
        "n_samples": int(n_samples),
        "start_token": int(start_token),
        "grid_res": int(grid_res),
        "seed": int(seed),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "codebook_scale": float(codebook_scale),
        "png": str(png_path),
        "tokens_pt": str(tok_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return imgs_cpu, tokens_cpu

def save_generated_grid_refined(
    imgs: torch.Tensor,
    out_dir: Path,
    *,
    metric_tag: str,
    seed: int,
    n_samples: int,
    temperature: float,
    top_p: float,
    codebook_scale: float,
    tag: str = "refined",
    cmap: str = "gray",
    dpi: int = 200,
):
    """
    Save a grid image from a batch tensor with filenames that encode
    the sampling configuration (metric + hyperparameters).

    This is a drop-in alternative to `save_generated_grid`, but it handles
    naming internally and returns the generated output path.

    Args:
        imgs: Tensor of shape (N, C, H, W) on CPU (recommended).
        out_dir: Directory where the PNG will be saved.
        metric_tag: Identifier such as "riemann" or "euclidean".
        seed: Sampling seed.
        n_samples: Number of samples in the grid (used for layout).
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        codebook_scale: Multiplicative scaling applied to codebook vectors.
        tag: Optional string label ("refined" by default).
        cmap: Colormap for grayscale images.
        dpi: DPI for saved PNG.

    Returns:
        out_path: Path of the saved PNG file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Format floats to be filesystem-friendly (e.g., 0.90 -> "0p90")
    def _fmt(x: float, nd: int = 2) -> str:
        return f"{float(x):.{nd}f}".replace(".", "p")

    tp_str = _fmt(top_p, nd=2)
    temp_str = _fmt(temperature, nd=2)
    cs_str = _fmt(codebook_scale, nd=2)

    tag = tag or "refined"
    metric_tag = metric_tag or "metric"

    fname = (
        f"samples_{metric_tag}_{tag}_seed{int(seed)}"
        f"_tp{tp_str}_temp{temp_str}_cs{cs_str}.png"
    )
    out_path = out_dir / fname

    # Grid layout: square-ish
    N, C, H, W = imgs.shape
    nrow = int(np.sqrt(n_samples))
    ncol = int(np.ceil(N / nrow))

    fig = plt.figure(figsize=(ncol * 1.5, nrow * 1.5))
    for i in range(N):
        ax = plt.subplot(nrow, ncol, i + 1)
        if C == 1:
            ax.imshow(imgs[i, 0], cmap=cmap)
        else:
            ax.imshow(imgs[i].permute(1, 2, 0))
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return out_path
