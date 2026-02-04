"""
Training script for the Lattice VQ-VAE model.

This script:
- loads a torchvision dataset (MNIST or Fashion-MNIST),
- trains a Lattice VQ-VAE with a discrete codebook,
- logs reconstruction, VQ loss, and perplexity,
- saves checkpoints, figures, and metadata for reproducibility.
"""

from pathlib import Path
import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from src.models.lattice_vqvae import LatticeVQVAE, vqvae_loss_bce
from src.utils.seed import set_seed, seed_worker
from src.utils.io import save_json, save_torch


# ---------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------

def get_torchvision_dataset(name: str, data_root: str, transform):
    """
    Load a torchvision dataset given its name.

    Supported datasets:
    - MNIST
    - Fashion-MNIST
    """
    name = name.lower()

    if name == "mnist":
        return datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform
        )

    if name in ("fashion_mnist", "fashionmnist"):
        return datasets.FashionMNIST(
            root=data_root, train=True, download=True, transform=transform
        )

    raise ValueError(f"Unsupported dataset name: {name}")


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

@torch.no_grad()
def save_reconstruction_grid(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    out_path: Path,
    n: int = 16,
) -> None:
    """
    Save a grid of original images (top row) and reconstructions (bottom row).
    """
    model.eval()

    loader = DataLoader(dataset, batch_size=n, shuffle=False)
    x, _ = next(iter(loader))
    x = x.to(device)

    recon, _, _, _ = model(x)

    x = x.cpu()
    recon = recon.cpu()

    fig = plt.figure(figsize=(6, 3))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        ax.imshow(x[i, 0], cmap="gray")
        ax.axis("off")

        ax = plt.subplot(2, n, n + i + 1)
        ax.imshow(recon[i, 0], cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

def train_vqvae(cfg: dict, run_dir: Path) -> Tuple[torch.nn.Module, object]:
    """
    Train a Lattice VQ-VAE model.

    Configuration usage:
    - cfg["dataset"]: dataset parameters
    - cfg["vq_vae"]: architecture and training hyperparameters
    - cfg["codebook"]["n_codes"]: codebook size

    Optional:
    - cfg["vqvae_beta_commit"]: commitment loss coefficient (default: 0.25)
    - cfg["vqvae_weight"]: global weight for VQ loss (default: 1.0)
    """
    # -----------------------------------------------------------------
    # Reproducibility and device
    # -----------------------------------------------------------------
    seed = int(cfg.get("seed", 123))
    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    set_seed(
        seed,
        device=("cuda" if device.type == "cuda" else "cpu"),
        deterministic=bool(cfg.get("deterministic_cpu", False)),
    )

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------
    data_root = cfg.get("data_root", "./data")
    ds_name = cfg["dataset"]["name"]
    in_channels = int(cfg["dataset"]["channels"])

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = get_torchvision_dataset(
        ds_name, data_root=data_root, transform=transform
    )

    batch_size = int(cfg["vae"].get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=g,
    )

    # -----------------------------------------------------------------
    # Model and optimizer
    # -----------------------------------------------------------------
    vae_cfg = cfg["vq_vae"]

    batch_size = int(vae_cfg.get("batch_size", 128))

    model = LatticeVQVAE(
        in_channels=in_channels,
        latent_dim=int(vae_cfg["latent_dim"]),
        n_codes=int(cfg["codebook"]["n_codes"]),
        base_channels=int(vae_cfg.get("base_channels", 32)),
        beta_commit=float(vae_cfg.get("beta_commit", 0.25)),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(vae_cfg.get("lr", 1e-3)),
    )

    epochs = int(vae_cfg.get("epochs", 30))
    loss_reduction = str(vae_cfg.get("loss_reduction", "sum"))
    vq_weight = float(vae_cfg.get("vq_weight", 1.0))

    # -----------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------
    ckpt_path = run_dir / "checkpoints" / "vqvae_last.pt"
    recon_fig_path = run_dir / "figures" / "vqvae_recon_grid.png"
    loss_fig_path = run_dir / "figures" / "vqvae_loss_curve.png"
    meta_path = run_dir / "artifacts" / "vqvae_train_meta.json"

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    losses, recon_losses, vq_losses, perplexities = [], [], [], []

    print(
        f"[VQVAE] Dataset={ds_name} | device={device} | epochs={epochs} "
        f"| n_codes={cfg['codebook']['n_codes']}"
    )

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0
        total_ppl = 0.0

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            recon, vq_loss, _, perplexity = model(x)

            loss, recon_term, vq_term = vqvae_loss_bce(
                recon=recon,
                x=x,
                vq_loss=vq_loss,
                loss_reduction=loss_reduction,
                vq_weight=vq_weight,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_term.item()
            total_vq += vq_term.item()
            total_ppl += perplexity.item()

        if loss_reduction == "sum":
            epoch_loss = total_loss / len(train_dataset)
            epoch_recon = total_recon / len(train_dataset)
        else:
            epoch_loss = total_loss / len(train_loader)
            epoch_recon = total_recon / len(train_loader)

        epoch_vq = total_vq / len(train_loader)
        epoch_ppl = total_ppl / len(train_loader)

        losses.append(epoch_loss)
        recon_losses.append(epoch_recon)
        vq_losses.append(epoch_vq)
        perplexities.append(epoch_ppl)

        print(
            f"[VQVAE] Epoch {epoch:03d} | "
            f"loss={epoch_loss:.4f} | "
            f"recon={epoch_recon:.4f} | "
            f"vq={epoch_vq:.4f} | "
            f"ppl={epoch_ppl:.2f}"
        )

    elapsed = time.time() - t_start
    print(f"[VQVAE] Training completed in {elapsed:.1f}s")

    # -----------------------------------------------------------------
    # Saving artifacts
    # -----------------------------------------------------------------
    save_torch(
        ckpt_path,
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg,
        },
    )

    save_reconstruction_grid(
        model, train_dataset, device=device, out_path=recon_fig_path
    )

    fig = plt.figure()
    plt.plot(losses, label="total")
    plt.plot(recon_losses, label="reconstruction")
    plt.plot(vq_losses, label="vq")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    loss_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(loss_fig_path, dpi=200)
    plt.close(fig)

    meta = {
        "dataset": ds_name,
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "n_codes": cfg["codebook"]["n_codes"],
        "beta_commit": float(vae_cfg.get("beta_commit", 0.25)),
        "vq_weight": float(vae_cfg.get("vq_weight", 1.0)),
        "final_loss": losses[-1],
        "final_recon": recon_losses[-1],
        "final_vq": vq_losses[-1],
        "final_perplexity": perplexities[-1],
        "elapsed_sec": elapsed,
    }

    save_json(meta_path, meta)

    return model, train_dataset
