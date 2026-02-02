"""
VAE training script (LatticeVAE).

This script is intentionally simple and linear:
- load dataset (MNIST / FashionMNIST)
- build DataLoader with deterministic shuffling
- train LatticeVAE
- save checkpoint + basic artifacts (loss curve, recon grid)

The notebook should orchestrate and pass in:
- cfg (merged YAML config)
- run_dir (outputs/.../dataset/exp_name)
"""

from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from src.models.lattice_vae import LatticeVAE, vae_loss_bce_kl
from src.utils.seed import set_seed, seed_worker
from src.utils.io import save_json, save_torch


def _get_torchvision_dataset(name: str, data_root: str, transform):
    name = name.lower()
    if name == "mnist":
        return datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    if name in ("fashion_mnist", "fashionmnist"):
        return datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset name: {name}")


@torch.no_grad()
def _save_recon_grid(model, dataset, device, out_path: Path, n: int = 16):
    """
    Save a small grid of reconstructions for qualitative monitoring.
    """
    model.eval()
    x, _ = next(iter(DataLoader(dataset, batch_size=n, shuffle=False)))
    x = x.to(device)
    recon, _, _ = model(x)

    # Make a 2-row grid: originals on top, reconstructions below
    x = x.detach().cpu()
    recon = recon.detach().cpu()

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


def train_vae(cfg: dict, run_dir: Path):
    """
    Train LatticeVAE using parameters from cfg.

    Expected cfg keys:
      cfg["seed"], cfg["device"], cfg["num_workers"], cfg["pin_memory"], cfg["outputs_root"]
      cfg["dataset"]["name"], cfg["dataset"]["channels"], cfg["dataset"]["image_size"]
      cfg["vae"]["latent_dim"], cfg["vae"]["base_channels"], cfg["vae"]["lr"], cfg["vae"]["batch_size"],
      cfg["vae"]["epochs"], cfg["vae"]["beta_kl"], cfg["vae"]["loss_reduction"]
    """
    seed = int(cfg.get("seed", 123))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Best-effort reproducibility on GPU; strict determinism only on CPU if requested
    deterministic_cpu = bool(cfg.get("deterministic_cpu", False))
    set_seed(seed, device=("cuda" if device.type == "cuda" else "cpu"), deterministic=deterministic_cpu)

    data_root = cfg.get("data_root", "./data")
    ds_name = cfg["dataset"]["name"]
    in_channels = int(cfg["dataset"]["channels"])

    vae_cfg = cfg["vae"]
    latent_dim = int(vae_cfg["latent_dim"])
    base_channels = int(vae_cfg.get("base_channels", 32))
    lr = float(vae_cfg.get("lr", 1e-3))
    batch_size = int(vae_cfg.get("batch_size", 128))
    epochs = int(vae_cfg.get("epochs", 30))
    beta_kl = float(vae_cfg.get("beta_kl", 1.0))
    loss_reduction = str(vae_cfg.get("loss_reduction", "sum"))

    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))

    # Dataset (kept simple: ToTensor() maps to [0,1])
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = _get_torchvision_dataset(ds_name, data_root=data_root, transform=transform)

    # Deterministic DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and (device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    # Model
    model = LatticeVAE(in_channels=in_channels, latent_dim=latent_dim, base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Output paths
    ckpt_path = Path(run_dir) / "checkpoints" / "vae_last.pt"
    fig_recon_path = Path(run_dir) / "figures" / "vae_recon_grid.png"
    fig_loss_path = Path(run_dir) / "figures" / "vae_loss_curve.png"
    meta_path = Path(run_dir) / "artifacts" / "vae_train_meta.json"

    losses = []
    bces = []
    kls = []

    print(f"[VAE] Dataset={ds_name} | device={device} | epochs={epochs} | reduction={loss_reduction}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_kl = 0.0

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x)

            loss, bce, kl = vae_loss_bce_kl(
                recon=recon,
                x=x,
                mu=mu,
                logvar=logvar,
                beta_kl=beta_kl,
                reduction=loss_reduction,
            )

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_bce += float(bce.item())
            total_kl += float(kl.item())

        # For reporting we normalize per-image to keep logs comparable
        n_imgs = len(train_dataset)
        if loss_reduction == "sum":
            epoch_loss = total_loss / n_imgs
            epoch_bce = total_bce / n_imgs
            epoch_kl = total_kl / n_imgs
        else:
            # mean is already per-batch-element (for KL) and global mean (for BCE),
            # so logs are best left as-is (average over batches)
            epoch_loss = total_loss / len(train_loader)
            epoch_bce = total_bce / len(train_loader)
            epoch_kl = total_kl / len(train_loader)

        losses.append(epoch_loss)
        bces.append(epoch_bce)
        kls.append(epoch_kl)

        print(f"[VAE] Epoch {epoch:03d} | loss={epoch_loss:.4f} | bce={epoch_bce:.4f} | kl={epoch_kl:.4f}")

    elapsed = time.time() - t0
    print(f"[VAE] Training done in {elapsed:.1f}s")

    # Save checkpoint
    save_torch(ckpt_path, {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg,
    })

    # Save recon grid
    _save_recon_grid(model, train_dataset, device=device, out_path=fig_recon_path, n=16)

    # Save loss curve
    fig = plt.figure()
    plt.plot(losses, label="total")
    plt.plot(bces, label="bce")
    plt.plot(kls, label="kl")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    fig_loss_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_loss_path, dpi=200)
    plt.close(fig)

    # Save metadata
    meta = {
        "dataset": ds_name,
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "loss_reduction": loss_reduction,
        "beta_kl": beta_kl,
        "final_loss": float(losses[-1]),
        "final_bce": float(bces[-1]),
        "final_kl": float(kls[-1]),
        "checkpoint": str(ckpt_path),
        "recon_grid": str(fig_recon_path),
        "loss_curve": str(fig_loss_path),
        "elapsed_sec": elapsed,
    }
    save_json(meta_path, meta)

    return model, train_dataset
