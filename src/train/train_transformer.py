"""
Transformer training for discrete latent codes.

This module trains an autoregressive Transformer over sequences of discrete codes
obtained from a VAE latent lattice.

Outputs saved under:
  run_dir/checkpoints/
  run_dir/figures/
  run_dir/artifacts/

Expected workflow (from notebook):
  1) Load codebook artifacts: landmarks, bridge_to_medoids
  2) Build tokenizer = LandmarkToMedoidTokenizer(landmarks, bridge_to_medoids)
  3) Quantize dataset -> codes_dataset (N, L)
  4) Train transformer on (inputs, targets)
"""

from pathlib import Path
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from src.models.transformer import GenerativeTransformer2D
from src.utils.seed import set_seed, seed_worker
from src.utils.io import save_json, save_torch


def _make_autoregressive_dataset(codes_dataset: torch.Tensor, start_token: int):
    """
    codes_dataset: (N, L) integer codes in [0, n_codes-1]
    inputs:  (N, L) = [START, t1, ..., t_{L-1}]
    targets: (N, L) = [t1, ..., t_L]
    """
    N, L = codes_dataset.shape
    inputs = torch.cat(
        [torch.full((N, 1), start_token, dtype=torch.long), codes_dataset[:, :-1]],
        dim=1,
    )
    targets = codes_dataset
    return inputs, targets


def train_transformer(
    cfg: dict,
    run_dir: Path,
    codes_dataset: torch.Tensor,
):
    """
    Train the autoregressive Transformer.

    Expected cfg keys:
      cfg["seed"], cfg["device"], cfg["num_workers"], cfg["pin_memory"], cfg["deterministic_cpu"]
      cfg["dataset"]["name"]
      cfg["transformer"] dict with:
        - n_codes
        - start_token
        - grid_res
        - d_model
        - n_heads
        - n_layers
        - dropout
        - batch_size
        - epochs
        - lr
        - weight_decay
        - grad_clip
    """
    tr_cfg = cfg["transformer"]
    seed = int(cfg.get("seed", 123))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    deterministic_cpu = bool(cfg.get("deterministic_cpu", False))

    # Best-effort reproducibility on GPU; strict determinism only on CPU if requested
    set_seed(seed, device=("cuda" if device.type == "cuda" else "cpu"), deterministic=deterministic_cpu)

    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False)) and (device.type == "cuda")

    n_codes = int(tr_cfg["n_codes"])
    start_token = int(tr_cfg.get("start_token", n_codes))
    grid_res = int(tr_cfg.get("grid_res", 7))
    d_model = int(tr_cfg.get("d_model", 256))
    n_heads = int(tr_cfg.get("n_heads", 8))
    n_layers = int(tr_cfg.get("n_layers", 4))
    dropout = float(tr_cfg.get("dropout", 0.1))
    batch_size = int(tr_cfg.get("batch_size", 128))
    epochs = int(tr_cfg.get("epochs", 30))
    lr = float(tr_cfg.get("lr", 1e-4))
    weight_decay = float(tr_cfg.get("weight_decay", 0.01))
    grad_clip = float(tr_cfg.get("grad_clip", 1.0))

    # Sanity checks
    assert codes_dataset.dtype == torch.long, "codes_dataset must be torch.long"
    assert codes_dataset.dim() == 2, "codes_dataset must have shape (N, L)"
    assert codes_dataset.max().item() < n_codes, "codes_dataset contains tokens >= n_codes"
    assert codes_dataset.min().item() >= 0, "codes_dataset contains negative tokens"

    # Save codes_dataset for reproducibility (so you don't need to re-quantize)
    art_dir = Path(run_dir) / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    codes_path = art_dir / "codes_dataset.pt"
    save_torch(codes_path, {"codes_dataset": codes_dataset.cpu()})

    # Build autoregressive dataset
    inputs, targets = _make_autoregressive_dataset(codes_dataset, start_token=start_token)
    ds = TensorDataset(inputs, targets)

    # Deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    # Model
    transformer = GenerativeTransformer2D(
        n_codes=n_codes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        grid_res=grid_res,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(transformer.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Output paths
    ckpt_path = Path(run_dir) / "checkpoints" / "transformer_last.pt"
    fig_path = Path(run_dir) / "figures" / "transformer_ppl_curve.png"
    meta_path = Path(run_dir) / "artifacts" / "transformer_train_meta.json"

    ppl_history = []
    loss_history = []

    print(f"[TR] Training Transformer | device={device} | epochs={epochs} | n_codes={n_codes} | L={codes_dataset.shape[1]}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        transformer.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = transformer(x)  # (B, L, n_codes)
            loss = criterion(logits.reshape(-1, n_codes), y.reshape(-1))

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / len(train_loader)
        ppl = float(torch.exp(torch.tensor(avg_loss)).item())

        loss_history.append(avg_loss)
        ppl_history.append(ppl)

        print(f"[TR] Epoch {epoch:03d} | loss={avg_loss:.4f} | ppl={ppl:.2f}")

    elapsed = time.time() - t0
    print(f"[TR] Training done in {elapsed:.1f}s")

    # Save checkpoint
    save_torch(ckpt_path, {
        "model_state": transformer.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg,
    })

    # Save perplexity curve
    fig = plt.figure()
    plt.plot(ppl_history)
    plt.xlabel("epoch")
    plt.ylabel("perplexity")
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Save metadata
    meta = {
        "dataset": cfg["dataset"]["name"],
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "n_codes": n_codes,
        "start_token": start_token,
        "grid_res": grid_res,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "final_loss": float(loss_history[-1]),
        "final_ppl": float(ppl_history[-1]),
        "loss_curve": str(fig_path),
        "checkpoint": str(ckpt_path),
        "codes_dataset": str(codes_path),
        "elapsed_sec": elapsed,
    }
    save_json(meta_path, meta)

    return transformer
