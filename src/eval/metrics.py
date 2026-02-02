"""
Metrics for discrete-token modeling.

This module provides:
1) Token statistics for a tokenized dataset (codes_dataset).
2) Token statistics for generated token sequences.
3) Transformer evaluation perplexity (cross-entropy) on a token dataset.

All functions can also save:
- JSON summaries to run_dir/artifacts/
- Figures to run_dir/figures/
"""

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# -----------------------------
# Helper: empirical token stats
# -----------------------------

def _compute_token_stats(tokens_2d: torch.Tensor, vocab_size: int):
    """
    Compute empirical statistics from a 2D token tensor of shape (N, L).

    Returns a dict with:
      - N_sequences, seq_len, vocab_size
      - used_tokens, unused_tokens, pct_unused
      - token_utilization (percent of vocab used)
      - entropy_bits, entropy_nats
      - perplexity (empirical, from entropy)
      - effective_vocab_ratio = empirical_perplexity / vocab_size
      - per_position_used_tokens: (L,) number of unique tokens per position
      - per_position_perplexity: (L,) empirical perplexity per position
    """
    if tokens_2d.dtype != torch.long:
        tokens_2d = tokens_2d.long()
    tokens_2d = tokens_2d.detach().cpu()

    N, L = tokens_2d.shape

    # Global histogram
    flat = tokens_2d.reshape(-1).numpy()
    counts = np.bincount(flat, minlength=vocab_size).astype(np.float64)
    total = counts.sum()
    probs = counts / max(total, 1.0)

    used = int((counts > 0).sum())
    unused = int(vocab_size - used)
    pct_unused = float(unused / vocab_size)

    # Entropy
    p_nonzero = probs[probs > 0]
    entropy_nats = float(-(p_nonzero * np.log(p_nonzero)).sum())
    entropy_bits = float(entropy_nats / np.log(2.0))
    ppl_emp = float(np.exp(entropy_nats))  # empirical perplexity

    eff_vocab_ratio = float(ppl_emp / vocab_size)

    # Per-position stats
    per_pos_used = np.zeros(L, dtype=np.int64)
    per_pos_ppl = np.zeros(L, dtype=np.float64)

    for j in range(L):
        col = tokens_2d[:, j].numpy()
        c = np.bincount(col, minlength=vocab_size).astype(np.float64)
        per_pos_used[j] = int((c > 0).sum())
        p = c / max(c.sum(), 1.0)
        p_nz = p[p > 0]
        h_nats = float(-(p_nz * np.log(p_nz)).sum())
        per_pos_ppl[j] = float(np.exp(h_nats))

    return {
        "N_sequences": int(N),
        "seq_len": int(L),
        "vocab_size": int(vocab_size),
        "used_tokens": used,
        "unused_tokens": unused,
        "pct_unused": float(pct_unused),
        "token_utilization": float(100.0 * used / vocab_size),
        "entropy_nats": entropy_nats,
        "entropy_bits": entropy_bits,
        "perplexity": ppl_emp,
        "effective_vocab_ratio": eff_vocab_ratio,
        "per_position_used_tokens": per_pos_used.tolist(),
        "per_position_perplexity": per_pos_ppl.tolist(),
        "per_position_summary": {
            "used_tokens_min": int(per_pos_used.min()),
            "used_tokens_mean": float(per_pos_used.mean()),
            "used_tokens_max": int(per_pos_used.max()),
            "ppl_min": float(per_pos_ppl.min()),
            "ppl_mean": float(per_pos_ppl.mean()),
            "ppl_max": float(per_pos_ppl.max()),
        }
    }


def _save_json(path: Path, obj: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------
# 1) Dataset token utilization & entropy
# --------------------------------------

def evaluate_tokenized_dataset(
    run_dir: Path,
    codes_dataset: torch.Tensor,
    vocab_size: int,
    tag: str = "dataset",
    save: bool = True,
):
    """
    Compute empirical token utilization + entropy stats on the tokenized dataset.

    Saves:
      - artifacts/token_stats_<tag>.json
      - figures/token_stats_<tag>_per_position.png
    """
    stats = _compute_token_stats(codes_dataset, vocab_size=vocab_size)

    if save:
        run_dir = Path(run_dir)
        _save_json(run_dir / "artifacts" / f"token_stats_{tag}.json", stats)

        # Plot per-position perplexity & used tokens
        L = stats["seq_len"]
        xs = np.arange(L)

        fig = plt.figure()
        plt.plot(xs, stats["per_position_perplexity"])
        plt.xlabel("position")
        plt.ylabel("empirical perplexity")
        plt.tight_layout()
        out = run_dir / "figures" / f"token_stats_{tag}_per_position_ppl.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        plt.close(fig)

        fig = plt.figure()
        plt.plot(xs, stats["per_position_used_tokens"])
        plt.xlabel("position")
        plt.ylabel("unique tokens used")
        plt.tight_layout()
        out = run_dir / "figures" / f"token_stats_{tag}_per_position_used.png"
        plt.savefig(out, dpi=200)
        plt.close(fig)

    return stats


# ---------------------------------------
# 2) Generation token utilization & entropy
# ---------------------------------------

def evaluate_generated_tokens(
    run_dir: Path,
    generated_tokens: torch.Tensor,
    vocab_size: int,
    tag: str = "generation",
    save: bool = True,
):
    """
    Compute empirical token utilization + entropy stats on generated token sequences.

    Same outputs as dataset stats, but tagged differently.
    """
    return evaluate_tokenized_dataset(
        run_dir=run_dir,
        codes_dataset=generated_tokens,
        vocab_size=vocab_size,
        tag=tag,
        save=save,
    )


# ---------------------------------------
# 3) Transformer perplexity (cross-entropy)
# ---------------------------------------

@torch.no_grad()
def evaluate_transformer_perplexity(
    run_dir: Path,
    transformer: nn.Module,
    codes_dataset: torch.Tensor,
    n_codes: int,
    start_token: int,
    batch_size: int = 256,
    device=None,
    tag: str = "transformer_eval",
    save: bool = True,
):
    """
    Evaluate transformer cross-entropy perplexity on a tokenized dataset.

    Perplexity here is exp(average cross-entropy).
    This is different from the *empirical* perplexity computed from token frequencies.

    Saves:
      - artifacts/transformer_ppl_<tag>.json
      - figures/transformer_nll_hist_<tag>.png (optional histogram of per-batch losses)
    """
    if device is None:
        device = next(transformer.parameters()).device

    transformer.eval()

    # Build (inputs, targets)
    N, L = codes_dataset.shape
    inputs = torch.cat(
        [torch.full((N, 1), start_token, dtype=torch.long), codes_dataset[:, :-1]],
        dim=1
    )
    targets = codes_dataset
    ds = TensorDataset(inputs, targets)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss(reduction="mean")

    losses = []
    total_tokens = 0
    total_nll = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = transformer(x)  # (B, L, n_codes)
        loss = criterion(logits.reshape(-1, n_codes), y.reshape(-1))  # mean over tokens
        losses.append(float(loss.item()))

        B = y.size(0)
        total_tokens += int(B * L)
        total_nll += float(loss.item()) * (B * L)

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = float(np.exp(avg_nll))

    out = {
        "N_sequences": int(N),
        "seq_len": int(L),
        "n_codes": int(n_codes),
        "start_token": int(start_token),
        "batch_size": int(batch_size),
        "avg_nll": float(avg_nll),
        "perplexity": float(ppl),
        "batch_loss_mean": float(np.mean(losses)),
        "batch_loss_std": float(np.std(losses)),
    }

    if save:
        run_dir = Path(run_dir)
        _save_json(run_dir / "artifacts" / f"transformer_ppl_{tag}.json", out)

        # Histogram of batch losses (rough stability diagnostic)
        fig = plt.figure()
        plt.hist(losses, bins=30)
        plt.xlabel("batch cross-entropy")
        plt.ylabel("count")
        plt.tight_layout()
        out_fig = run_dir / "figures" / f"transformer_nll_hist_{tag}.png"
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_fig, dpi=200)
        plt.close(fig)

    return out
