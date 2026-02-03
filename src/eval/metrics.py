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

"""
Metrics for discrete-token modeling.

This module provides:
1) Empirical token statistics for a tokenized dataset (codes_dataset).
2) Empirical token statistics for generated token sequences.
3) Transformer evaluation perplexity (cross-entropy) on a token dataset.

Enhancements in this version:
- Adds global plots for:
  (a) used vs. unused tokens
  (b) token utilization (% of vocabulary used)
- Uses consistent, explicit file naming based on:
  metric_tag + sampler_tag + seed (+ refined hyperparams) + user tag
- Distinguishes BASE multinomial vs REFINED multinomial in saved artifacts/figures.

Outputs:
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
# Helper: stable tag formatting
# -----------------------------

def _fmt_float(x: float, nd: int = 2) -> str:
    """Filesystem-friendly float formatting: 0.90 -> '0p90'."""
    return f"{float(x):.{nd}f}".replace(".", "p")


def _build_run_id(
    *,
    metric_tag: str,
    sampler_tag: str,
    seed: int,
    user_tag: str,
    temperature: float = None,
    top_p: float = None,
    codebook_scale: float = None,
) -> str:
    """
    Build a unique identifier string used in filenames.

    Examples:
      dataset_riemann_base_seed42
      generation_euclidean_refined_seed123_tp0p90_temp0p80_cs1p15
    """
    metric_tag = metric_tag or "metric"
    sampler_tag = sampler_tag or "base"
    user_tag = user_tag or "run"

    rid = f"{user_tag}_{metric_tag}_{sampler_tag}_seed{int(seed)}"

    # Only append refined hyperparameters if they are provided (or if sampler_tag implies refined)
    if temperature is not None:
        rid += f"_temp{_fmt_float(temperature, 2)}"
    if top_p is not None:
        rid += f"_tp{_fmt_float(top_p, 2)}"
    if codebook_scale is not None:
        rid += f"_cs{_fmt_float(codebook_scale, 2)}"

    return rid


def _save_json(path: Path, obj: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Helper: empirical token stats
# -----------------------------

def _compute_token_stats(tokens_2d: torch.Tensor, vocab_size: int):
    """
    Compute empirical statistics from a 2D token tensor of shape (N, L).

    Returns a dict with:
      - N_sequences, seq_len, vocab_size
      - used_tokens, unused_tokens, pct_unused
      - token_utilization (% of vocab used)
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

    # Entropy (ignore zeros)
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
    
    counts = np.bincount(flat, minlength=vocab_size).astype(np.float64)
  
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
        "token_counts": counts.tolist(),
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


# --------------------------------------
# Plot helpers (NEW in this version)
# --------------------------------------

def _plot_global_token_usage(stats: dict, out_path: Path):
    """
    Plot used vs unused tokens as a simple bar chart.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    used = stats["used_tokens"]
    unused = stats["unused_tokens"]

    fig = plt.figure()
    plt.bar(["used", "unused"], [used, unused])
    plt.ylabel("# tokens")
    plt.title("Vocabulary usage (global)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_token_utilization(stats: dict, out_path: Path):
    """
    Plot token utilization as a histogram of token usage frequencies.

    Unlike a single scalar (% of vocabulary used), this plot shows the
    distribution of how frequently tokens are sampled/used, highlighting:
      - unused tokens (frequency = 0)
      - rare tokens (very small frequency)
      - dominant tokens (large frequency)

    The histogram is computed over token *relative frequencies*:
        freq_i = count_i / sum(counts)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = np.asarray(stats["token_counts"], dtype=np.float64)
    total = counts.sum()

    # Relative frequency of each token (0..1)
    freqs = counts / max(total, 1.0)

    # We use log-spaced bins to make rare vs dominant tokens visible.
    # Include 0 explicitly as its own bin edge.
    nonzero = freqs[freqs > 0]
    if nonzero.size == 0:
        # Degenerate case: nothing used (should not happen, but be safe)
        fig = plt.figure()
        plt.text(0.5, 0.5, "No tokens were used.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    # Log bins for non-zero freqs
    fmin = nonzero.min()
    fmax = nonzero.max()
    log_bins = np.logspace(np.log10(fmin), np.log10(fmax), num=40)

    # Combine a zero bin with log bins
    bins = np.concatenate(([0.0], log_bins))

    fig = plt.figure(figsize=(8, 4))
    plt.hist(freqs, bins=bins)

    plt.xscale("symlog", linthresh=1e-6)  # handles 0 + log behavior
    plt.xlabel("Token relative frequency")
    plt.ylabel("Number of tokens")
    plt.title("Token utilization histogram (frequency distribution)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_per_position_curves(stats: dict, out_dir: Path, run_id: str):
    """
    Plot per-position perplexity and per-position used tokens.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L = stats["seq_len"]
    xs = np.arange(L)

    # Per-position empirical perplexity
    fig = plt.figure()
    plt.plot(xs, stats["per_position_perplexity"])
    plt.xlabel("position")
    plt.ylabel("empirical perplexity")
    plt.title("Per-position empirical perplexity")
    plt.tight_layout()
    plt.savefig(out_dir / f"token_stats_{run_id}_per_position_ppl.png", dpi=200)
    plt.close(fig)

    # Per-position unique tokens
    fig = plt.figure()
    plt.plot(xs, stats["per_position_used_tokens"])
    plt.xlabel("position")
    plt.ylabel("unique tokens used")
    plt.title("Per-position unique token count")
    plt.tight_layout()
    plt.savefig(out_dir / f"token_stats_{run_id}_per_position_used.png", dpi=200)
    plt.close(fig)


# --------------------------------------
# 1) Dataset token utilization & entropy
# --------------------------------------

def evaluate_tokenized_dataset(
    run_dir: Path,
    codes_dataset: torch.Tensor,
    vocab_size: int,
    *,
    metric_tag: str,
    sampler_tag: str,
    seed: int,
    user_tag: str = "dataset",
    temperature: float = None,
    top_p: float = None,
    codebook_scale: float = None,
    save: bool = True,
):
    """
    Compute empirical token utilization + entropy stats on a tokenized dataset.

    Naming convention distinguishes:
      - metric_tag: e.g., "riemann" / "euclidean"
      - sampler_tag: e.g., "base" / "refined"
      - seed: generation seed (for dataset you may still pass a consistent seed)
      - refined params: only when provided

    Saves (if save=True):
      - artifacts/token_stats_<run_id>.json
      - figures/token_stats_<run_id>_global_usage.png
      - figures/token_stats_<run_id>_token_utilization.png
      - figures/token_stats_<run_id>_per_position_ppl.png
      - figures/token_stats_<run_id>_per_position_used.png
    """
    run_id = _build_run_id(
        metric_tag=metric_tag,
        sampler_tag=sampler_tag,
        seed=seed,
        user_tag=user_tag,
        temperature=temperature,
        top_p=top_p,
        codebook_scale=codebook_scale,
    )

    stats = _compute_token_stats(codes_dataset, vocab_size=vocab_size)

    # Include identifiers in the JSON for easier downstream parsing
    stats = dict(stats)
    stats.update({
        "run_id": run_id,
        "metric_tag": str(metric_tag),
        "sampler_tag": str(sampler_tag),
        "seed": int(seed),
        "user_tag": str(user_tag),
        "temperature": None if temperature is None else float(temperature),
        "top_p": None if top_p is None else float(top_p),
        "codebook_scale": None if codebook_scale is None else float(codebook_scale),
    })

    if save:
        run_dir = Path(run_dir)

        # JSON summary
        _save_json(run_dir / "artifacts" / f"token_stats_{run_id}.json", stats)

        # Global plots (NEW)
        _plot_global_token_usage(
            stats, run_dir / "figures" / f"token_stats_{run_id}_global_usage.png"
        )
        _plot_token_utilization(
            stats, run_dir / "figures" / f"token_stats_{run_id}_token_utilization.png"
        )

        # Per-position plots (existing, but now properly named)
        _plot_per_position_curves(
            stats, run_dir / "figures", run_id
        )

    return stats


# ---------------------------------------
# 2) Generation token utilization & entropy
# ---------------------------------------

def evaluate_generated_tokens(
    run_dir: Path,
    generated_tokens: torch.Tensor,
    vocab_size: int,
    *,
    metric_tag: str,
    sampler_tag: str,
    seed: int,
    user_tag: str = "generation",
    temperature: float = None,
    top_p: float = None,
    codebook_scale: float = None,
    save: bool = True,
):
    """
    Compute empirical token utilization + entropy stats on generated token sequences.

    This is identical to evaluate_tokenized_dataset, but semantically used for
    generated tokens.
    """
    return evaluate_tokenized_dataset(
        run_dir=run_dir,
        codes_dataset=generated_tokens,
        vocab_size=vocab_size,
        metric_tag=metric_tag,
        sampler_tag=sampler_tag,
        seed=seed,
        user_tag=user_tag,
        temperature=temperature,
        top_p=top_p,
        codebook_scale=codebook_scale,
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
    *,
    metric_tag: str,
    sampler_tag: str,
    seed: int,
    user_tag: str = "trainset",
    temperature: float = None,
    top_p: float = None,
    codebook_scale: float = None,
    batch_size: int = 256,
    device=None,
    save: bool = True,
):
    """
    Evaluate transformer cross-entropy perplexity on a tokenized dataset.

    Perplexity here is exp(average cross-entropy), which differs from the
    empirical perplexity computed from token frequencies.

    Saves:
      - artifacts/transformer_ppl_<run_id>.json
      - figures/transformer_nll_hist_<run_id>.png
    """
    if device is None:
        device = next(transformer.parameters()).device

    run_id = _build_run_id(
        metric_tag=metric_tag,
        sampler_tag=sampler_tag,
        seed=seed,
        user_tag=user_tag,
        temperature=temperature,
        top_p=top_p,
        codebook_scale=codebook_scale,
    )

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
        loss = criterion(logits.reshape(-1, n_codes), y.reshape(-1))
        losses.append(float(loss.item()))

        B = y.size(0)
        total_tokens += int(B * L)
        total_nll += float(loss.item()) * (B * L)

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = float(np.exp(avg_nll))

    out = {
        "run_id": run_id,
        "metric_tag": str(metric_tag),
        "sampler_tag": str(sampler_tag),
        "seed": int(seed),
        "user_tag": str(user_tag),
        "temperature": None if temperature is None else float(temperature),
        "top_p": None if top_p is None else float(top_p),
        "codebook_scale": None if codebook_scale is None else float(codebook_scale),
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
        _save_json(run_dir / "artifacts" / f"transformer_ppl_{run_id}.json", out)

        # Histogram of batch losses (stability diagnostic)
        fig = plt.figure()
        plt.hist(losses, bins=30)
        plt.xlabel("batch cross-entropy")
        plt.ylabel("count")
        plt.title("Transformer batch loss histogram")
        plt.tight_layout()
        out_fig = run_dir / "figures" / f"transformer_nll_hist_{run_id}.png"
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_fig, dpi=200)
        plt.close(fig)

    return out


