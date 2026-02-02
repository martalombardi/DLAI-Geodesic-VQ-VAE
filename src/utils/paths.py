"""
Path utilities for experiment management.

This module defines a single helper to create a standardized
directory structure for each experiment run.

Each run directory contains subfolders for:
- checkpoints: model weights and optimizer states,
- artifacts: intermediate objects (e.g., codebooks, tokenized datasets),
- samples: generated images and sampled tokens,
- figures: plots and evaluation visualizations.

This structure ensures that each experiment is self-contained
and reproducible.
"""

from pathlib import Path
from datetime import datetime


def make_run_dir(outputs_root, dataset, exp_name=None):
    """
    Create and return the directory structure for a single experiment run.

    The directory layout is:
        outputs_root / dataset / exp_name /
            ├─ checkpoints/
            ├─ artifacts/
            ├─ samples/
            └─ figures/

    If exp_name is not provided, a timestamp is used.

    Args:
        outputs_root: Root directory where all outputs are stored.
        dataset: Dataset name (used as an intermediate folder).
        exp_name: Optional experiment name. If None, a timestamp is generated.

    Returns:
        Path object pointing to the created run directory.
    """
    root = Path(outputs_root) / dataset
    root.mkdir(parents=True, exist_ok=True)

    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = root / exp_name

    # Standard subdirectories for a run
    for sub in ["checkpoints", "artifacts", "samples", "figures"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    return run_dir
