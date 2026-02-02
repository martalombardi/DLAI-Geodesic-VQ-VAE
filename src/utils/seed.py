"""
Reproducibility utilities.

Key principles:
- Always set seeds BEFORE creating models and DataLoaders.
- DataLoader shuffling must use a dedicated torch.Generator.
- CPU generation can be fully deterministic.
- GPU generation is reproducible in practice, but not guaranteed bitwise.
"""

import os
import random
import numpy as np
import torch


def set_seed(
    seed: int,
    device: str = "cpu",
    deterministic: bool = False,
):
    """
    Set random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: random seed.
        device: "cpu" or "cuda".
        deterministic:
            - If True and device == "cpu": enforce strict determinism.
            - If True and device == "cuda": enable best-effort deterministic settings.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if device == "cuda":
        # cuDNN settings (best-effort determinism)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

        if deterministic:
            # Do NOT force torch.use_deterministic_algorithms(True) on GPU
            # because it may break attention / matmul on Colab.
            pass

    else:  # CPU
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)


def seed_worker(worker_id: int):
    """
    DataLoader worker seed function.
    Ensures each worker has a deterministic NumPy/Python RNG state.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
