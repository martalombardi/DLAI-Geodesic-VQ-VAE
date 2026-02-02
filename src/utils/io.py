"""
I/O utilities for configuration handling and artifact persistence.

This module provides lightweight helpers to:
- load and save YAML configuration files,
- save JSON metadata files,
- save PyTorch objects (e.g., model checkpoints),
- merge configuration dictionaries (e.g., global + dataset-specific).

The goal is to centralize all file I/O logic in a single place
to improve reproducibility and avoid code duplication.
"""

import json
import yaml
import torch
from pathlib import Path


def load_yaml(path):
    """
    Load a YAML file and return its contents as a Python dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        dict containing the parsed YAML content.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path, obj):
    """
    Save a Python object to a YAML file.

    The parent directory is created automatically if it does not exist.
    Keys are kept in insertion order for readability.

    Args:
        path: Output file path.
        obj: Python object (typically a dict) to be serialized.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(path, obj):
    """
    Save a Python object to a JSON file.

    The parent directory is created automatically if it does not exist.
    The file is formatted with indentation for readability.

    Args:
        path: Output file path.
        obj: Python object (typically metadata) to be serialized.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_torch(path, obj):
    """
    Save a PyTorch object to disk using torch.save.

    This function is typically used for:
    - model state_dicts,
    - optimizer state_dicts,
    - serialized tensors.

    Args:
        path: Output file path.
        obj: PyTorch object to be saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def merge_dicts(a, b):
    """
    Recursively merge two dictionaries.

    Values in dictionary `b` override values in dictionary `a`.
    Nested dictionaries are merged recursively.

    This is mainly used to combine:
    - a global configuration
    - a dataset- or experiment-specific configuration

    Args:
        a: Base dictionary.
        b: Dictionary with overriding values.

    Returns:
        A new dictionary containing the merged configuration.
    """
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out
