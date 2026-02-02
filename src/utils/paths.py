from pathlib import Path
from datetime import datetime

def make_run_dir(outputs_root, dataset, exp_name=None):
    root = Path(outputs_root) / dataset
    root.mkdir(parents=True, exist_ok=True)

    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = root / exp_name
    for sub in ["checkpoints", "artifacts", "samples", "figures"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    return run_dir
