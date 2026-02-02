from pathlib import Path
import numpy as np
import json

def save_codebook_artifacts(run_dir: Path, out: dict, meta: dict):
    art = Path(run_dir) / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    np.save(art / "landmarks.npy", out["landmarks"])
    np.save(art / "bridge_to_medoids.npy", out["bridge_to_medoids"])
    np.save(art / "codebook.npy", out["codebook"])
    np.save(art / "medoid_indices.npy", out["medoid_indices"])

    with open(art / "codebook_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
