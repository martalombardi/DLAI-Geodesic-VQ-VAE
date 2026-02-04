import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import silhouette_score

try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    print("Warning: sklearn-extra not found. Install with: pip install scikit-learn-extra")

def get_analysis_dir(run_dir):
    """Creates and returns a path for analysis results."""
    analysis_path = Path(run_dir) / "analysis_results"
    analysis_path.mkdir(parents=True, exist_ok=True)
    return analysis_path

@torch.no_grad()
def eval_reconstruction_losses_vqvae(model, loader, device):
    model.eval()
    bce_sum = 0.0
    n_pix = 0
    for x, _ in loader:
        x = x.to(device)
        recon, _, _, _ = model(x)
        bce = F.binary_cross_entropy(recon, x, reduction="sum")
        bce_sum += float(bce.item())
        n_pix += int(x.numel())
    return {"bce_per_pixel": bce_sum / max(n_pix, 1), "n_images": len(loader.dataset)}

def nn_l2_distances(gen_flat, ref_flat, block=128):
    dmins = []
    for i in range(0, gen_flat.size(0), block):
        g = gen_flat[i:i+block]
        g2 = (g*g).sum(dim=1, keepdim=True)
        r2 = (ref_flat*ref_flat).sum(dim=1).unsqueeze(0)
        dist2 = g2 + r2 - 2.0 * (g @ ref_flat.t())
        dmin = torch.sqrt(torch.clamp(dist2.min(dim=1).values, min=0.0))
        dmins.append(dmin.cpu())
    return torch.cat(dmins, dim=0)

@torch.no_grad()
def extract_latents_pre_quant(model, loader, device):
    model.eval()
    Z = []
    for x, _ in loader:
        x = x.to(device)
        z_e = model.encoder(x)
        z = z_e.mean(dim=(2,3)) # Mean pooling grid to vector
        Z.append(z.cpu())
    return torch.cat(Z, dim=0).numpy()

def build_geodesic_distances(Z, k_neighbors=10):
    n_samples = Z.shape[0]
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean").fit(Z)
    dists, nbrs = nn.kneighbors(Z)
    
    rows = np.repeat(np.arange(n_samples), k_neighbors)
    cols = nbrs.reshape(-1)
    vals = dists.reshape(-1)
    
    A = csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    A = A.minimum(A.T) # Symmetrize
    
    D_geo = dijkstra(A, directed=False)
    finite = np.isfinite(D_geo)
    D_geo[~finite] = D_geo[finite].max() * 1.5
    return D_geo
