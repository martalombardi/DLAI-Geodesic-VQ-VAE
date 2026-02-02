import numpy as np
import torch
from torch.autograd.functional import jvp

from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# ============================================================
# 1) Landmark collection (with positional indexing)
# ============================================================

@torch.no_grad()
def collect_latent_landmarks(
    vae_model,
    loader,
    device,
    latent_dim: int,
    grid_res: int,
    n_landmarks: int = 5000,
):
    """
    Collect latent means (mu) from the encoder and flatten the grid into landmarks.

    Returns:
        landmarks: (N, latent_dim) np.ndarray
        positions: (N,) np.ndarray with integer indices in [0, grid_res*grid_res-1]
                  indicating the original grid cell of each landmark.
    """
    vae_model.eval()
    collected = []
    pos_list = []

    for x, _ in loader:
        x = x.to(device)

        # Works with either:
        # - encoder returning (mu, logvar), or
        # - encoder returning a tensor that can be chunked into mu/logvar.
        out = vae_model.encoder(x)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu, _ = out
        else:
            mu, _ = torch.chunk(out, 2, dim=1)

        B, D, H, W = mu.shape
        assert D == latent_dim, f"Expected latent_dim={latent_dim}, got {D}"
        assert H == grid_res and W == grid_res, f"Expected grid_res={grid_res}, got {H}x{W}"

        # (B, D, H, W) -> (B*H*W, D)
        flat_mu = mu.permute(0, 2, 3, 1).reshape(-1, latent_dim).cpu()
        collected.append(flat_mu)

        # Positions: 0..(H*W-1) repeated for each element in the batch
        pos = torch.arange(H * W).repeat(B)
        pos_list.append(pos)

        if sum(t.size(0) for t in collected) >= n_landmarks:
            break

    landmarks = torch.cat(collected, dim=0)[:n_landmarks].numpy()
    positions = torch.cat(pos_list, dim=0)[:n_landmarks].numpy().astype(np.int64)
    return landmarks, positions


# ============================================================
# 2) kNN adjacency
# ============================================================

def build_knn_adjacency(landmarks: np.ndarray, knn_k: int = 15):
    """
    Build a sparse kNN connectivity graph (unweighted).
    """
    knn = NearestNeighbors(n_neighbors=knn_k).fit(landmarks)
    adj = knn.kneighbors_graph(landmarks, mode="connectivity")  # sparse
    return adj


def symmetrize_adjacency(adj):
    """
    Make adjacency undirected by union of edges.
    """
    return ((adj + adj.T) > 0).astype(np.float32)


# ============================================================
# 3) Edge weight computation
# ============================================================

def compute_edge_weights_euclidean(landmarks: np.ndarray, adjacency, eps_weight: float = 1e-5):
    """
    Edge weights = Euclidean distance in latent space.
    Returns a weighted sparse matrix with same sparsity as adjacency.
    """
    sources, targets = adjacency.nonzero()
    diffs = landmarks[sources] - landmarks[targets]
    w = np.linalg.norm(diffs, axis=1).astype(np.float32)
    w = np.maximum(w, eps_weight)
    return csr_matrix((w, (sources, targets)), shape=adjacency.shape)


def compute_edge_weights_riemannian_jvp(
    vae_model,
    landmarks: np.ndarray,
    positions: np.ndarray,
    adjacency,
    device,
    latent_dim: int,
    grid_res: int,
    batch_edges: int = 64,
    eps_weight: float = 1e-5,
):
    """
    Edge weights approximated by || J(mid) * (p2 - p1) || via JVP,
    where J is the decoder Jacobian wrt the latent patch inserted at its original grid cell.

    This implements your "positional patch":
    each landmark keeps its grid cell index, and the decoder input places z in that cell.
    """
    vae_model.eval()

    sources, targets = adjacency.nonzero()
    points = torch.from_numpy(landmarks).float().to(device)
    pos_idx = torch.from_numpy(positions).long().to(device)

    weights = np.zeros(len(sources), dtype=np.float32)

    def decode_from_single_patch(z_patch: torch.Tensor, pos_flat: torch.Tensor):
        """
        z_patch: (B, latent_dim)
        pos_flat: (B,) integer indices in [0, grid_res*grid_res-1]

        We build a latent grid of zeros and place each z_patch into its corresponding cell.
        Output is flattened image vector per sample to compute norms in output space.
        """
        B = z_patch.size(0)
        grid = torch.zeros(B, latent_dim, grid_res, grid_res, device=device)

        ii = pos_flat // grid_res
        jj = pos_flat % grid_res

        grid[torch.arange(B, device=device), :, ii, jj] = z_patch
        out = vae_model.decoder(grid)            # (B, C, H, W)
        return out.reshape(B, -1)                # (B, P)

    for i in range(0, len(sources), batch_edges):
        end = min(i + batch_edges, len(sources))
        s = sources[i:end]
        t = targets[i:end]

        p1 = points[s]
        p2 = points[t]
        mid = 0.5 * (p1 + p2)
        tan = (p2 - p1)

        # Use the source node position (consistent with your current choice)
        pos_batch = pos_idx[s]

        # JVP: derivative wrt z only; positions are constants
        _, jvp_out = jvp(lambda z: decode_from_single_patch(z, pos_batch), (mid,), (tan,))
        w = torch.linalg.norm(jvp_out, dim=1).detach().cpu().numpy()
        weights[i:end] = w

    weights = np.maximum(weights, eps_weight)
    return csr_matrix((weights, (sources, targets)), shape=adjacency.shape)


# ============================================================
# 4) Shortest path distances + KMedoids codebook
# ============================================================

def shortest_path_distances(weighted_adj, directed: bool = False, inf_fill_factor: float = 2.0):
    """
    Compute all-pairs shortest path distances on the weighted graph.
    Replace inf with a finite large value for KMedoids stability.
    """
    dist = dijkstra(weighted_adj, directed=directed)

    finite = np.isfinite(dist)
    if not np.all(finite):
        max_finite = np.nanmax(dist[finite])
        dist[~finite] = max_finite * inf_fill_factor

    return dist


def fit_kmedoids_codebook(dist_matrix: np.ndarray, landmarks: np.ndarray, n_codes: int, seed: int = 42):
    """
    Fit KMedoids with a precomputed distance matrix.
    Returns:
        codebook: (n_codes, latent_dim)
        medoid_indices: (n_codes,)
    """
    kmed = KMedoids(n_clusters=n_codes, metric="precomputed", random_state=seed)
    kmed.fit(dist_matrix)
    medoid_indices = kmed.medoid_indices_
    codebook = landmarks[medoid_indices]
    return codebook, medoid_indices


def compute_bridge_to_medoids(weighted_adj, medoid_indices, directed: bool = False):
    """
    Compute distances from every node to each medoid (shortest path).
    Output shape: (N, n_codes), where entry (i, m) = dist(node i -> medoid m).
    """
    # dijkstra returns shape (len(indices), N) when indices are provided
    d = dijkstra(weighted_adj, indices=medoid_indices, directed=directed)
    return d.T  # (N, n_codes)


# ============================================================
# 5) One-call pipeline
# ============================================================

def build_codebook_and_bridges(
    vae_model,
    train_loader,
    device,
    latent_dim: int,
    grid_res: int,
    n_landmarks: int,
    knn_k: int,
    metric: str,
    n_codes: int,
    kmedoids_seed: int = 42,
    symmetrize: bool = True,
    batch_edges: int = 64,
    inf_fill_factor: float = 2.0,
    eps_weight: float = 1e-5,
):
    """
    End-to-end pipeline:
    - collect landmarks + positions
    - build kNN adjacency
    - compute edge weights (riemannian or euclidean)
    - shortest path distance matrix
    - KMedoids codebook
    - bridge distances to medoids
    """
    landmarks, positions = collect_latent_landmarks(
        vae_model=vae_model,
        loader=train_loader,
        device=device,
        latent_dim=latent_dim,
        grid_res=grid_res,
        n_landmarks=n_landmarks,
    )

    adj = build_knn_adjacency(landmarks, knn_k=knn_k)
    if symmetrize:
        adj = symmetrize_adjacency(adj)

    metric = metric.lower()
    if metric == "euclidean":
        weighted_adj = compute_edge_weights_euclidean(landmarks, adj, eps_weight=eps_weight)
    elif metric == "riemannian":
        weighted_adj = compute_edge_weights_riemannian_jvp(
            vae_model=vae_model,
            landmarks=landmarks,
            positions=positions,
            adjacency=adj,
            device=device,
            latent_dim=latent_dim,
            grid_res=grid_res,
            batch_edges=batch_edges,
            eps_weight=eps_weight,
        )
    else:
        raise ValueError("metric must be 'riemannian' or 'euclidean'")

    dist = shortest_path_distances(weighted_adj, directed=False, inf_fill_factor=inf_fill_factor)
    codebook, medoid_indices = fit_kmedoids_codebook(dist, landmarks, n_codes=n_codes, seed=kmedoids_seed)
    bridge = compute_bridge_to_medoids(weighted_adj, medoid_indices, directed=False)

    return {
        "landmarks": landmarks,
        "positions": positions,
        "adjacency": adj,
        "weighted_adjacency": weighted_adj,
        "dist_matrix": dist,
        "codebook": codebook,
        "medoid_indices": medoid_indices,
        "bridge_to_medoids": bridge,  # (N_landmarks, n_codes)
    }
