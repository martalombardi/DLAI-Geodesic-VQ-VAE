import numpy as np
from scipy.spatial import KDTree

class LandmarkToMedoidTokenizer:
    """
    Tokenizer that maps a latent vector z to a discrete code index in two steps:

    1) Find the nearest landmark (KDTree in Euclidean space).
    2) Map the landmark to the closest medoid using a precomputed bridge matrix:
       bridge_to_medoids[i, m] = shortest-path distance from landmark i to medoid m.

    This works for both:
    - Riemannian shortest-path bridges
    - Euclidean shortest-path bridges
    because the only difference is how 'bridge_to_medoids' is computed.
    """
    def __init__(self, landmarks: np.ndarray, bridge_to_medoids: np.ndarray):
        self.tree = KDTree(landmarks)
        # For each landmark i, pick the nearest medoid index
        self.landmark_to_code = bridge_to_medoids.argmin(axis=1).astype(np.int64)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """
        Args:
            z: (N, D) latent vectors (numpy).

        Returns:
            tokens: (N,) integer codes in [0, n_codes-1].
        """
        _, idx = self.tree.query(z)          # nearest landmark indices
        return self.landmark_to_code[idx]    # map landmark -> code
