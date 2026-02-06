# VQ-VAE a posteriori with Geodesic Quantization

**Deep Learning and Applied AI (DLAI) Project — 2025** *Sapienza University of Rome, 2nd Semester, A.Y. 2024/2025*

Discrete latent representations are a key component of modern generative pipelines, as they enable the use of powerful autoregressive priors while preserving spatial structure. Most existing discretization strategies rely on Euclidean distances in latent space, implicitly assuming a flat geometry that is rarely justified for deep generative models. This project investigates how the geometry induced by a convolutional VAE decoder can be exploited to construct more meaningful discrete representations. The main contribution is a geometry-aware discretization method that replaces standard Euclidean vector quantization with a Riemannian approximation of latent distances, computed via Jacobian–vector products and shortest paths on a neighborhood graph. These geodesic distances are then used to learn a discrete codebook through k-medoids clustering, ensuring consistency with the intrinsic structure of the latent manifold. The approach is evaluated on MNIST and FashionMNIST using the same lattice-structured VAE and Transformer prior as a standard VQ-VAE baseline, allowing for a controlled comparison. Results show that the proposed discretization leads to richer token usage, clearer latent organization and more expressive generative behavior, especially on more heterogeneous datasets.

## Reproducibility

All experiments are fully reproducible.

To install the required dependencies:
```bash
pip install -r requirements.txt
```

The notebook `dlai_geodesic_vq_vae.ipynb` executes the full pipeline in a deterministic and reproducible way, including environment setup, data loading, model training, Riemannian discretization, evaluation and generation.

## Project Structure
### Configuration
All experiments are configured through YAML files stored in `configs/`:
* **`global.yaml`**: Shared settings.
* **`mnist.yaml`**: MNIST-specific configuration.
* **`fashion_mnist.yaml`**: FashionMNIST-specific configuration.

### Data
The `data/` directory contains MNIST and FashionMNIST datasets automatically downloaded.

### Outputs
All experiment results are saved in `outputs/` (formerly `smoke_test`), organized by dataset and model.

### Source Code Structure
The main logic is contained in `src/`:

* **`models/`**: Implementations of VAE, VQ-VA and Transformer architectures.
* **`geodesy/`**: Riemannian distance computation using Jacobian–vector products, graph construction and shortest-path algorithms.
* **`tokenization/`**: Discretization pipelines for both Euclidean and Riemannian quantization.
* **`train/`**: Training scripts for VAE, VQ-VAE and Transformer priors.
* **`eval/`**: Evaluation and analysis code.
* **`utils/`**: Reproducibility utilities, logging helpers and shared functions.

### Notebooks
* **`dlai_geodesic_vq_vae.ipynb`**: Main executable notebook (end-to-end, fully reproducible). You can view the static version at this [link](https://nbviewer.org/github/martalombardi/DLAI-Geodesic-VQ-VAE/blob/main/dlai_geodesic_vq_vae.ipynb).

## Development Note
The entire project was developed and executed on **Google Colab**. The notebook is optimized for Colab environments to ensure that the heavy computations required for Jacobian–vector products and shortest-path algorithms are handled efficiently.
