import torch
import numpy as np
from torch.utils.data import DataLoader

@torch.no_grad()
def quantize_dataset_to_tokens(
    vae_model,
    dataset,
    tokenizer,
    device,
    latent_dim: int,
    grid_res: int,
    batch_size: int = 128,
):
    """
    Encode images with the VAE encoder, flatten the latent grid, and tokenize.
    Returns a tensor of shape (N_images, grid_res*grid_res).
    """
    vae_model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_tokens = []
    for x, _ in loader:
        x = x.to(device)
        recon, mu, logvar = vae_model(x)

        # (B, D, H, W) -> (B*H*W, D)
        z_flat = mu.permute(0, 2, 3, 1).reshape(-1, latent_dim).detach().cpu().numpy()
        tok = tokenizer(z_flat)  # (B*H*W,)
        all_tokens.append(tok.reshape(-1, grid_res * grid_res))

    codes = np.concatenate(all_tokens, axis=0)  # (N_images, 49)
    return torch.from_numpy(codes).long()
