import torch
import numpy as np
from torch.utils.data import DataLoader

@torch.no_grad()
def quantize_dataset_to_tokens_vae(
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


import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def quantize_dataset_to_tokens_vqvae(
    vqvae_model,
    dataset,
    device,
    grid_res: int = 7,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    Encode images with VQ-VAE encoder + quantizer and return tokens.

    Returns:
        codes: torch.LongTensor of shape (N_images, grid_res*grid_res)
               with values in [0, n_codes-1]
    """
    vqvae_model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and (device.type == "cuda"),
        drop_last=False,
    )

    all_codes = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)

        # forward returns recon, vq_loss, indices(B,H,W), perplexity
        _, _, indices, _ = vqvae_model(x)

        # (B, H, W) -> (B, H*W)
        codes = indices.reshape(indices.shape[0], -1).detach().cpu()
        all_codes.append(codes)

    codes_dataset = torch.cat(all_codes, dim=0).long()

    # sanity
    assert codes_dataset.shape[1] == grid_res * grid_res, f"Expected L={grid_res*grid_res}, got {codes_dataset.shape[1]}"
    return codes_dataset    
