import torch
import torch.nn as nn

class GenerativeTransformer2D(nn.Module):
    """
    Autoregressive Transformer over a flattened 2D grid of discrete codes.

    Input sequence:
        [START, t1, t2, ..., t_{L-1}]
    Target sequence:
        [t1, t2, ..., t_L]

    2D positional encoding is implemented with separate row/col embeddings.
    """
    def __init__(
        self,
        n_codes: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        grid_res: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_codes = n_codes
        self.grid_res = grid_res

        # +1 for START token
        self.emb = nn.Embedding(n_codes + 1, d_model)

        # 2D positional embeddings (row/col)
        assert d_model % 2 == 0, "d_model must be even for row/col split"
        self.row_emb = nn.Embedding(grid_res, d_model // 2)
        self.col_emb = nn.Embedding(grid_res, d_model // 2)

        # START position embedding (learned, separate from grid)
        self.start_pos_emb = nn.Parameter(torch.randn(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_codes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) token indices, where T <= 1 + grid_res*grid_res

        Returns:
            logits: (B, T, n_codes)
        """
        B, T = x.shape

        # Token embedding
        x_emb = self.emb(x)  # (B, T, d_model)

        # Build 2D positions for the grid tokens (length = grid_res*grid_res)
        pos = torch.arange(self.grid_res * self.grid_res, device=x.device)
        rows = pos // self.grid_res
        cols = pos % self.grid_res

        grid_pos = torch.cat([self.row_emb(rows), self.col_emb(cols)], dim=-1)  # (L, d_model)
        grid_pos = grid_pos.unsqueeze(0)  # (1, L, d_model)

        full_pos = torch.cat([self.start_pos_emb, grid_pos], dim=1)  # (1, 1+L, d_model)

        x_in = x_emb + full_pos[:, :T, :]

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)

        h = self.transformer(x_in, mask=mask)
        return self.head(h)
