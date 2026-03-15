from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryRouter(nn.Module):
    """
    Maps scene geometry descriptor to per-layer LoRA mixing weights.
    Uses a shared MLP with per-layer embeddings.
    """

    def __init__(self, d_geo: int = 256, num_bases: int = 6, num_layers: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.layer_embed = nn.Embedding(num_layers, d_geo)
        self.mlp = nn.Sequential(
            nn.Linear(d_geo, d_geo),
            nn.GELU(),
            nn.Linear(d_geo, num_bases),
        )

    def forward(self, z_geo: torch.Tensor) -> Dict[int, torch.Tensor]:
        alphas = {}
        for i in range(self.num_layers):
            layer_emb = self.layer_embed(
                torch.tensor(i, device=z_geo.device)
            )
            input_i = z_geo + layer_emb.unsqueeze(0)
            alphas[i] = F.softmax(self.mlp(input_i), dim=-1)
        return alphas
