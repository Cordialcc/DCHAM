import torch
import torch.nn as nn

from .config import DCHAMConfig
from .depth_features import DepthFeatureNet
from .hyper_attention import HyperAttentionHead


class DCHAM(nn.Module):
    """
    Depth-Conditioned HyperAttention Module.

    Components:
      1. DepthFeatureNet: depth map -> multi-scale spatial features
      2. HyperAttentionHead x 2 (fine + coarse): depth generates W_K, W_V
      3. Spatial queries + text conditioning: defines extraction targets
      4. Output projection: projects to LLM dimension
    """

    def __init__(self, config: DCHAMConfig):
        super().__init__()
        self.config = config
        M = config.num_queries
        d_model = config.d_model
        d_lm = config.d_lm

        # Component 1: Depth Feature Network
        self.depth_net = DepthFeatureNet(d_depth=config.d_depth)

        # Component 2: Two-scale HyperAttention
        head_kwargs = dict(
            d_vit=config.d_vit, d_head=config.d_head,
            num_heads=config.num_heads, num_basis=config.num_basis,
            rank=config.rank, d_depth=config.d_depth,
        )
        self.hyper_fine = HyperAttentionHead(**head_kwargs)
        self.hyper_coarse = HyperAttentionHead(**head_kwargs)

        # Component 3: Spatial queries + text conditioning
        self.queries = nn.Parameter(torch.randn(M, d_model) * 0.02)
        self.text_proj = nn.Sequential(
            nn.Linear(d_lm, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.head_proj = nn.Linear(d_model, config.num_heads * config.d_head)

        # Component 4: Output
        self.scale_embed = nn.Embedding(2, d_model)
        self.out_proj = nn.Linear(d_model, d_lm)
        self.out_norm = nn.LayerNorm(d_lm)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        depth_map: torch.Tensor,
        text_embeddings: torch.Tensor,
        vit_grid_h: int,
        vit_grid_w: int,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens:  R^{B x N x d_vit}
            depth_map:      R^{B x 1 x H x W}
            text_embeddings: R^{B x L x d_lm}
            vit_grid_h, vit_grid_w: ViT patch grid dimensions

        Returns:
            spatial_tokens: R^{B x M x d_lm}
        """
        B = visual_tokens.shape[0]
        cfg = self.config
        M_half = cfg.num_queries // 2

        # 1. Depth features
        depth_feats = self.depth_net(depth_map, vit_grid_h, vit_grid_w)

        # 2. Text-conditioned spatial queries
        text_pool = text_embeddings.mean(dim=1)
        text_mod = self.text_proj(text_pool)
        Q = self.queries.unsqueeze(0) + text_mod.unsqueeze(1)
        Q = self.head_proj(Q).view(B, cfg.num_queries, cfg.num_heads, cfg.d_head)
        Q_fine, Q_coarse = Q[:, :M_half], Q[:, M_half:]

        # 3. Two-scale HyperAttention
        S_fine = self.hyper_fine(visual_tokens, depth_feats["global"], Q_fine)
        S_coarse = self.hyper_coarse(visual_tokens, depth_feats["global"], Q_coarse)

        # 4. Scale embedding + concatenate
        dev = S_fine.device
        S_fine = S_fine + self.scale_embed(torch.zeros(1, dtype=torch.long, device=dev))
        S_coarse = S_coarse + self.scale_embed(torch.ones(1, dtype=torch.long, device=dev))
        S = torch.cat([S_fine, S_coarse], dim=1)

        # 5. Project to LLM dimension
        return self.out_norm(self.out_proj(S))
