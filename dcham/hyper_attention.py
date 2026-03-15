import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperAttentionHead(nn.Module):
    """
    Core innovation: depth features dynamically generate attention projection
    matrices via basis kernel combination.

    Depth -> HyperNetwork -> alpha coefficients -> W_K, W_V from basis library
    Then: tri-modal attention with Q (text), K/V (depth-generated from RGB)
    """

    def __init__(
        self,
        d_vit: int = 1280,
        d_head: int = 64,
        num_heads: int = 8,
        num_basis: int = 12,
        rank: int = 24,
        d_depth: int = 128,
    ):
        super().__init__()
        self.d_head = d_head
        self.num_heads = num_heads
        self.num_basis = num_basis
        self.scale = math.sqrt(d_head)

        # Context Encoder: global depth -> context vector
        ctx_dim = d_depth * 2
        self.context_enc = nn.Sequential(
            nn.Linear(d_depth, d_depth),
            nn.GELU(),
            nn.LayerNorm(d_depth),
            nn.Linear(d_depth, ctx_dim),
            nn.GELU(),
            nn.LayerNorm(ctx_dim),
        )

        # Coefficient Generator: context -> per-head basis weights
        self.coeff_K = nn.Linear(ctx_dim, num_heads * num_basis)
        self.coeff_V = nn.Linear(ctx_dim, num_heads * num_basis)

        # Basis Kernel Library (core learnable parameters)
        # Each basis: B_i = U_i @ V_i^T  in R^{d_head x d_vit}
        self.U_K = nn.Parameter(torch.randn(num_basis, d_head, rank) * 0.02)
        self.V_K = nn.Parameter(torch.randn(num_basis, d_vit, rank) * 0.02)
        self.U_V = nn.Parameter(torch.randn(num_basis, d_head, rank) * 0.02)
        self.V_V = nn.Parameter(torch.randn(num_basis, d_vit, rank) * 0.02)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        depth_global: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: R^{B x N x d_vit} -- ViT output (frozen)
            depth_global:  R^{B x d_depth}   -- global depth feature
            Q:             R^{B x M x H x d_head} -- spatial queries

        Returns:
            R^{B x M x (H * d_head)} -- spatial token features
        """
        B = visual_tokens.shape[0]

        # --- HyperNetwork: depth -> basis coefficients ---
        ctx = self.context_enc(depth_global)
        alpha_K = self.coeff_K(ctx).view(B, self.num_heads, self.num_basis)
        alpha_V = self.coeff_V(ctx).view(B, self.num_heads, self.num_basis)
        alpha_K = F.softmax(alpha_K, dim=-1)
        alpha_V = F.softmax(alpha_V, dim=-1)

        # --- Basis kernel assembly ---
        K = self._assemble(visual_tokens, alpha_K, self.V_K, self.U_K)
        V = self._assemble(visual_tokens, alpha_V, self.V_V, self.U_V)

        # --- Tri-modal attention ---
        attn = torch.einsum("bmhd, bnhd -> bhmn", Q, K) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhmn, bnhd -> bmhd", attn, V)

        return out.reshape(B, Q.shape[1], self.num_heads * self.d_head)

    def _assemble(
        self,
        X: torch.Tensor,
        alpha: torch.Tensor,
        V_basis: nn.Parameter,
        U_basis: nn.Parameter,
    ) -> torch.Tensor:
        """
        Efficient basis kernel assembly. Never materializes full W matrix.

        X:       R^{B x N x d_vit}
        alpha:   R^{B x H x n}
        V_basis: R^{n x d_vit x r}
        U_basis: R^{n x d_head x r}

        Returns: R^{B x N x H x d_head}
        """
        Z = torch.einsum("bnd, kdr -> bnkr", X, V_basis)
        K_all = torch.einsum("bnkr, kfr -> bnkf", Z, U_basis)
        return torch.einsum("bhk, bnkf -> bnhf", alpha, K_all)
