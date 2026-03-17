"""
Baseline and ablation models for DepthPEFT study.

1. StaticLoRA         — Fixed LoRA, same param budget, no depth conditioning.
2. UniformAlphaGeoLoRA — GeoLoRA with router replaced by uniform 1/K weights.
3. DepthTokenInjection — Depth features prepended as visual tokens (Spa3R-style).
4. DepthGate          — Standard LoRA + depth-conditioned scalar gating (Level 1).
5. DepthFiLM          — Standard LoRA + depth-conditioned rank-space FiLM (Level 2).
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GeoLoRAConfig
from .depth_geometry import DepthGeometryNet


# ---------------------------------------------------------------------------
# Helper: compute static LoRA rank to match GeoLoRA's total param budget
# ---------------------------------------------------------------------------

def _compute_static_lora_rank(config: GeoLoRAConfig) -> int:
    """
    Choose rank r so that total static-LoRA params ≈ total GeoLoRA params.

    GeoLoRA budget = DepthGeometryNet + GeometryRouter + LoRABasisBanks + gates.
    StaticLoRA budget = sum over layers/projs of (r * d_in + d_out * r).
    """
    # --- GeoLoRA param count ---
    # DepthGeometryNet (rough, matches depth_geometry.py architecture)
    depth_net_params = (
        3 * 64 * 7 * 7 + 64            # conv1 weight+bias
        + 64 * 128 * 3 * 3 + 128       # conv2
        + 128 * 256 * 3 * 3 + 256      # conv3
        + 256                           # layernorm
        + 256 * config.d_geo + config.d_geo  # proj
    )
    # GeometryRouter
    num_layers = len(config.target_layers)
    router_params = (
        num_layers * config.d_geo       # layer_embed
        + config.d_geo * config.d_geo + config.d_geo  # mlp[0]
        + config.d_geo * config.num_bases + config.num_bases  # mlp[2]
    )
    # LoRABasisBanks
    bank_params = 0
    for proj in config.target_projections:
        d_in, d_out = config.proj_dims(proj)
        bank_params += config.num_bases * (
            config.lora_rank * d_in + d_out * config.lora_rank
        )
    bank_params *= num_layers
    # Gates (one scalar per layer per projection)
    gate_params = num_layers * len(config.target_projections)

    total_geolora = depth_net_params + router_params + bank_params + gate_params

    # --- StaticLoRA param count per unit of rank ---
    rank_cost = 0
    for proj in config.target_projections:
        d_in, d_out = config.proj_dims(proj)
        rank_cost += d_in + d_out  # A is (r, d_in), B is (d_out, r)
    rank_cost *= num_layers

    r = max(1, round(total_geolora / rank_cost))
    return r


# ============================================================================
# 1. StaticLoRA — standard fixed LoRA
# ============================================================================

class StaticLoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a fixed low-rank adapter.
    output = frozen(h) + (h @ A^T) @ B^T
    """

    def __init__(self, frozen_linear: nn.Linear, rank: int):
        super().__init__()
        self.frozen_linear = frozen_linear
        for p in self.frozen_linear.parameters():
            p.requires_grad = False

        d_in = frozen_linear.in_features
        d_out = frozen_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.scaling = 1.0 / rank

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        frozen_out = self.frozen_linear(h)
        lora_out = F.linear(F.linear(h, self.lora_A), self.lora_B)
        return frozen_out + self.scaling * lora_out


class Qwen2VLWithStaticLoRA(nn.Module):
    """
    Qwen2.5-VL + standard fixed LoRA on the same layers/projections.
    Same parameter budget as GeoLoRA, no depth conditioning.
    """

    def __init__(self, base_model, config: GeoLoRAConfig, processor):
        super().__init__()
        self.base = base_model
        self.config = config
        self.processor = processor
        self.lora_rank = _compute_static_lora_rank(config)
        self._wrapped_layers = {}

        self._wrap_target_layers()

    def _wrap_target_layers(self):
        llm_layers = self.base.model.layers
        for local_idx, global_idx in enumerate(self.config.target_layers):
            layer = llm_layers[global_idx]
            self._wrapped_layers[local_idx] = {}
            for proj_name in self.config.target_projections:
                original_linear = getattr(layer.self_attn, proj_name)
                wrapper = StaticLoRALinear(original_linear, self.lora_rank)
                setattr(layer.self_attn, proj_name, wrapper)
                self._wrapped_layers[local_idx][proj_name] = wrapper

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        config: GeoLoRAConfig,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_name)

        for param in base_model.parameters():
            param.requires_grad = False

        return cls(base_model, config, processor)

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        depth_maps=None,
        labels=None,
        **kwargs,
    ):
        # depth_maps accepted but ignored — keeps evaluation scripts uniform
        return self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

    def trainable_parameters(self):
        lora_params = []
        for layer_wrappers in self._wrapped_layers.values():
            for wrapper in layer_wrappers.values():
                lora_params.extend(
                    [wrapper.lora_A, wrapper.lora_B]
                )
        return [{"params": lora_params, "lr": self.config.lr_lora_bank}]


# ============================================================================
# 2. UniformAlphaGeoLoRA — GeoLoRA with uniform mixing weights
# ============================================================================

class UniformRouter(nn.Module):
    """
    Dummy router that always returns alpha = 1/K for every layer.
    Keeps the GeoLoRA basis-bank mechanism but removes scene adaptation.
    """

    def __init__(self, num_bases: int, num_layers: int):
        super().__init__()
        self.num_bases = num_bases
        self.num_layers = num_layers

    def forward(self, z_geo: torch.Tensor) -> Dict[int, torch.Tensor]:
        batch = z_geo.size(0)
        uniform = torch.full(
            (batch, self.num_bases),
            1.0 / self.num_bases,
            device=z_geo.device,
            dtype=z_geo.dtype,
        )
        return {i: uniform for i in range(self.num_layers)}


def make_uniform_alpha_geolora(config: GeoLoRAConfig):
    """
    Build a Qwen2VLWithGeoLoRA instance whose router is replaced by
    UniformRouter.  Call *after* Qwen2VLWithGeoLoRA.from_pretrained.

    Usage::

        from geolora.model import Qwen2VLWithGeoLoRA
        model = Qwen2VLWithGeoLoRA.from_pretrained(model_name, config)
        make_uniform_alpha_geolora(config, model)

    Returns the patched model (modified in-place).
    """
    from .model import Qwen2VLWithGeoLoRA

    def _patch(model: Qwen2VLWithGeoLoRA):
        num_layers = len(config.target_layers)
        model.geolora.router = UniformRouter(
            num_bases=config.num_bases,
            num_layers=num_layers,
        )
        # Freeze the dummy router (no trainable params) — depth_net still trains
        return model

    return _patch


# ---------------------------------------------------------------------------
# Helper: compute depth token count to match GeoLoRA's total param budget
# ---------------------------------------------------------------------------

def _compute_matched_depth_tokens(config: GeoLoRAConfig) -> int:
    """
    Choose n_tokens so that DepthGeometryNet + DepthTokenProjector ≈ total GeoLoRA params.

    Projector layout:
        Linear(d_geo, d_geo*2)          — d_geo * (d_geo*2) + (d_geo*2) params
        Linear(d_geo*2, n * d_lm)       — (d_geo*2) * (n * d_lm) + (n * d_lm) params

    We solve for n = floor((target - first_linear_cost) / per_token_cost).
    """
    # --- GeoLoRA total param count (same formula as _compute_static_lora_rank) ---
    depth_net_params = (
        3 * 64 * 7 * 7 + 64
        + 64 * 128 * 3 * 3 + 128
        + 128 * 256 * 3 * 3 + 256
        + 256
        + 256 * config.d_geo + config.d_geo
    )
    num_layers = len(config.target_layers)
    router_params = (
        num_layers * config.d_geo
        + config.d_geo * config.d_geo + config.d_geo
        + config.d_geo * config.num_bases + config.num_bases
    )
    bank_params = 0
    for proj in config.target_projections:
        d_in, d_out = config.proj_dims(proj)
        bank_params += config.num_bases * (
            config.lora_rank * d_in + d_out * config.lora_rank
        )
    bank_params *= num_layers
    gate_params = num_layers * len(config.target_projections)
    total_geolora = depth_net_params + router_params + bank_params + gate_params

    # --- Projector budget (subtract shared DepthGeometryNet) ---
    projector_budget = total_geolora - depth_net_params

    # First linear: d_geo -> d_geo * 2  (weight + bias)
    first_linear = config.d_geo * (config.d_geo * 2) + (config.d_geo * 2)
    # Per-token cost of second linear: (d_geo * 2) * d_lm + d_lm
    per_token = (config.d_geo * 2) * config.d_lm + config.d_lm

    n = max(1, math.floor((projector_budget - first_linear) / per_token))
    return n


# ============================================================================
# 3. DepthTokenInjection — depth features as prepended visual tokens
# ============================================================================

class DepthTokenProjector(nn.Module):
    """
    Projects the scene geometry descriptor into a sequence of depth tokens.
    z_geo (B, d_geo) -> depth_tokens (B, N_depth, d_lm)
    """

    def __init__(self, d_geo: int, d_lm: int, n_depth_tokens: int):
        super().__init__()
        self.n_tokens = n_depth_tokens
        self.proj = nn.Sequential(
            nn.Linear(d_geo, d_geo * 2),
            nn.GELU(),
            nn.Linear(d_geo * 2, n_depth_tokens * d_lm),
        )

    def forward(self, z_geo: torch.Tensor) -> torch.Tensor:
        out = self.proj(z_geo)                       # (B, N_depth * d_lm)
        return out.view(z_geo.size(0), self.n_tokens, -1)  # (B, N_depth, d_lm)


class Qwen2VLWithDepthTokens(nn.Module):
    """
    Qwen2.5-VL with depth features injected as extra tokens prepended to
    the inputs_embeds sequence (Spa3R-style).

    Uses the same DepthGeometryNet to extract z_geo, then projects it into
    N_depth tokens in the LLM hidden dimension.  The token count is computed
    dynamically to match GeoLoRA's total parameter budget.
    """

    def __init__(self, base_model, config: GeoLoRAConfig, processor):
        super().__init__()
        self.base = base_model
        self.config = config
        self.processor = processor

        self.n_depth_tokens = _compute_matched_depth_tokens(config)
        self.depth_net = DepthGeometryNet(d_geo=config.d_geo)
        self.token_proj = DepthTokenProjector(
            d_geo=config.d_geo,
            d_lm=config.d_lm,
            n_depth_tokens=self.n_depth_tokens,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        config: GeoLoRAConfig,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_name)

        for param in base_model.parameters():
            param.requires_grad = False

        model = cls(base_model, config, processor)
        model.depth_net = model.depth_net.to(dtype=torch_dtype)
        model.token_proj = model.token_proj.to(dtype=torch_dtype)
        return model

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        depth_maps=None,
        labels=None,
        **kwargs,
    ):
        # Get input embeddings from the base model's embedding layer
        inputs_embeds = self.base.model.embed_tokens(input_ids)

        if depth_maps is not None:
            z_geo = self.depth_net(depth_maps)
            depth_tokens = self.token_proj(z_geo)  # (B, N_depth, d_lm)

            # Prepend depth tokens to the embedding sequence
            inputs_embeds = torch.cat([depth_tokens, inputs_embeds], dim=1)

            # Extend attention mask for the prepended tokens
            batch = attention_mask.size(0)
            depth_mask = torch.ones(
                batch, self.n_depth_tokens,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([depth_mask, attention_mask], dim=1)

            # Shift labels to account for prepended tokens
            if labels is not None:
                ignore = torch.full(
                    (batch, self.n_depth_tokens),
                    -100,
                    device=labels.device,
                    dtype=labels.dtype,
                )
                labels = torch.cat([ignore, labels], dim=1)

        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

    def trainable_parameters(self):
        params = list(self.depth_net.parameters()) + list(self.token_proj.parameters())
        return [{"params": params, "lr": self.config.lr_depth_router}]


# ============================================================================
# 4. DepthGate — Standard LoRA + depth-conditioned scalar gating (Level 1)
# ============================================================================

class DepthGatedLoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with standard LoRA, gated by a depth-derived scalar.
    output = frozen(h) + g(z_geo) * (1/r) * B @ A @ h
    """

    def __init__(self, frozen_linear: nn.Linear, rank: int, d_geo: int):
        super().__init__()
        self.frozen_linear = frozen_linear
        for p in self.frozen_linear.parameters():
            p.requires_grad = False

        d_in = frozen_linear.in_features
        d_out = frozen_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = 1.0 / rank

        # Depth gate: z_geo -> scalar gate via learned linear
        self.gate_proj = nn.Linear(d_geo, 1)
        self._z_geo = None

    def set_z_geo(self, z_geo: torch.Tensor):
        self._z_geo = z_geo

    def clear_z_geo(self):
        self._z_geo = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        frozen_out = self.frozen_linear(h)
        lora_out = F.linear(F.linear(h, self.lora_A), self.lora_B)
        if self._z_geo is not None:
            gate = torch.sigmoid(self.gate_proj(self._z_geo))  # (B, 1)
            gate = gate.unsqueeze(1)  # (B, 1, 1) for broadcasting over seq
            return frozen_out + gate * self.scaling * lora_out
        return frozen_out + self.scaling * lora_out


class Qwen2VLWithDepthGate(nn.Module):
    """
    DepthPEFT Level 1: Standard LoRA + depth-conditioned scalar gating.
    LoRA weights are fixed (learned during training); their OUTPUT is
    gated per-scene by g = sigmoid(w * z_geo + b).
    """

    def __init__(self, base_model, config: GeoLoRAConfig, processor):
        super().__init__()
        self.base = base_model
        self.config = config
        self.processor = processor
        self.lora_rank = _compute_static_lora_rank(config)
        self._wrapped_layers = {}

        self.depth_net = DepthGeometryNet(d_geo=config.d_geo)
        self._wrap_target_layers()

    def _wrap_target_layers(self):
        llm_layers = self.base.model.layers
        for local_idx, global_idx in enumerate(self.config.target_layers):
            layer = llm_layers[global_idx]
            self._wrapped_layers[local_idx] = {}
            for proj_name in self.config.target_projections:
                original_linear = getattr(layer.self_attn, proj_name)
                wrapper = DepthGatedLoRALinear(
                    original_linear, self.lora_rank, self.config.d_geo,
                )
                setattr(layer.self_attn, proj_name, wrapper)
                self._wrapped_layers[local_idx][proj_name] = wrapper

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        config: GeoLoRAConfig,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_name)

        for param in base_model.parameters():
            param.requires_grad = False

        model = cls(base_model, config, processor)
        model.depth_net = model.depth_net.to(dtype=torch_dtype)
        return model

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        depth_maps=None,
        labels=None,
        **kwargs,
    ):
        # Compute z_geo and set it on all wrappers
        if depth_maps is not None:
            z_geo = self.depth_net(depth_maps)
            for lw in self._wrapped_layers.values():
                for wrapper in lw.values():
                    wrapper.set_z_geo(z_geo)

        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

        # Clear z_geo after forward
        for lw in self._wrapped_layers.values():
            for wrapper in lw.values():
                wrapper.clear_z_geo()

        return outputs

    def trainable_parameters(self):
        depth_params = list(self.depth_net.parameters())
        lora_params = []
        gate_params = []
        for lw in self._wrapped_layers.values():
            for wrapper in lw.values():
                lora_params.extend([wrapper.lora_A, wrapper.lora_B])
                gate_params.extend(wrapper.gate_proj.parameters())
        return [
            {"params": depth_params + gate_params, "lr": self.config.lr_depth_router},
            {"params": lora_params, "lr": self.config.lr_lora_bank},
        ]


# ============================================================================
# 5. DepthFiLM — Standard LoRA + depth-conditioned rank-space FiLM (Level 2)
# ============================================================================

class DepthFiLMLoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with LoRA, conditioned by rank-space FiLM.
    z = A @ h  ->  z_cond = gamma * z + beta  ->  delta = B @ z_cond
    gamma, beta in R^r derived from z_geo.
    """

    def __init__(self, frozen_linear: nn.Linear, rank: int, d_geo: int):
        super().__init__()
        self.frozen_linear = frozen_linear
        for p in self.frozen_linear.parameters():
            p.requires_grad = False

        d_in = frozen_linear.in_features
        d_out = frozen_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = 1.0 / rank
        self.rank = rank

        # FiLM: z_geo -> (gamma, beta) in R^r each
        self.film_proj = nn.Linear(d_geo, 2 * rank)
        # Initialize gamma=1, beta=0 (identity FiLM at start)
        nn.init.zeros_(self.film_proj.weight)
        with torch.no_grad():
            self.film_proj.bias[:rank].fill_(1.0)   # gamma init = 1
            self.film_proj.bias[rank:].fill_(0.0)    # beta init = 0

        self._z_geo = None

    def set_z_geo(self, z_geo: torch.Tensor):
        self._z_geo = z_geo

    def clear_z_geo(self):
        self._z_geo = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        frozen_out = self.frozen_linear(h)
        z = F.linear(h, self.lora_A)  # (B, seq, r)

        if self._z_geo is not None:
            film = self.film_proj(self._z_geo)  # (B, 2r)
            gamma = film[:, :self.rank].unsqueeze(1)   # (B, 1, r)
            beta = film[:, self.rank:].unsqueeze(1)     # (B, 1, r)
            z = gamma * z + beta

        delta = F.linear(z, self.lora_B)  # (B, seq, d_out)
        return frozen_out + self.scaling * delta


class Qwen2VLWithDepthFiLM(nn.Module):
    """
    DepthPEFT Level 2: Standard LoRA + rank-space FiLM from depth.
    Richer than DepthGate: per-dimension scale+shift in LoRA's rank space.
    """

    def __init__(self, base_model, config: GeoLoRAConfig, processor):
        super().__init__()
        self.base = base_model
        self.config = config
        self.processor = processor
        self.lora_rank = _compute_static_lora_rank(config)
        self._wrapped_layers = {}

        self.depth_net = DepthGeometryNet(d_geo=config.d_geo)
        self._wrap_target_layers()

    def _wrap_target_layers(self):
        llm_layers = self.base.model.layers
        for local_idx, global_idx in enumerate(self.config.target_layers):
            layer = llm_layers[global_idx]
            self._wrapped_layers[local_idx] = {}
            for proj_name in self.config.target_projections:
                original_linear = getattr(layer.self_attn, proj_name)
                wrapper = DepthFiLMLoRALinear(
                    original_linear, self.lora_rank, self.config.d_geo,
                )
                setattr(layer.self_attn, proj_name, wrapper)
                self._wrapped_layers[local_idx][proj_name] = wrapper

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        config: GeoLoRAConfig,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_name)

        for param in base_model.parameters():
            param.requires_grad = False

        model = cls(base_model, config, processor)
        model.depth_net = model.depth_net.to(dtype=torch_dtype)
        return model

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        depth_maps=None,
        labels=None,
        **kwargs,
    ):
        if depth_maps is not None:
            z_geo = self.depth_net(depth_maps)
            for lw in self._wrapped_layers.values():
                for wrapper in lw.values():
                    wrapper.set_z_geo(z_geo)

        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

        for lw in self._wrapped_layers.values():
            for wrapper in lw.values():
                wrapper.clear_z_geo()

        return outputs

    def trainable_parameters(self):
        depth_params = list(self.depth_net.parameters())
        lora_params = []
        film_params = []
        for lw in self._wrapped_layers.values():
            for wrapper in lw.values():
                lora_params.extend([wrapper.lora_A, wrapper.lora_B])
                film_params.extend(wrapper.film_proj.parameters())
        return [
            {"params": depth_params + film_params, "lr": self.config.lr_depth_router},
            {"params": lora_params, "lr": self.config.lr_lora_bank},
        ]
