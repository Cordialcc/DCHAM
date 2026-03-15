from dataclasses import dataclass


@dataclass
class DCHAMConfig:
    # Qwen2.5-VL-7B dimensions (frozen, reference only)
    d_vit: int = 1280
    d_lm: int = 3584
    vit_patch_size: int = 14

    # DCHAM internal dimensions
    d_model: int = 512      # H * d_head
    d_head: int = 64
    num_heads: int = 8
    num_basis: int = 12     # n: basis kernels
    rank: int = 24          # r: rank per basis
    num_queries: int = 16   # M: spatial query tokens (8 fine + 8 coarse)
    d_depth: int = 128      # depth feature dimension

    # Training
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate_dcham: float = 1e-4
    learning_rate_lora: float = 2e-5
    warmup_ratio: float = 0.05
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
