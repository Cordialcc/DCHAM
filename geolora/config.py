from dataclasses import dataclass


@dataclass
class GeoLoRAConfig:
    # Qwen2.5-VL-7B dimensions (frozen, reference only)
    d_lm: int = 3584
    num_llm_layers: int = 28

    # GeoLoRA dimensions
    d_geo: int = 256           # scene geometry descriptor dimension
    num_bases: int = 6         # K: number of LoRA bases
    lora_rank: int = 16        # r: rank per basis
    target_layers: tuple = (20, 21, 22, 23, 24, 25, 26, 27)  # L=8 upper layers
    target_projections: tuple = ("q_proj", "v_proj")

    # Projection dimensions (Qwen2.5-VL-7B GQA)
    q_proj_in: int = 3584
    q_proj_out: int = 3584
    v_proj_in: int = 3584
    v_proj_out: int = 512      # 4 KV heads x 128 head_dim

    # Training
    lr_depth_router: float = 1e-4
    lr_lora_bank: float = 5e-5
    lr_gates: float = 1e-3
    warmup_ratio: float = 0.05
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8

    def proj_dims(self, proj_name: str) -> tuple:
        """Return (d_in, d_out) for a given projection name."""
        if proj_name == "q_proj":
            return self.q_proj_in, self.q_proj_out
        elif proj_name == "v_proj":
            return self.v_proj_in, self.v_proj_out
        raise ValueError(f"Unknown projection: {proj_name}")
