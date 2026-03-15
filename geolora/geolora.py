import torch
import torch.nn as nn

from .config import GeoLoRAConfig
from .depth_geometry import DepthGeometryNet
from .lora_bank import LoRABasisBank
from .router import GeometryRouter


class GeoLoRA(nn.Module):
    """
    Geometry-Conditioned Dynamic LoRA.
    Orchestrates: DepthGeometryNet -> GeometryRouter -> LoRABasisBank
    Produces per-layer, per-projection LoRA deltas from a depth map.
    """

    def __init__(self, config: GeoLoRAConfig):
        super().__init__()
        self.config = config
        num_layers = len(config.target_layers)

        self.depth_net = DepthGeometryNet(d_geo=config.d_geo)

        self.router = GeometryRouter(
            d_geo=config.d_geo,
            num_bases=config.num_bases,
            num_layers=num_layers,
        )

        self.banks = nn.ModuleDict()
        for i in range(num_layers):
            layer_banks = nn.ModuleDict()
            for proj in config.target_projections:
                d_in, d_out = config.proj_dims(proj)
                layer_banks[proj] = LoRABasisBank(
                    num_bases=config.num_bases,
                    lora_rank=config.lora_rank,
                    d_in=d_in,
                    d_out=d_out,
                )
            self.banks[str(i)] = layer_banks

    def compute_deltas(
        self,
        depth_map: torch.Tensor,
        hidden_states: dict,
    ) -> dict:
        z_geo = self.depth_net(depth_map)
        alphas = self.router(z_geo)

        deltas = {}
        for i, alpha in alphas.items():
            h = hidden_states[i]
            layer_deltas = {}
            for proj in self.config.target_projections:
                bank = self.banks[str(i)][proj]
                layer_deltas[proj] = bank(h, alpha)
            deltas[i] = layer_deltas
        return deltas
