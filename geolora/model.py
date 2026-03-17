import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from .config import GeoLoRAConfig
from .geolora import GeoLoRA
from .injection import DynamicLoRALinear


class Qwen2VLWithGeoLoRA(nn.Module):
    """
    Wraps Qwen2.5-VL with GeoLoRA.

    Architecture:
      1. ViT + Merger produce visual tokens (frozen, standard path)
      2. Depth map -> GeoLoRA -> per-layer LoRA deltas
      3. Target LLM layers have q_proj/v_proj replaced with DynamicLoRALinear
      4. Before each forward pass, deltas are injected into the wrappers
    """

    def __init__(self, base_model, geolora: GeoLoRA, config: GeoLoRAConfig, processor):
        super().__init__()
        self.base = base_model
        self.geolora = geolora
        self.config = config
        self.processor = processor
        self._wrapped_layers = {}

        self._wrap_target_layers()

    def _wrap_target_layers(self):
        """Replace q_proj/v_proj in target layers with DynamicLoRALinear."""
        llm_layers = self.base.model.layers
        for local_idx, global_idx in enumerate(self.config.target_layers):
            layer = llm_layers[global_idx]
            self._wrapped_layers[local_idx] = {}
            for proj_name in self.config.target_projections:
                original_linear = getattr(layer.self_attn, proj_name)
                wrapper = DynamicLoRALinear(original_linear)
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
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_name)

        for param in base_model.parameters():
            param.requires_grad = False

        geolora = GeoLoRA(config).to(dtype=torch_dtype)
        return cls(base_model, geolora, config, processor)

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
            z_geo = self.geolora.depth_net(depth_maps)

            # Question-conditioned router needs text embeddings
            if self.config.router_type == "question_conditioned":
                with torch.no_grad():
                    text_embeds = self.base.model.embed_tokens(input_ids)
                    q_embed = text_embeds.mean(dim=1)  # (B, d_lm)
                alphas = self.geolora.router(z_geo, q_embed=q_embed)
            else:
                alphas = self.geolora.router(z_geo)

            hooks = []
            for local_idx, global_idx in enumerate(self.config.target_layers):
                layer = self.base.model.layers[global_idx]

                def make_pre_hook(li, alpha):
                    def hook_fn(module, args):
                        h = args[0]
                        for proj_name in self.config.target_projections:
                            bank = self.geolora.banks[str(li)][proj_name]
                            delta = bank(h, alpha)
                            self._wrapped_layers[li][proj_name].set_delta(delta)
                        return args
                    return hook_fn

                hook = layer.register_forward_pre_hook(
                    make_pre_hook(local_idx, alphas[local_idx])
                )
                hooks.append(hook)
        else:
            hooks = []

        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

        for hook in hooks:
            hook.remove()
        for layer_wrappers in self._wrapped_layers.values():
            for wrapper in layer_wrappers.values():
                wrapper.clear_delta()

        return outputs

    def trainable_parameters(self):
        """Return parameter groups with different learning rates."""
        depth_router_params = []
        bank_params = []
        gate_params = []

        for name, param in self.geolora.named_parameters():
            if param.requires_grad:
                if "depth_net" in name or "router" in name:
                    depth_router_params.append(param)
                else:
                    bank_params.append(param)

        for layer_wrappers in self._wrapped_layers.values():
            for wrapper in layer_wrappers.values():
                gate_params.append(wrapper.gate)

        return [
            {"params": depth_router_params, "lr": self.config.lr_depth_router},
            {"params": bank_params, "lr": self.config.lr_lora_bank},
            {"params": gate_params, "lr": self.config.lr_gates},
        ]
