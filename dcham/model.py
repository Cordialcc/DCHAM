import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

from .config import DCHAMConfig
from .module import DCHAM


class Qwen2VLWithDCHAM(nn.Module):
    """
    Wraps Qwen2.5-VL with DCHAM module.

    Architecture:
      1. ViT processes image -> visual_tokens (pre-merger captured via hook)
      2. Merger produces merged_visual_tokens (standard path)
      3. DCHAM produces spatial_tokens from pre-merger tokens + depth
      4. spatial_tokens are injected into the LLM input sequence
    """

    def __init__(self, base_model, dcham: DCHAM, processor):
        super().__init__()
        self.base = base_model
        self.dcham = dcham
        self.processor = processor
        self._pre_merger_features = None

        self._hook = self.base.visual.merger.register_forward_pre_hook(
            self._capture_pre_merger
        )

    def _capture_pre_merger(self, module, args):
        self._pre_merger_features = args[0].detach().clone()

    @classmethod
    def from_pretrained(
        cls, model_name: str, dcham_config: DCHAMConfig,
        torch_dtype=torch.bfloat16, device_map="auto",
    ):
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_name)

        # Freeze ViT + Merger
        for param in base_model.visual.parameters():
            param.requires_grad = False

        # Apply LoRA to LLM
        lora_config = LoraConfig(
            r=dcham_config.lora_rank,
            lora_alpha=dcham_config.lora_alpha,
            target_modules=list(dcham_config.lora_target_modules),
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_config)

        dcham = DCHAM(dcham_config).to(dtype=torch_dtype)
        return cls(base_model, dcham, processor)

    def forward(
        self, input_ids, attention_mask, pixel_values, image_grid_thw,
        depth_maps, labels=None,
    ):
        # Step 1: Run visual encoder (hook captures pre-merger features)
        image_embeds = self.base.visual(pixel_values, grid_thw=image_grid_thw)
        pre_merger = self._pre_merger_features
        self._pre_merger_features = None

        # Step 2: Get text embeddings for DCHAM conditioning
        embed_layer = self.base.get_input_embeddings()
        text_embeds = embed_layer(input_ids)

        # Step 3: Run DCHAM per batch item
        B = input_ids.shape[0]
        spatial_tokens_list = []
        offset = 0
        for i in range(B):
            t, h, w = image_grid_thw[i].tolist()
            n_patches = int(t * h * w)
            vis_tok = pre_merger[offset:offset + n_patches].unsqueeze(0)
            offset += n_patches
            st = self.dcham(
                vis_tok, depth_maps[i:i + 1], text_embeds[i:i + 1],
                int(h), int(w),
            )
            spatial_tokens_list.append(st.squeeze(0))

        # Step 4: Build inputs_embeds with spatial tokens prepended
        spatial_stack = torch.stack(spatial_tokens_list)  # R^{B x M x d_lm}

        # Replace image placeholders in text_embeds with image_embeds
        # then prepend spatial tokens
        inputs_embeds = self._replace_image_tokens(
            input_ids, text_embeds, image_embeds, image_grid_thw,
        )
        inputs_embeds = torch.cat([spatial_stack, inputs_embeds], dim=1)

        # Extend attention mask
        M = self.dcham.config.num_queries
        spatial_mask = torch.ones(
            B, M, device=attention_mask.device, dtype=attention_mask.dtype,
        )
        extended_mask = torch.cat([spatial_mask, attention_mask], dim=1)

        # Extend labels (ignore spatial token positions)
        extended_labels = None
        if labels is not None:
            ignore = torch.full(
                (B, M), -100, device=labels.device, dtype=labels.dtype,
            )
            extended_labels = torch.cat([ignore, labels], dim=1)

        # Step 5: Forward through LLM
        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=extended_labels,
        )

    def _replace_image_tokens(self, input_ids, text_embeds, image_embeds, image_grid_thw):
        """
        Replace <|image_pad|> tokens in text_embeds with visual embeddings.

        NOTE: This is a simplified implementation. The exact token IDs for
        <|vision_start|>, <|image_pad|>, <|vision_end|> depend on the
        processor's tokenizer. Refine during server integration testing.
        """
        # Find image_pad token id
        try:
            pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        except Exception:
            # Fallback: return text_embeds as-is if token not found
            return text_embeds

        result = text_embeds.clone()
        img_offset = 0

        for b in range(input_ids.shape[0]):
            mask = input_ids[b] == pad_id
            n_img_tokens = mask.sum().item()

            if n_img_tokens > 0 and img_offset + n_img_tokens <= image_embeds.shape[0]:
                result[b, mask] = image_embeds[img_offset:img_offset + n_img_tokens]
                img_offset += n_img_tokens

        return result
