#!/usr/bin/env python3
"""DepthPEFT: Depth-Aware Visual Token Adapter + Standard LoRA.

Architecture: Depth cross-attention adapter between ViT/Merger and LLM.
Visual tokens are enriched with depth geometry BEFORE entering the LLM.
Standard PEFT LoRA fine-tunes the LLM normally — fully compatible with
gradient checkpointing.

Usage:
    python scripts/train_depth_adapter.py --dry-run
    python scripts/train_depth_adapter.py --epochs 3
"""

import json
import argparse
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, Union, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

import transformers
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLCausalLMOutputWithPast,
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

IGNORE_INDEX = -100


# ─── Depth Encoder ────────────────────────────────────────────────

class DepthEncoder(nn.Module):
    """Encode depth map into features aligned with visual token count.

    Input:  depth_map (B, 1, H, W)
    Output: depth_features (B, N_visual_tokens, d_model)

    Uses a lightweight CNN to produce spatial features, then adaptive
    pooling to match the visual token grid dimensions.
    """

    def __init__(self, d_model: int = 3584, d_mid: int = 256):
        super().__init__()
        # Depth CNN: 1-channel → d_mid features
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=4, padding=3), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, d_mid, 3, stride=2, padding=1), nn.GELU(),
        )
        # Project to model dimension
        self.proj = nn.Linear(d_mid, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, depth_map: torch.Tensor,
                grid_h: int, grid_w: int) -> torch.Tensor:
        """
        Args:
            depth_map: (B, 1, H, W) normalized depth
            grid_h, grid_w: target spatial dims matching visual token grid
        Returns:
            (B, grid_h * grid_w, d_model) depth features
        """
        x = self.conv(depth_map)                          # (B, d_mid, h', w')
        x = F.adaptive_avg_pool2d(x, (grid_h, grid_w))   # (B, d_mid, gh, gw)
        x = x.flatten(2).transpose(1, 2)                  # (B, gh*gw, d_mid)
        x = self.norm(self.proj(x))                       # (B, gh*gw, d_model)
        return x


# ─── Depth-Aware Cross-Attention Adapter ──────────────────────────

class DepthCrossAttentionAdapter(nn.Module):
    """Cross-attention: visual tokens attend to depth features.

    visual_enriched = visual + gate * proj(CrossAttn(Q=visual, K=depth, V=depth))

    Zero-init gate ensures the model starts from base VLM behavior.
    """

    def __init__(self, d_model: int = 3584, n_heads: int = 8, d_depth: int = 3584):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln_v = nn.LayerNorm(d_model)
        self.ln_d = nn.LayerNorm(d_depth)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_depth, d_model, bias=False)
        self.v_proj = nn.Linear(d_depth, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Zero-init gate: model starts identical to base VLM
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, visual: torch.Tensor,
                depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual: (B, N, d_model) visual tokens from ViT+Merger
            depth:  (B, N, d_model) depth features (same spatial count)
        Returns:
            (B, N, d_model) enriched visual tokens
        """
        B, N, D = visual.shape

        q = self.q_proj(self.ln_v(visual))    # (B, N, D)
        k = self.k_proj(self.ln_d(depth))     # (B, N, D)
        v = self.v_proj(self.ln_d(depth))     # (B, N, D)

        # Multi-head attention
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)  # (B, H, N, head_dim)
        attn = attn.transpose(1, 2).reshape(B, N, D)    # (B, N, D)

        delta = self.out_proj(attn)
        return visual + self.gate * delta


# ─── Modified Qwen2.5-VL with Depth Adapter ──────────────────────

class DepthAwareQwen2_5_VLModel(Qwen2_5_VLModel):
    """Qwen2.5-VL with depth-aware visual token enrichment.

    Overrides forward() to:
    1. Process image as normal → image_embeds
    2. Process depth map → depth_features (matched to visual grid)
    3. Cross-attend: image_embeds enriched by depth_features
    4. Scatter enriched embeds into input sequence
    5. Continue with LLM as normal

    The depth adapter operates BEFORE the LLM stack, so gradient
    checkpointing on the LLM layers is fully compatible.
    """

    def __init__(self, config):
        super().__init__(config)
        d = config.hidden_size  # 3584
        self.depth_encoder = DepthEncoder(d_model=d, d_mid=256)
        self.depth_adapter = DepthCrossAttentionAdapter(
            d_model=d, n_heads=8, d_depth=d,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # NEW: depth map input
        depth_map: Optional[torch.Tensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                # Standard: ViT + Merger → image_embeds
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)

                # === NEW: Depth-aware enrichment ===
                if depth_map is not None and image_grid_thw is not None:
                    # Get visual token grid dimensions
                    # image_grid_thw: (num_images, 3) = (t, h, w)
                    # After merger (spatial_merge_size=2): grid = h//2, w//2
                    grid_t, grid_h, grid_w = image_grid_thw[0].tolist()
                    # image_embeds shape: (N_total, d_model) where N_total = t*h*w

                    depth_feats = self.depth_encoder(
                        depth_map, grid_h, grid_w,
                    )  # (B, h*w, d_model)

                    # Reshape image_embeds to (B, N, d_model) for cross-attn
                    # Note: image_embeds is (N_total, d) — need to handle batch
                    N_img = grid_t * grid_h * grid_w
                    B = depth_map.shape[0]

                    if image_embeds.shape[0] == N_img * B:
                        img_3d = image_embeds.view(B, N_img, -1)
                    else:
                        img_3d = image_embeds.unsqueeze(0)  # (1, N, d)

                    # Pad/pool depth_feats if spatial dims don't match exactly
                    if depth_feats.shape[1] != img_3d.shape[1]:
                        depth_feats = F.adaptive_avg_pool1d(
                            depth_feats.transpose(1, 2),
                            img_3d.shape[1]
                        ).transpose(1, 2)

                    # Cross-attention: visual tokens attend to depth
                    img_enriched = self.depth_adapter(img_3d, depth_feats)

                    # Reshape back to flat (N_total, d)
                    image_embeds = img_enriched.reshape(-1, image_embeds.shape[-1])
                # === END NEW ===

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: "
                        f"tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: "
                        f"tokens: {n_video_tokens}, features {n_video_features}"
                    )
                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None and input_ids is not None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts, attention_mask,
            )

        # Standard LLM forward (gradient checkpointing applies here)
        outputs = self.model_forward_without_embedding(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        return outputs

    def model_forward_without_embedding(self, inputs_embeds, **kwargs):
        """Forward through the transformer layers only (skip embedding).

        This reuses the parent class's logic after the embedding step.
        """
        # The parent Qwen2_5_VLModel.forward after embedding assembly
        # calls the decoder layers. We replicate that here.
        hidden_states = inputs_embeds

        # Normalization and decoder layers are in self.layers
        causal_mask = self._update_causal_mask(
            kwargs.get("attention_mask"),
            hidden_states,
            kwargs.get("cache_position"),
            kwargs.get("past_key_values"),
            kwargs.get("output_attentions", False),
        )

        position_embeddings = self.rotary_emb(
            hidden_states, kwargs.get("position_ids")
        )

        all_hidden_states = () if kwargs.get("output_hidden_states") else None
        all_self_attns = () if kwargs.get("output_attentions") else None

        for decoder_layer in self.layers:
            if kwargs.get("output_hidden_states"):
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=kwargs.get("position_ids"),
                past_key_value=kwargs.get("past_key_values"),
                output_attentions=kwargs.get("output_attentions"),
                use_cache=kwargs.get("use_cache"),
                cache_position=kwargs.get("cache_position"),
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            if kwargs.get("output_attentions"):
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if kwargs.get("output_hidden_states"):
            all_hidden_states += (hidden_states,)

        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=kwargs.get("past_key_values"),
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ─── Dataset (identical to working train_lora.py + depth) ────────

class LazyDepthDataset(Dataset):
    """Working train_lora.py dataset + depth map loading."""

    def __init__(self, data_path, processor, max_length=4096, depth_dir=None):
        with open(data_path) as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        self.depth_dir = Path(depth_dir) if depth_dir else None
        print(f"LazyDepthDataset: {len(self.data)} samples, depth={'ON' if self.depth_dir else 'OFF'}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        conversations = entry["conversations"]
        image_path = entry["images"][0]
        pil_image = Image.open(image_path).convert("RGB")

        messages = []
        for i in range(0, len(conversations), 2):
            human = conversations[i]
            gpt = conversations[i + 1] if i + 1 < len(conversations) else None
            if i == 0:
                content = [{"type": "image", "image": pil_image}, {"type": "text", "text": human["value"]}]
            else:
                content = [{"type": "text", "text": human["value"]}]
            messages.append({"role": "user", "content": content})
            if gpt:
                messages.append({"role": "assistant", "content": gpt["value"]})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")

        input_ids = inputs.input_ids[0]
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        labels = self._create_labels(input_ids)

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }
        if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
            result["pixel_values"] = inputs.pixel_values
        if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
            result["image_grid_thw"] = inputs.image_grid_thw

        # Load depth
        if self.depth_dir is not None:
            stem = Path(image_path).stem
            depth_path = self.depth_dir / f"{stem}.npy"
            if depth_path.exists():
                depth = np.load(depth_path).astype(np.float32)
            else:
                depth = np.zeros((pil_image.height, pil_image.width), dtype=np.float32)
            dt = torch.from_numpy(depth).unsqueeze(0)
            dmax = dt.max()
            if dmax > 0:
                dt = dt / dmax
            result["depth_map"] = dt

        return result

    def _create_labels(self, input_ids):
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX
        ids = input_ids.tolist()
        i = 0
        while i < len(ids) - 2:
            if ids[i] == 151644 and ids[i+1] == 77091 and ids[i+2] == 198:
                j = i + 3
                while j < len(ids) and ids[j] != 151645:
                    j += 1
                if j < len(ids):
                    labels[i+3:j+1] = input_ids[i+3:j+1]
                i = j + 1
            else:
                i += 1
        return labels


# ─── Collator (identical to working + depth) ─────────────────────

@dataclass
class DepthCollator:
    pad_token_id: int
    max_length: int = 4096

    def __call__(self, instances):
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.max_length]
        labels = labels[:, :self.max_length]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.pad_token_id),
        }
        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.cat([inst["pixel_values"] for inst in instances], dim=0)
        if "image_grid_thw" in instances[0]:
            batch["image_grid_thw"] = torch.cat([inst["image_grid_thw"] for inst in instances], dim=0)
        if "depth_map" in instances[0]:
            depths = [inst["depth_map"] for inst in instances]
            max_h = max(d.shape[1] for d in depths)
            max_w = max(d.shape[2] for d in depths)
            batch["depth_map"] = torch.stack([
                F.pad(d, (0, max_w - d.shape[2], 0, max_h - d.shape[1]))
                for d in depths
            ])
        return batch


# ─── Custom Trainer (passes depth_map correctly) ─────────────────

class DepthAdapterTrainer(Trainer):
    """Passes depth_map to model.forward via the inner model's forward."""

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # The PEFT model wraps our DepthAwareQwen, so depth_map needs
        # to be passed through. PEFT's forward passes **kwargs through.
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/xmu/djd/qwen2.5vl_lora/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train-data", default="/home/xmu/djd/qwen2.5vl_lora/results/llamafactory_train.json")
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--depth-dir", default="/home/xmu/djd/qwen2.5vl_lora/spatialqa_gtads/depth_dav2")
    parser.add_argument("--output-dir", default="/home/xmu/djd/experiments/output/depth_adapter")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Processor
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )

    # Load model — but replace the inner model class
    print(f"Loading {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Replace the inner model with our depth-aware version
    # Copy weights from original model to new model
    original_state = model.model.state_dict()
    depth_model = DepthAwareQwen2_5_VLModel(model.config)
    depth_model.load_state_dict(original_state, strict=False)
    # Copy the depth adapter (newly initialized)
    model.model = depth_model

    # Count new depth params
    depth_params = sum(
        p.numel() for n, p in model.model.named_parameters()
        if "depth" in n
    )
    print(f"Depth adapter params: {depth_params:,}")

    # Apply LoRA on LLM layers
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Make depth adapter params trainable (PEFT may have frozen them)
    for name, param in model.named_parameters():
        if "depth_encoder" in name or "depth_adapter" in name:
            param.requires_grad = True

    # Gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Dataset
    train_ds = LazyDepthDataset(args.train_data, processor, args.max_length, args.depth_dir)

    collator = DepthCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        max_length=args.max_length,
    )

    # Training
    output_dir = args.output_dir
    if args.dry_run:
        output_dir += "_dryrun"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_steps=2 if args.dry_run else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=500,
        save_total_limit=3,
        report_to="none" if args.dry_run else "wandb",
        run_name="depth_adapter",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        overwrite_output_dir=True,
    )

    trainer = DepthAdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print("Starting training: depth_adapter")
    trainer.train()
    trainer.save_model()

    # Save depth adapter separately
    depth_state = {
        "depth_encoder": model.base_model.model.model.depth_encoder.state_dict(),
        "depth_adapter": model.base_model.model.model.depth_adapter.state_dict(),
    }
    torch.save(depth_state, f"{output_dir}/depth_adapter.pt")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
