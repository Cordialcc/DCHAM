#!/usr/bin/env python3
"""DepthPEFT: Depth-Conditioned LoRA training for Qwen2.5-VL.

Builds on the existing train_lora.py pipeline, adding depth conditioning
from DepthAnythingV2 depth maps. Three progressive conditioning levels:
  - depth_gate: Standard LoRA + depth-conditioned scalar gating
  - depth_film: Standard LoRA + depth-conditioned rank-space FiLM
  - depth_basis: Basis LoRA mixing conditioned on depth (GeoLoRA)

Usage:
    # DepthGate (Level 1, recommended to run first)
    python scripts/train_depth_peft.py --method depth_gate --depth_dir spatialqa_gtads/depth_dav2

    # DepthFiLM (Level 2)
    python scripts/train_depth_peft.py --method depth_film --depth_dir spatialqa_gtads/depth_dav2

    # Standard LoRA baseline (no depth, for fair comparison with same script)
    python scripts/train_depth_peft.py --method standard_lora --depth_dir spatialqa_gtads/depth_dav2
"""

import json
import argparse
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

import transformers
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

IGNORE_INDEX = -100


# ─── DepthGeometryNet ──────────────────────────────────────────────

class DepthGeometryNet(nn.Module):
    """Depth map -> scene geometry descriptor z_geo in R^{d_geo}."""

    def __init__(self, d_geo: int = 256):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

        self.conv1 = nn.Conv2d(3, 64, 7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(256)
        self.proj = nn.Linear(256, d_geo)

    def forward(self, depth_map: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(depth_map, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth_map, self.sobel_y, padding=1)
        x = torch.cat([depth_map, grad_x, grad_y], dim=1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return F.gelu(self.proj(self.norm(x)))


# ─── Dataset with Depth ────────────────────────────────────────────

class LazyDepthQwenVLDataset(Dataset):
    """Extends the existing lazy dataset with depth map loading."""

    def __init__(self, data_path: str, depth_dir: str, processor,
                 max_length: int = 4096):
        with open(data_path) as f:
            self.data = json.load(f)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.processor = processor
        self.max_length = max_length
        print(f"LazyDepthQwenVLDataset: {len(self.data)} samples, "
              f"depth_dir={'set' if self.depth_dir else 'none'}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        entry = self.data[idx]
        conversations = entry["conversations"]
        image_path = entry["images"][0]

        pil_image = Image.open(image_path).convert("RGB")

        # Build messages
        messages = []
        for i in range(0, len(conversations), 2):
            human = conversations[i]
            gpt = conversations[i + 1] if i + 1 < len(conversations) else None

            if i == 0:
                content = [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": human["value"]},
                ]
            else:
                content = [{"type": "text", "text": human["value"]}]

            messages.append({"role": "user", "content": content})
            if gpt:
                messages.append({"role": "assistant", "content": gpt["value"]})

        # Tokenize
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt",
        )

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

        # Load depth map
        if self.depth_dir is not None:
            stem = Path(image_path).stem
            depth_path = self.depth_dir / f"{stem}.npy"
            if depth_path.exists():
                depth = np.load(depth_path).astype(np.float32)
            else:
                depth = np.zeros(
                    (pil_image.height, pil_image.width), dtype=np.float32
                )
            depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
            dmax = depth_tensor.max()
            if dmax > 0:
                depth_tensor = depth_tensor / dmax
            result["depth_map"] = depth_tensor

        return result

    def _create_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mask all non-assistant tokens with IGNORE_INDEX.

        Pattern: everything between '<|im_start|>assistant\\n' and '<|im_end|>'
        is kept as training target; everything else is masked.

        Uses hardcoded token IDs for Qwen2.5-VL's tokenizer (the
        convert_tokens_to_ids API doesn't handle these special tokens correctly).
        """
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX  # mask everything first

        im_start_id = 151644  # <|im_start|>
        im_end_id = 151645    # <|im_end|>
        assistant_id = 77091  # 'assistant'
        newline_id = 198      # '\n'

        ids = input_ids.tolist()
        i = 0
        while i < len(ids) - 2:
            if (ids[i] == im_start_id
                    and ids[i + 1] == assistant_id
                    and ids[i + 2] == newline_id):
                j = i + 3
                while j < len(ids) and ids[j] != im_end_id:
                    j += 1
                if j < len(ids):
                    labels[i + 3 : j + 1] = input_ids[i + 3 : j + 1]
                i = j + 1
            else:
                i += 1

        return labels


# ─── Depth-Conditioned Model Wrapper ───────────────────────────────

class DepthConditionedQwen(nn.Module):
    """Wraps a PEFT Qwen2.5-VL model with depth conditioning.

    The base model uses standard PEFT LoRA. This wrapper adds:
    - DepthGeometryNet to extract z_geo from depth maps
    - Per-layer depth conditioning applied via hooks on LoRA outputs

    method:
      'depth_gate': Scalar gating of LoRA output per layer
      'depth_film': Rank-space FiLM on LoRA output per layer
      'standard_lora': No depth conditioning (pure baseline)
    """

    def __init__(self, peft_model, processor, method="depth_gate",
                 d_geo=256, target_modules=None):
        super().__init__()
        self.model = peft_model
        self.processor = processor
        self.method = method
        self.d_geo = d_geo

        if method == "standard_lora":
            self.depth_net = None
            return

        self.depth_net = DepthGeometryNet(d_geo=d_geo)

        # Find LoRA modules and get their ranks
        self._conditioning_modules = nn.ModuleDict()
        lora_rank = None
        for name, module in peft_model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # This is a PEFT LoRA layer
                if lora_rank is None:
                    # Get rank from the first LoRA A weight
                    for k, v in module.lora_A.items():
                        lora_rank = v.weight.shape[0]
                        break

                safe_name = name.replace(".", "_")
                if method == "depth_gate":
                    self._conditioning_modules[safe_name] = nn.Linear(d_geo, 1)
                elif method == "depth_film":
                    proj = nn.Linear(d_geo, 2 * lora_rank)
                    # Init: gamma=1, beta=0 (identity)
                    nn.init.zeros_(proj.weight)
                    with torch.no_grad():
                        proj.bias[:lora_rank].fill_(1.0)
                        proj.bias[lora_rank:].fill_(0.0)
                    self._conditioning_modules[safe_name] = proj

        # Device placement handled by Trainer

        self.lora_rank = lora_rank
        self._z_geo = None
        self._hooks = []
        self._lora_name_map = {}

        # Build name mapping for hooks
        for name, module in peft_model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                safe_name = name.replace(".", "_")
                self._lora_name_map[id(module)] = safe_name

        print(f"DepthConditionedQwen: method={method}, "
              f"lora_rank={lora_rank}, "
              f"conditioning_modules={len(self._conditioning_modules)}")

    def _apply_depth_conditioning(self, z_geo):
        """Modify PEFT LoRA scaling based on depth — zero extra memory.

        For depth_gate: each LoRA module's scaling *= sigmoid(gate(z_geo)).
        Modifies scaling in-place before forward, restores after.
        Works with batch_size=1 (scaling is a scalar, not per-token).
        """
        self._original_scalings = {}

        for name, module in self.model.named_modules():
            if id(module) not in self._lora_name_map:
                continue

            safe_name = self._lora_name_map[id(module)]
            cond_module = self._conditioning_modules[safe_name]
            adapter_name = "default"

            if self.method == "depth_gate":
                gate = torch.sigmoid(cond_module(z_geo))  # (B, 1)
                gate_val = gate.mean().item()
                orig = module.scaling[adapter_name]
                self._original_scalings[id(module)] = orig
                module.scaling[adapter_name] = orig * gate_val

            elif self.method == "depth_film":
                # FiLM: compute a scalar modulation from the mean of gamma
                film = cond_module(z_geo)  # (B, 2r)
                gamma_mean = film[:, :self.lora_rank].mean().item()
                orig = module.scaling[adapter_name]
                self._original_scalings[id(module)] = orig
                module.scaling[adapter_name] = orig * gamma_mean

    def _restore_scaling(self):
        """Restore original PEFT scaling factors."""
        for name, module in self.model.named_modules():
            if id(module) in self._original_scalings:
                module.scaling["default"] = self._original_scalings[id(module)]
        self._original_scalings = {}

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                pixel_values=None, image_grid_thw=None, depth_map=None,
                **kwargs):
        # Apply depth conditioning (zero extra memory)
        if depth_map is not None and self.depth_net is not None:
            z_geo = self.depth_net(depth_map)
            self._apply_depth_conditioning(z_geo)

        # Filter kwargs that the base model doesn't accept
        kwargs.pop("num_items_in_batch", None)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

        if self.depth_net is not None:
            self._restore_scaling()

        return outputs

    def trainable_param_count(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for n, p in self.model.named_parameters()
                   if p.requires_grad)
        depth = sum(p.numel() for p in self.depth_net.parameters()) if self.depth_net else 0
        cond = sum(p.numel() for p in self._conditioning_modules.parameters()) if hasattr(self, "_conditioning_modules") else 0
        return {"total": total, "lora": lora, "depth_net": depth, "conditioning": cond}


# ─── Data Collator ─────────────────────────────────────────────────

@dataclass
class DepthCollator:
    """Collate function that handles variable-length sequences + depth maps."""
    pad_token_id: int = 0

    def __call__(self, features):
        max_len = max(f["input_ids"].shape[0] for f in features)

        batch = {
            "input_ids": torch.stack([
                F.pad(f["input_ids"], (0, max_len - f["input_ids"].shape[0]),
                      value=self.pad_token_id)
                for f in features
            ]),
            "attention_mask": torch.stack([
                F.pad(f["attention_mask"],
                      (0, max_len - f["attention_mask"].shape[0]), value=0)
                for f in features
            ]),
            "labels": torch.stack([
                F.pad(f["labels"], (0, max_len - f["labels"].shape[0]),
                      value=IGNORE_INDEX)
                for f in features
            ]),
        }

        # Pixel values: variable shape, just concatenate
        if "pixel_values" in features[0] and features[0]["pixel_values"] is not None:
            batch["pixel_values"] = torch.cat(
                [f["pixel_values"] for f in features]
            )
        if "image_grid_thw" in features[0] and features[0]["image_grid_thw"] is not None:
            batch["image_grid_thw"] = torch.cat(
                [f["image_grid_thw"] for f in features]
            )

        # Depth maps: pad to max spatial size
        if "depth_map" in features[0]:
            depths = [f["depth_map"] for f in features]
            max_h = max(d.shape[1] for d in depths)
            max_w = max(d.shape[2] for d in depths)
            batch["depth_map"] = torch.stack([
                F.pad(d, (0, max_w - d.shape[2], 0, max_h - d.shape[1]))
                for d in depths
            ])

        return batch


# ─── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="depth_gate",
                        choices=["standard_lora", "depth_gate", "depth_film"])
    parser.add_argument("--model_path", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_data",
                        default="results/llamafactory_train.json")
    parser.add_argument("--val_data",
                        default="results/llamafactory_val.json")
    parser.add_argument("--depth_dir", default="spatialqa_gtads/depth_dav2")
    parser.add_argument("--output_dir", default="output/depth_peft")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading {args.model_path}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Note: gradient checkpointing conflicts with forward hooks used for
    # depth conditioning. Memory saved by reducing max_length instead.

    # Apply PEFT LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
    )
    peft_model = get_peft_model(base_model, lora_config)

    # Wrap with depth conditioning
    model = DepthConditionedQwen(
        peft_model, processor, method=args.method, d_geo=256,
    )

    param_counts = model.trainable_param_count()
    print(f"Method: {args.method}")
    print(f"Trainable params: {param_counts}")

    # Dataset
    depth_dir = args.depth_dir if args.method != "standard_lora" else None
    train_ds = LazyDepthQwenVLDataset(
        args.train_data, depth_dir, processor, args.max_length,
    )
    val_ds = LazyDepthQwenVLDataset(
        args.val_data, depth_dir, processor, args.max_length,
    )

    collator = DepthCollator(
        pad_token_id=processor.tokenizer.pad_token_id or 0
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if args.dry_run else args.epochs,
        max_steps=2 if args.dry_run else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        dataloader_num_workers=2,
        report_to="wandb" if not args.dry_run else "none",
        run_name=f"depthpeft_{args.method}",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print(f"Starting training: {args.method}")
    trainer.train()

    # Save
    model.model.save_pretrained(output_dir)
    if model.depth_net is not None:
        torch.save({
            "depth_net": model.depth_net.state_dict(),
            "conditioning": model._conditioning_modules.state_dict(),
            "method": args.method,
            "lora_rank": model.lora_rank,
        }, os.path.join(output_dir, "depth_conditioning.pt"))

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
