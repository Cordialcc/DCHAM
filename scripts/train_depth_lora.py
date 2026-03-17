#!/usr/bin/env python3
"""DepthPEFT training: Depth-conditioned LoRA for Qwen2.5-VL.

Minimal fork of the working train_lora.py, adding depth conditioning
via LoRA scaling modification. NO extra nn.Module wrapper — PEFT model
goes directly to Trainer, preserving gradient checkpointing compatibility.

Methods:
  standard_lora: Identical to train_lora.py (baseline, no depth)
  depth_gate: LoRA scaling *= sigmoid(gate(z_geo)) per LoRA module

Usage:
    # Baseline (no depth)
    python scripts/train_depth_lora.py --method standard_lora

    # DepthGate (Level 1)
    python scripts/train_depth_lora.py --method depth_gate \
        --depth-dir spatialqa_gtads/depth_dav2
"""

import json
import argparse
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Sequence

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
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

IGNORE_INDEX = -100


# ─── DepthGeometryNet (standalone, not wrapping anything) ─────────

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

    def forward(self, depth_map):
        grad_x = F.conv2d(depth_map, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth_map, self.sobel_y, padding=1)
        x = torch.cat([depth_map, grad_x, grad_y], dim=1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return F.gelu(self.proj(self.norm(x)))


# ─── Depth Gate modules (per LoRA layer) ──────────────────────────

class DepthGateBank(nn.Module):
    """Collection of per-LoRA-module depth gates.

    For each LoRA module in the PEFT model, stores a Linear(d_geo, 1)
    that produces a sigmoid gate value from z_geo.
    """

    def __init__(self, lora_module_names: list, d_geo: int = 256):
        super().__init__()
        self.gates = nn.ModuleDict({
            name.replace(".", "_"): nn.Linear(d_geo, 1)
            for name in lora_module_names
        })

    def get_gate_values(self, z_geo: torch.Tensor) -> dict:
        """Compute all gate values from z_geo. Returns {safe_name: scalar}."""
        result = {}
        for safe_name, gate_linear in self.gates.items():
            gate = torch.sigmoid(gate_linear(z_geo))  # (B, 1)
            result[safe_name] = gate.mean().item()
        return result


# ─── Lazy Dataset (extends working version with depth) ────────────

class LazyDepthQwenVLDataset(Dataset):
    """Extends LazyQwenVLDataset with depth map loading.

    Identical to the working train_lora.py dataset, plus one extra field:
    'depth_map' tensor loaded from .npy files.
    """

    def __init__(self, data_path: str, processor, max_length: int = 4096,
                 depth_dir: str = None):
        with open(data_path) as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        self.depth_dir = Path(depth_dir) if depth_dir else None
        print(f"LazyDepthQwenVLDataset: {len(self.data)} samples, "
              f"depth={'ON' if self.depth_dir else 'OFF'}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        entry = self.data[idx]
        conversations = entry["conversations"]
        image_path = entry["images"][0]

        pil_image = Image.open(image_path).convert("RGB")

        # Build messages (identical to working script)
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

        # Tokenize (identical to working script)
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

        # === ONLY NEW PART: depth map loading ===
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
        """Identical to working train_lora.py."""
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        im_start_id = 151644
        im_end_id = 151645
        assistant_id = 77091
        newline_id = 198

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


# ─── Data Collator (extends working version with depth) ───────────

@dataclass
class DepthQwenVLDataCollator:
    """Identical to working QwenVLDataCollator, plus depth_map handling."""

    pad_token_id: int
    max_length: int = 4096

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # === IDENTICAL to working collator ===
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        input_ids = input_ids[:, :self.max_length]
        labels = labels[:, :self.max_length]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.pad_token_id),
        }

        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.cat(
                [inst["pixel_values"] for inst in instances], dim=0
            )
        if "image_grid_thw" in instances[0]:
            batch["image_grid_thw"] = torch.cat(
                [inst["image_grid_thw"] for inst in instances], dim=0
            )

        # === ONLY NEW PART: depth map padding ===
        if "depth_map" in instances[0]:
            depths = [inst["depth_map"] for inst in instances]
            max_h = max(d.shape[1] for d in depths)
            max_w = max(d.shape[2] for d in depths)
            batch["depth_map"] = torch.stack([
                F.pad(d, (0, max_w - d.shape[2], 0, max_h - d.shape[1]))
                for d in depths
            ])

        return batch


# ─── Custom Trainer (handles depth_map + gate updates) ────────────

class DepthPEFTTrainer(Trainer):
    """Extends Trainer to handle depth conditioning.

    Before each forward pass, extracts depth_map from inputs,
    runs DepthGeometryNet, and modifies PEFT LoRA scaling.
    After forward, restores original scaling.

    This avoids wrapping the model in an extra nn.Module.
    """

    def __init__(self, *args, depth_net=None, gate_bank=None,
                 lora_modules=None, method="standard_lora", **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_net = depth_net
        self.gate_bank = gate_bank
        self.lora_modules = lora_modules or {}
        self.method = method
        self._original_scalings = {}

    def compute_loss(self, model, inputs, num_items_in_batch=None,
                     return_outputs=False):
        # Extract depth_map before passing to model
        depth_map = inputs.pop("depth_map", None)

        # Apply depth conditioning
        if (depth_map is not None
                and self.depth_net is not None
                and self.method != "standard_lora"):
            z_geo = self.depth_net(depth_map.to(model.device))
            gate_values = self.gate_bank.get_gate_values(z_geo)

            # Modify LoRA scaling in-place
            for name, module in self.lora_modules.items():
                safe_name = name.replace(".", "_")
                if safe_name in gate_values:
                    orig = module.scaling["default"]
                    self._original_scalings[name] = orig
                    module.scaling["default"] = orig * gate_values[safe_name]

        # Standard forward (model is the raw PEFT model)
        outputs = model(**inputs)

        # Restore scaling
        for name, orig in self._original_scalings.items():
            self.lora_modules[name].scaling["default"] = orig
        self._original_scalings = {}

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ─── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="depth_gate",
                        choices=["standard_lora", "depth_gate"])
    parser.add_argument("--model-path", type=str,
                        default="/home/xmu/djd/qwen2.5vl_lora/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train-data", type=str,
                        default="/home/xmu/djd/qwen2.5vl_lora/results/llamafactory_train.json")
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--depth-dir", type=str,
                        default="/home/xmu/djd/qwen2.5vl_lora/spatialqa_gtads/depth_dav2")
    parser.add_argument("--output-dir", type=str,
                        default="/home/xmu/djd/experiments/output/depth_peft")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--min-pixels", type=int, default=3136)
    # LoRA (same defaults as your working script's r=64 run)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-target", type=str,
                        default="q_proj,v_proj,k_proj,o_proj")
    # Training (same defaults as your working script)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"depthpeft_{args.method}"

    output_dir = f"{args.output_dir}/{args.method}"
    if args.dry_run:
        output_dir += "_dryrun"

    # ─── Processor ───
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    print(f"Image resolution: min_pixels={args.min_pixels}, "
          f"max_pixels={args.max_pixels}")

    # ─── Model (identical to working script) ───
    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # ─── LoRA (identical to working script) ───
    target_modules = [m.strip() for m in args.lora_target.split(",")]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Gradient checkpointing (identical to working script)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # ─── Depth conditioning (NEW — only for depth_gate) ───
    depth_net = None
    gate_bank = None
    lora_modules = {}

    if args.method == "depth_gate":
        # Find all LoRA modules
        lora_names = []
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_names.append(name)
                lora_modules[name] = module

        depth_net = DepthGeometryNet(d_geo=256).to(
            dtype=torch.bfloat16, device="cuda"
        )
        gate_bank = DepthGateBank(lora_names, d_geo=256).to(
            dtype=torch.bfloat16, device="cuda"
        )

        # Count params
        depth_params = sum(p.numel() for p in depth_net.parameters())
        gate_params = sum(p.numel() for p in gate_bank.parameters())
        print(f"DepthGate: depth_net={depth_params:,} + "
              f"gate_bank={gate_params:,} = {depth_params+gate_params:,} "
              f"depth-conditioning params")
        print(f"LoRA modules found: {len(lora_names)}")

    # ─── Dataset ───
    depth_dir = args.depth_dir if args.method != "standard_lora" else None
    train_dataset = LazyDepthQwenVLDataset(
        args.train_data, processor, args.max_length, depth_dir=depth_dir,
    )
    val_dataset = None
    if args.val_data:
        val_dataset = LazyDepthQwenVLDataset(
            args.val_data, processor, args.max_length, depth_dir=depth_dir,
        )

    data_collator = DepthQwenVLDataCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        max_length=args.max_length,
    )

    # ─── Training args ───
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_steps=2 if args.dry_run else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none" if args.dry_run else "wandb",
        run_name=args.run_name,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        overwrite_output_dir=True,
    )

    # ─── Train ───
    trainer = DepthPEFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        depth_net=depth_net,
        gate_bank=gate_bank,
        lora_modules=lora_modules,
        method=args.method,
    )

    print(f"Starting training: {args.method}")
    trainer.train()

    # ─── Save ───
    trainer.save_model()
    if depth_net is not None:
        torch.save({
            "depth_net": depth_net.state_dict(),
            "gate_bank": gate_bank.state_dict(),
            "method": args.method,
        }, f"{output_dir}/depth_conditioning.pt")

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
