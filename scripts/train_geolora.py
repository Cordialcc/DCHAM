"""
Usage: python scripts/train_geolora.py --config configs/geolora.yaml [--method geolora]
"""
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from geolora.config import GeoLoRAConfig
from geolora.model import Qwen2VLWithGeoLoRA
from geolora.baselines import (
    Qwen2VLWithStaticLoRA,
    Qwen2VLWithDepthTokens,
    Qwen2VLWithDepthGate,
    Qwen2VLWithDepthFiLM,
    make_uniform_alpha_geolora,
)
from geolora.dataset import SpatialQADataset
from geolora.collator import SpatialQACollator

METHODS = (
    "geolora", "static_lora", "depth_tokens", "uniform_alpha",
    "question_conditioned", "depth_gate", "depth_film",
)


def build_model(method: str, cfg: dict, geolora_cfg: GeoLoRAConfig):
    """Instantiate the model variant specified by --method."""
    base_name = cfg["model"]["base_model"]
    kwargs = dict(torch_dtype=torch.bfloat16)

    if method == "geolora":
        print("Loading Qwen2.5-VL + GeoLoRA...")
        return Qwen2VLWithGeoLoRA.from_pretrained(base_name, geolora_cfg, **kwargs)

    if method == "static_lora":
        print("Loading Qwen2.5-VL + StaticLoRA...")
        return Qwen2VLWithStaticLoRA.from_pretrained(base_name, geolora_cfg, **kwargs)

    if method == "uniform_alpha":
        print("Loading Qwen2.5-VL + GeoLoRA (uniform alpha)...")
        model = Qwen2VLWithGeoLoRA.from_pretrained(base_name, geolora_cfg, **kwargs)
        patch = make_uniform_alpha_geolora(geolora_cfg)
        patch(model)
        return model

    if method == "depth_tokens":
        print("Loading Qwen2.5-VL + DepthTokens...")
        return Qwen2VLWithDepthTokens.from_pretrained(base_name, geolora_cfg, **kwargs)

    if method == "depth_gate":
        print("Loading Qwen2.5-VL + DepthGate (Level 1)...")
        return Qwen2VLWithDepthGate.from_pretrained(base_name, geolora_cfg, **kwargs)

    if method == "depth_film":
        print("Loading Qwen2.5-VL + DepthFiLM (Level 2)...")
        return Qwen2VLWithDepthFiLM.from_pretrained(base_name, geolora_cfg, **kwargs)

    if method == "question_conditioned":
        print("Loading Qwen2.5-VL + GeoLoRA (question-conditioned router)...")
        qc_cfg = GeoLoRAConfig(
            **{k: getattr(geolora_cfg, k) for k in geolora_cfg.__dataclass_fields__
               if k != "router_type"},
            router_type="question_conditioned",
        )
        return Qwen2VLWithGeoLoRA.from_pretrained(base_name, qc_cfg, **kwargs)

    raise ValueError(f"Unknown method: {method}")


def _has_gates(model) -> bool:
    return hasattr(model, "_wrapped_layers") and isinstance(model, Qwen2VLWithGeoLoRA)


def _log_step(model, method, epoch, global_step, total_steps, loss_val):
    msg = (
        f"Epoch {epoch} Step {global_step}/{total_steps} "
        f"Loss: {loss_val:.4f}"
    )
    if _has_gates(model):
        gate_vals = [
            w.gate.item()
            for lw in model._wrapped_layers.values()
            for w in lw.values()
        ]
        avg_gate = sum(gate_vals) / len(gate_vals)
        msg += f" AvgGate: {avg_gate:.4f}"
    print(msg)


def _save_checkpoint(model, method, output_dir, step):
    save_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)

    if method in ("geolora", "uniform_alpha", "question_conditioned"):
        torch.save(
            {"geolora": model.geolora.state_dict(), "step": step},
            os.path.join(save_dir, "geolora.pt"),
        )
        gates = {
            f"layer_{li}_{pn}": w.gate.data.item()
            for li, lw in model._wrapped_layers.items()
            for pn, w in lw.items()
        }
        torch.save(gates, os.path.join(save_dir, "gates.pt"))

    elif method == "static_lora":
        lora_sd = {}
        for li, lw in model._wrapped_layers.items():
            for pn, wrapper in lw.items():
                lora_sd[f"layer_{li}_{pn}_A"] = wrapper.lora_A.data
                lora_sd[f"layer_{li}_{pn}_B"] = wrapper.lora_B.data
        torch.save(
            {"static_lora": lora_sd, "step": step},
            os.path.join(save_dir, "static_lora.pt"),
        )

    elif method == "depth_tokens":
        torch.save(
            {
                "depth_net": model.depth_net.state_dict(),
                "token_proj": model.token_proj.state_dict(),
                "step": step,
            },
            os.path.join(save_dir, "depth_tokens.pt"),
        )

    elif method in ("depth_gate", "depth_film"):
        # Save depth_net + all wrapped layers (LoRA + conditioning params)
        wrapper_sd = {}
        for li, lw in model._wrapped_layers.items():
            for pn, wrapper in lw.items():
                wrapper_sd[f"layer_{li}_{pn}_A"] = wrapper.lora_A.data
                wrapper_sd[f"layer_{li}_{pn}_B"] = wrapper.lora_B.data
                if method == "depth_gate":
                    wrapper_sd[f"layer_{li}_{pn}_gate"] = wrapper.gate_proj.state_dict()
                else:
                    wrapper_sd[f"layer_{li}_{pn}_film"] = wrapper.film_proj.state_dict()
        torch.save(
            {
                "depth_net": model.depth_net.state_dict(),
                "wrappers": wrapper_sd,
                "step": step,
            },
            os.path.join(save_dir, f"{method}.pt"),
        )

    print(f"Saved {method} checkpoint to {save_dir}")


def main(config_path: str, method: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    geolora_cfg = GeoLoRAConfig(
        **{k: tuple(v) if isinstance(v, list) else v
           for k, v in cfg["model"]["geolora"].items()}
    )

    model = build_model(method, cfg, geolora_cfg)

    train_ds = SpatialQADataset(
        cfg["data"]["train_file"], cfg["data"]["image_dir"],
        cfg["data"]["depth_dir"], model.processor,
        cfg["training"]["max_seq_length"],
    )
    collator = SpatialQACollator(
        pad_token_id=model.processor.tokenizer.pad_token_id or 0
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True,
    )

    param_groups = model.trainable_parameters()
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    accum = cfg["training"]["gradient_accumulation_steps"]
    total_steps = len(train_loader) // accum * cfg["training"]["num_epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = os.path.join(cfg["output"]["output_dir"], method)
    os.makedirs(output_dir, exist_ok=True)

    model.train()
    global_step = 0

    for epoch in range(cfg["training"]["num_epochs"]):
        for step, batch in enumerate(train_loader):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / accum

            loss.backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % cfg["output"]["logging_steps"] == 0:
                    _log_step(
                        model, method, epoch, global_step,
                        total_steps, loss.item() * accum,
                    )

                if global_step % cfg["output"]["save_steps"] == 0:
                    _save_checkpoint(model, method, output_dir, global_step)

    _save_checkpoint(model, method, output_dir, "final")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geolora.yaml")
    parser.add_argument(
        "--method", default="geolora", choices=METHODS,
        help="Model variant to train (default: geolora)",
    )
    args = parser.parse_args()
    main(args.config, args.method)
