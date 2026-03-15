"""
Usage: python scripts/train_geolora.py --config configs/geolora.yaml
"""
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from geolora.config import GeoLoRAConfig
from geolora.model import Qwen2VLWithGeoLoRA
from geolora.dataset import SpatialQADataset
from geolora.collator import SpatialQACollator


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    geolora_cfg = GeoLoRAConfig(
        **{k: tuple(v) if isinstance(v, list) else v
           for k, v in cfg["model"]["geolora"].items()}
    )

    print("Loading Qwen2.5-VL + GeoLoRA...")
    model = Qwen2VLWithGeoLoRA.from_pretrained(
        cfg["model"]["base_model"], geolora_cfg, torch_dtype=torch.bfloat16,
    )

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

    model.train()
    global_step = 0
    os.makedirs(cfg["output"]["output_dir"], exist_ok=True)

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
                    gate_vals = [
                        w.gate.item()
                        for lw in model._wrapped_layers.values()
                        for w in lw.values()
                    ]
                    avg_gate = sum(gate_vals) / len(gate_vals)
                    print(
                        f"Epoch {epoch} Step {global_step}/{total_steps} "
                        f"Loss: {loss.item()*accum:.4f} "
                        f"AvgGate: {avg_gate:.4f}"
                    )

                if global_step % cfg["output"]["save_steps"] == 0:
                    _save_checkpoint(model, cfg, global_step)

    _save_checkpoint(model, cfg, "final")
    print("Training complete.")


def _save_checkpoint(model, cfg, step):
    save_dir = os.path.join(cfg["output"]["output_dir"], f"checkpoint-{step}")
    os.makedirs(save_dir, exist_ok=True)
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
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geolora.yaml")
    main(parser.parse_args().config)
