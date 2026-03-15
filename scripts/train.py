"""
Training script for DCHAM + Qwen2.5-VL.

Usage: python scripts/train.py --config configs/default.yaml
"""
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from dcham.config import DCHAMConfig
from dcham.model import Qwen2VLWithDCHAM
from dcham.dataset import SpatialQADataset
from dcham.collator import SpatialQACollator


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dcham_cfg = DCHAMConfig(**cfg["model"]["dcham"])
    dcham_cfg.lora_rank = cfg["model"]["lora"]["rank"]
    dcham_cfg.lora_alpha = cfg["model"]["lora"]["alpha"]

    print("Loading Qwen2.5-VL + DCHAM...")
    model = Qwen2VLWithDCHAM.from_pretrained(
        cfg["model"]["base_model"], dcham_cfg, torch_dtype=torch.bfloat16,
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

    # Dual learning rate: DCHAM vs LoRA
    dcham_params = list(model.dcham.parameters())
    lora_params = [
        p for n, p in model.base.named_parameters()
        if p.requires_grad and "lora" in n
    ]
    optimizer = torch.optim.AdamW([
        {"params": dcham_params, "lr": cfg["training"]["learning_rate_dcham"]},
        {"params": lora_params, "lr": cfg["training"]["learning_rate_lora"]},
    ], weight_decay=0.01)

    accum = cfg["training"]["gradient_accumulation_steps"]
    total_steps = len(train_loader) // accum * cfg["training"]["num_epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    global_step = 0
    os.makedirs(cfg["output"]["output_dir"], exist_ok=True)

    for epoch in range(cfg["training"]["num_epochs"]):
        for step, batch in enumerate(train_loader):
            batch = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / accum

            loss.backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % cfg["output"]["logging_steps"] == 0:
                    print(
                        f"Epoch {epoch} Step {global_step}/{total_steps} "
                        f"Loss: {loss.item() * accum:.4f}"
                    )

                if global_step % cfg["output"]["save_steps"] == 0:
                    save_dir = f"{cfg['output']['output_dir']}/checkpoint-{global_step}"
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(
                        {"dcham": model.dcham.state_dict(), "step": global_step},
                        f"{save_dir}/dcham.pt",
                    )
                    model.base.save_pretrained(save_dir)
                    print(f"Saved to {save_dir}")

    save_dir = f"{cfg['output']['output_dir']}/final"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({"dcham": model.dcham.state_dict()}, f"{save_dir}/dcham.pt")
    model.base.save_pretrained(save_dir)
    print(f"Training complete. Final model at {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
