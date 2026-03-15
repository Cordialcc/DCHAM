"""
Evaluation script for DCHAM + Qwen2.5-VL on SpatialQA.

Usage: python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/final
"""
import argparse
import json

import torch
import yaml
from tqdm import tqdm

from dcham.config import DCHAMConfig
from dcham.model import Qwen2VLWithDCHAM
from dcham.dataset import SpatialQADataset


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dcham_cfg = DCHAMConfig(**cfg["model"]["dcham"])
    dcham_cfg.lora_rank = cfg["model"]["lora"]["rank"]
    dcham_cfg.lora_alpha = cfg["model"]["lora"]["alpha"]

    model = Qwen2VLWithDCHAM.from_pretrained(
        cfg["model"]["base_model"], dcham_cfg, torch_dtype=torch.bfloat16,
    )
    state = torch.load(f"{checkpoint_path}/dcham.pt", map_location="cpu")
    model.dcham.load_state_dict(state["dcham"])
    model.base.load_adapter(checkpoint_path)
    model.to("cuda")
    model.set_mode("eval")

    test_ds = SpatialQADataset(
        cfg["data"]["val_file"], cfg["data"]["image_dir"],
        cfg["data"]["depth_dir"], model.processor,
    )

    results = []
    for i in tqdm(range(len(test_ds))):
        item = test_ds[i]
        inputs = {
            k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
            for k, v in item.items()
        }
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen = model.base.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                max_new_tokens=256,
            )
        pred = model.processor.decode(gen[0], skip_special_tokens=True)
        results.append({"index": i, "prediction": pred})

    out_path = f"{checkpoint_path}/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
