"""
Usage: python scripts/evaluate_geolora.py --config configs/geolora.yaml --checkpoint outputs_geolora/final
"""
import argparse
import json
import os

import torch
import yaml
from tqdm import tqdm

from geolora.config import GeoLoRAConfig
from geolora.model import Qwen2VLWithGeoLoRA
from geolora.dataset import SpatialQADataset


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    geolora_cfg = GeoLoRAConfig(
        **{k: tuple(v) if isinstance(v, list) else v
           for k, v in cfg["model"]["geolora"].items()}
    )

    model = Qwen2VLWithGeoLoRA.from_pretrained(
        cfg["model"]["base_model"], geolora_cfg, torch_dtype=torch.bfloat16,
    )

    state = torch.load(
        os.path.join(checkpoint_path, "geolora.pt"), map_location="cpu"
    )
    model.geolora.load_state_dict(state["geolora"])

    gates = torch.load(
        os.path.join(checkpoint_path, "gates.pt"), map_location="cpu"
    )
    for key, val in gates.items():
        parts = key.split("_")
        li = int(parts[1])
        pn = "_".join(parts[2:])
        model._wrapped_layers[li][pn].gate.data.fill_(val)

    model.to("cuda")
    model.eval()

    test_ds = SpatialQADataset(
        cfg["data"]["val_file"], cfg["data"]["image_dir"],
        cfg["data"]["depth_dir"], model.processor,
    )

    results = []
    for i in tqdm(range(len(test_ds))):
        item = test_ds[i]
        inputs = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
                  for k, v in item.items()}
        if "depth_map" in inputs:
            inputs["depth_maps"] = inputs.pop("depth_map")

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)

        results.append({"index": i, "loss": outputs.loss.item()})

    out_path = os.path.join(checkpoint_path, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geolora.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
