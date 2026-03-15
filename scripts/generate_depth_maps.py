"""
Generate depth maps for NuScenes images using DepthAnythingV2.

Usage:
    python scripts/generate_depth_maps.py \
        --image_dir /path/to/nuscenes/CAM_FRONT \
        --output_dir /path/to/depth_maps
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline


def main(image_dir: str, output_dir: str, model_size: str = "vits"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "depth-estimation",
        model=f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf",
        device=device,
    )
    image_files = (
        sorted(Path(image_dir).glob("*.jpg"))
        + sorted(Path(image_dir).glob("*.png"))
    )

    for img_path in tqdm(image_files, desc="Generating depth maps"):
        out_path = Path(output_dir) / f"{img_path.stem}.npy"
        if out_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        result = pipe(image)
        depth = np.array(result["depth"], dtype=np.float32)
        np.save(out_path, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model_size", default="vits", choices=["vits", "vitb", "vitl"],
    )
    args = parser.parse_args()
    main(args.image_dir, args.output_dir, args.model_size)
