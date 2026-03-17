"""
Analyze GeoLoRA routing behavior: alpha consistency across scenes and question types.

Addresses reviewer concern: "Is a single scene-global z_geo too coarse for
different question types on the same scene?"

Usage:
  python scripts/analyze_routing.py \
    --config configs/geolora.yaml \
    --checkpoint outputs_geolora/final
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from geolora.config import GeoLoRAConfig
from geolora.model import Qwen2VLWithGeoLoRA
from geolora.dataset import SpatialQADataset


def extract_alphas(model, depth_maps):
    """Run depth through GeoLoRA encoder+router and return alpha dicts."""
    with torch.no_grad():
        z_geo = model.geolora.depth_net(depth_maps)
        alphas = model.geolora.router(z_geo)
    return {k: v.cpu().numpy() for k, v in alphas.items()}, z_geo.cpu().numpy()


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
        os.path.join(checkpoint_path, "geolora.pt"), map_location="cpu",
    )
    model.geolora.load_state_dict(state["geolora"])

    gates = torch.load(
        os.path.join(checkpoint_path, "gates.pt"), map_location="cpu",
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

    # ---------- Collect alphas per image and per qa_type ----------
    image_alphas = defaultdict(list)
    image_qa_types = defaultdict(list)
    type_alphas = defaultdict(list)
    all_z_geos = {}

    for i in tqdm(range(len(test_ds)), desc="Extracting alphas"):
        item = test_ds[i]
        raw = test_ds.data[i]
        qa_type = raw.get("qa_type", "unknown")
        image_stem = Path(raw["images"][0]).stem

        depth_map = item["depth_map"].unsqueeze(0).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            alphas, z_geo = extract_alphas(model, depth_map)

        alpha_vec = alphas[0][0]  # layer 0, batch 0, shape (K,)
        image_alphas[image_stem].append(alpha_vec)
        image_qa_types[image_stem].append(qa_type)
        type_alphas[qa_type].append(alpha_vec)
        if image_stem not in all_z_geos:
            all_z_geos[image_stem] = z_geo[0]

    # ---------- Analysis 1: Same-image alpha consistency ----------
    print("\n" + "=" * 60)
    print("Analysis 1: Same-image / different-question alpha consistency")
    print("=" * 60)

    multi_question_images = {k: v for k, v in image_alphas.items() if len(v) > 1}
    print(f"Images with multiple questions: {len(multi_question_images)}")

    intra_image_vars = []
    if multi_question_images:
        for stem, alpha_list in multi_question_images.items():
            stacked = np.stack(alpha_list)
            variance = np.var(stacked, axis=0).mean()
            intra_image_vars.append(variance)

        avg_var = np.mean(intra_image_vars)
        print(f"Average intra-image alpha variance: {avg_var:.8f}")
        if avg_var < 1e-6:
            print("-> Routing IS scene-global: same image always gets same alpha")
            print("   (This confirms the architecture works as designed)")
        else:
            print("-> Unexpected variance! Routing should be deterministic for same depth.")

    # ---------- Analysis 2: Cross-scene alpha diversity ----------
    print("\n" + "=" * 60)
    print("Analysis 2: Cross-scene alpha diversity")
    print("=" * 60)

    all_alpha_vecs = []
    for stem in all_z_geos:
        if image_alphas[stem]:
            all_alpha_vecs.append(image_alphas[stem][0])

    cross_scene_var = None
    if len(all_alpha_vecs) > 1:
        stacked = np.stack(all_alpha_vecs)
        cross_scene_var = np.var(stacked, axis=0)
        print(f"Per-basis variance across scenes: {cross_scene_var}")
        print(f"Mean cross-scene variance: {cross_scene_var.mean():.6f}")

        avg_alpha = np.mean(stacked, axis=0)
        entropy = -np.sum(avg_alpha * np.log(avg_alpha + 1e-8))
        max_entropy = np.log(len(avg_alpha))
        print(f"Average alpha: {avg_alpha}")
        print(f"Entropy: {entropy:.4f} (max: {max_entropy:.4f}, "
              f"ratio: {entropy/max_entropy:.4f})")

    # ---------- Analysis 3: Per-type alpha patterns ----------
    print("\n" + "=" * 60)
    print("Analysis 3: Per question-type alpha patterns")
    print("=" * 60)

    per_type_mean = {}
    for qa_type in sorted(type_alphas.keys()):
        alpha_list = type_alphas[qa_type]
        stacked = np.stack(alpha_list)
        mean_alpha = np.mean(stacked, axis=0)
        per_type_mean[qa_type] = mean_alpha.tolist()
        print(f"  {qa_type:<30s} n={len(alpha_list):>4d}  "
              f"mean_alpha={np.array2string(mean_alpha, precision=3, separator=', ')}")

    # ---------- Save results ----------
    results = {
        "num_images": len(image_alphas),
        "multi_question_images": len(multi_question_images),
        "intra_image_variance": float(np.mean(intra_image_vars)) if intra_image_vars else None,
        "cross_scene_variance": cross_scene_var.tolist() if cross_scene_var is not None else None,
        "per_type_mean_alpha": per_type_mean,
    }

    out_path = os.path.join(checkpoint_path, "routing_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geolora.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
