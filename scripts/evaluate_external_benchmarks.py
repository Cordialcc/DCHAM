"""
Evaluate GeoLoRA on external spatial reasoning benchmarks.
Addresses reviewer concern about self-contained evaluation (P1+P2+P3 all from same thesis).

Supported benchmarks:
  - CV-Bench spatial subset (distance, spatial relation questions)
  - SpatialBench (if available)

Supported methods:
  - geolora: Full GeoLoRA with depth-conditioned dynamic LoRA hooks
  - static_lora: Static LoRA baseline (no depth conditioning, no hooks needed)
  - depth_tokens: Depth tokens appended to input (handled in forward, no hooks needed)

Usage:
  python scripts/evaluate_external_benchmarks.py \
    --config configs/geolora.yaml \
    --checkpoint outputs_geolora/final \
    --benchmark cv-bench \
    --depth-dir /path/to/depth_maps \
    --method geolora
"""
import argparse
import json
import os
import re

import numpy as np
import torch
import yaml
from tqdm import tqdm
from PIL import Image

from geolora.config import GeoLoRAConfig
from geolora.model import Qwen2VLWithGeoLoRA
from geolora.baselines import Qwen2VLWithStaticLoRA, Qwen2VLWithDepthTokens


# --------------------------------------------------------------------------- #
# CV-Bench loader
# --------------------------------------------------------------------------- #

def load_cvbench_spatial(data_dir: str):
    """
    Load spatial subset of CV-Bench.
    CV-Bench contains distance estimation and spatial relation questions
    with images and ground truth answers.

    Expected structure:
      data_dir/
        spatial/
          metadata.json   (or questions.jsonl)
          images/
    """
    meta_path = os.path.join(data_dir, "spatial", "metadata.json")
    jsonl_path = os.path.join(data_dir, "spatial", "questions.jsonl")

    samples = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            samples = json.load(f)
    elif os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            samples = [json.loads(line) for line in f]
    else:
        raise FileNotFoundError(
            f"CV-Bench spatial data not found at {data_dir}/spatial/. "
            "Download from https://github.com/nyu-visionx/CV-Bench"
        )

    image_dir = os.path.join(data_dir, "spatial", "images")
    for s in samples:
        s["image_path"] = os.path.join(
            image_dir, s.get("image", s.get("image_id", "") + ".jpg")
        )
    return samples


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def build_prompt(question: str, choices: list = None) -> list:
    """Build a chat-template message list for a spatial question."""
    content = [{"type": "image"}]
    if choices:
        options = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
        content.append({"type": "text", "text": f"{question}\n{options}\nAnswer:"})
    else:
        content.append({"type": "text", "text": f"{question}\nAnswer:"})
    return [{"role": "user", "content": content}]


def extract_answer(generated: str) -> str:
    """Extract the answer letter or short answer from generated text."""
    generated = generated.strip()
    match = re.match(r"^\(?([A-D])\)?", generated)
    if match:
        return match.group(1)
    return generated.split("\n")[0].strip()


def load_depth_for_image(image_path: str, depth_dir: str, image_size: tuple):
    """Load depth map for an external benchmark image."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    depth_path = os.path.join(depth_dir, f"{stem}.npy")

    if os.path.exists(depth_path):
        depth = np.load(depth_path).astype(np.float32)
    else:
        depth = np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # 1 x H x W
    depth_max = depth_tensor.max()
    if depth_max > 0:
        depth_tensor = depth_tensor / depth_max
    return depth_tensor.unsqueeze(0)  # 1 x 1 x H x W


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #

def _register_geolora_hooks(model, depth_maps, input_ids=None):
    """
    Register GeoLoRA forward pre-hooks on target layers so that
    depth-conditioned LoRA deltas are active during generation.

    Returns a list of hook handles to be removed after generation.
    """
    z_geo = model.geolora.depth_net(depth_maps)

    # Question-conditioned router needs text embeddings
    if model.config.router_type == "question_conditioned" and input_ids is not None:
        with torch.no_grad():
            text_embeds = model.base.model.embed_tokens(input_ids)
            q_embed = text_embeds.mean(dim=1)
        alphas = model.geolora.router(z_geo, q_embed=q_embed)
    else:
        alphas = model.geolora.router(z_geo)

    hooks = []
    for local_idx, global_idx in enumerate(model.config.target_layers):
        layer = model.base.model.layers[global_idx]

        def make_pre_hook(li, alpha):
            def hook_fn(module, args):
                h = args[0]
                for proj_name in model.config.target_projections:
                    bank = model.geolora.banks[str(li)][proj_name]
                    delta = bank(h, alpha)
                    model._wrapped_layers[li][proj_name].set_delta(delta)
                return args
            return hook_fn

        hook = layer.register_forward_pre_hook(
            make_pre_hook(local_idx, alphas[local_idx])
        )
        hooks.append(hook)
    return hooks


def _cleanup_hooks(model, hooks):
    """Remove hooks and clear all LoRA deltas from wrapped layers."""
    for hook in hooks:
        hook.remove()
    for layer_wrappers in model._wrapped_layers.values():
        for wrapper in layer_wrappers.values():
            wrapper.clear_delta()


def evaluate_cvbench(model, samples, depth_dir, device="cuda", method="geolora"):
    """Run evaluation on CV-Bench spatial subset."""
    results = []
    correct = 0
    total = 0

    for sample in tqdm(samples, desc="CV-Bench Spatial"):
        image_path = sample["image_path"]
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert("RGB")
        question = sample.get("question", sample.get("prompt", ""))
        choices = sample.get("choices", sample.get("options", None))
        gt_answer = sample.get("answer", sample.get("ground_truth", ""))

        messages = build_prompt(question, choices)
        text = model.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = model.processor(
            text=[text], images=[image], return_tensors="pt",
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Load depth map (used by geolora hooks and depth_tokens input)
        depth_maps = load_depth_for_image(image_path, depth_dir, image.size)
        depth_maps = depth_maps.to(device, dtype=torch.bfloat16)

        # Register GeoLoRA hooks before generation so depth conditioning is active
        hooks = []
        if method in ("geolora", "uniform_alpha", "question_conditioned"):
            hooks = _register_geolora_hooks(
                model, depth_maps, input_ids=inputs.get("input_ids"),
            )

        # Build generation inputs depending on method
        gen_inputs = {k: v for k, v in inputs.items()}

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if method == "depth_tokens":
                # generate() doesn't call the wrapper's forward(), so we must
                # manually construct inputs_embeds with depth tokens prepended.
                inputs_embeds = model.base.model.embed_tokens(gen_inputs["input_ids"])
                z_geo = model.depth_net(depth_maps)
                depth_tokens = model.token_proj(z_geo)  # (1, N_depth, d_lm)
                inputs_embeds = torch.cat([depth_tokens, inputs_embeds], dim=1)

                batch = gen_inputs["attention_mask"].size(0)
                depth_mask = torch.ones(
                    batch, model.n_depth_tokens,
                    device=gen_inputs["attention_mask"].device,
                    dtype=gen_inputs["attention_mask"].dtype,
                )
                gen_inputs["attention_mask"] = torch.cat(
                    [depth_mask, gen_inputs["attention_mask"]], dim=1,
                )
                del gen_inputs["input_ids"]
                gen_inputs["inputs_embeds"] = inputs_embeds
            # static_lora: no hooks, no depth_maps -- just the base + static LoRA weights

            generated_ids = model.base.generate(**gen_inputs, max_new_tokens=64)

        # Clean up hooks and deltas
        if hooks:
            _cleanup_hooks(model, hooks)

        input_len = inputs["input_ids"].shape[1]
        if method == "depth_tokens":
            # generate() output includes positions for the prepended depth tokens
            input_len += model.n_depth_tokens
        generated_text = model.processor.tokenizer.decode(
            generated_ids[0][input_len:], skip_special_tokens=True,
        )

        pred = extract_answer(generated_text)
        is_correct = pred.upper() == str(gt_answer).upper()
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "gt_answer": str(gt_answer),
            "predicted": pred,
            "generated_full": generated_text,
            "correct": is_correct,
            "category": sample.get("category", sample.get("type", "spatial")),
        })

    accuracy = correct / total if total > 0 else 0.0

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1

    category_acc = {
        cat: v["correct"] / v["total"] for cat, v in categories.items()
    }

    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_category": category_acc,
        "results": results,
    }


def main(config_path: str, checkpoint_path: str, benchmark: str,
         benchmark_dir: str, depth_dir: str, method: str = "geolora"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    geolora_cfg = GeoLoRAConfig(
        **{k: tuple(v) if isinstance(v, list) else v
           for k, v in cfg["model"]["geolora"].items()}
    )

    base_name = cfg["model"]["base_model"]
    kwargs = dict(torch_dtype=torch.bfloat16)

    if method == "geolora":
        print("Loading Qwen2.5-VL + GeoLoRA...")
        model = Qwen2VLWithGeoLoRA.from_pretrained(base_name, geolora_cfg, **kwargs)
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

    elif method == "static_lora":
        print("Loading Qwen2.5-VL + StaticLoRA...")
        model = Qwen2VLWithStaticLoRA.from_pretrained(base_name, geolora_cfg, **kwargs)
        state = torch.load(
            os.path.join(checkpoint_path, "static_lora.pt"), map_location="cpu",
        )
        lora_sd = state["static_lora"]
        for li, lw in model._wrapped_layers.items():
            for pn, wrapper in lw.items():
                wrapper.lora_A.data.copy_(lora_sd[f"layer_{li}_{pn}_A"])
                wrapper.lora_B.data.copy_(lora_sd[f"layer_{li}_{pn}_B"])

    elif method == "depth_tokens":
        print("Loading Qwen2.5-VL + DepthTokens...")
        model = Qwen2VLWithDepthTokens.from_pretrained(base_name, geolora_cfg, **kwargs)
        state = torch.load(
            os.path.join(checkpoint_path, "depth_tokens.pt"), map_location="cpu",
        )
        model.depth_net.load_state_dict(state["depth_net"])
        model.token_proj.load_state_dict(state["token_proj"])

    elif method == "uniform_alpha":
        print("Loading Qwen2.5-VL + GeoLoRA (uniform alpha)...")
        model = Qwen2VLWithGeoLoRA.from_pretrained(base_name, geolora_cfg, **kwargs)
        from geolora.baselines import make_uniform_alpha_geolora
        patch = make_uniform_alpha_geolora(geolora_cfg)
        patch(model)
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

    elif method == "question_conditioned":
        print("Loading Qwen2.5-VL + GeoLoRA (question-conditioned)...")
        qc_cfg = GeoLoRAConfig(
            **{k: getattr(geolora_cfg, k) for k in geolora_cfg.__dataclass_fields__
               if k != "router_type"},
            router_type="question_conditioned",
        )
        model = Qwen2VLWithGeoLoRA.from_pretrained(base_name, qc_cfg, **kwargs)
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

    else:
        raise ValueError(f"Unknown method: {method}")

    model.to("cuda")
    model.eval()

    if benchmark == "cv-bench":
        samples = load_cvbench_spatial(benchmark_dir)
        results = evaluate_cvbench(model, samples, depth_dir, method=method)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    out_path = os.path.join(
        checkpoint_path, f"{benchmark}_{method}_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Benchmark: {benchmark} | Method: {method}")
    print(f"Overall accuracy: {results['overall_accuracy']:.4f} "
          f"({results['correct']}/{results['total']})")
    if results.get("per_category"):
        print("Per-category:")
        for cat, acc in sorted(results["per_category"].items()):
            print(f"  {cat}: {acc:.4f}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geolora.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", default="cv-bench",
                        choices=["cv-bench", "spatialbench"])
    parser.add_argument("--benchmark-dir", required=True,
                        help="Path to benchmark data directory")
    parser.add_argument("--depth-dir", required=True,
                        help="Path to depth maps for benchmark images")
    parser.add_argument("--method", default="geolora",
                        choices=["geolora", "static_lora", "depth_tokens",
                                 "uniform_alpha", "question_conditioned"],
                        help="Method to evaluate (default: geolora)")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.benchmark, args.benchmark_dir,
         args.depth_dir, args.method)
