"""
Evaluate GeoLoRA and baselines with depth ablation variants and question-type analysis.

Usage:
  python scripts/evaluate_geolora.py --config configs/geolora.yaml --checkpoint outputs_geolora/final
  python scripts/evaluate_geolora.py --config configs/geolora.yaml --checkpoint outputs_geolora/final \
      --method static_lora
  python scripts/evaluate_geolora.py --config configs/geolora.yaml --checkpoint outputs_geolora/final \
      --depth-mode random --seed 42
  python scripts/evaluate_geolora.py --config configs/geolora.yaml --checkpoint outputs_geolora/final \
      --depth-mode corrupted --corruption-sigma 0.3
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
import yaml
from tqdm import tqdm

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

METHODS = (
    "geolora", "static_lora", "depth_tokens", "uniform_alpha",
    "question_conditioned", "depth_gate", "depth_film",
)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Depth ablation helpers
# ---------------------------------------------------------------------------

def apply_depth_ablation(
    depth_tensor: torch.Tensor,
    mode: str,
    *,
    corruption_sigma: float = 0.2,
    gt_depth_dir: str | None = None,
    depth_stem: str | None = None,
) -> torch.Tensor:
    """
    Apply a depth ablation transformation.

    Args:
        depth_tensor: Original predicted depth, shape (1, H, W).
        mode: One of 'predicted', 'random', 'zeros', 'corrupted', 'gt'.
        corruption_sigma: Standard deviation for Gaussian noise in 'corrupted' mode.
        gt_depth_dir: Directory containing ground-truth depth .npy files.
        depth_stem: Filename stem for looking up ground-truth depth.

    Returns:
        Transformed depth tensor with the same shape.
    """
    if mode == "predicted":
        return depth_tensor

    if mode == "zeros":
        return torch.zeros_like(depth_tensor)

    if mode == "random":
        return torch.rand_like(depth_tensor)

    if mode == "corrupted":
        noise = torch.randn_like(depth_tensor) * corruption_sigma
        corrupted = (depth_tensor + noise).clamp(0.0, 1.0)
        return corrupted

    if mode == "gt":
        if gt_depth_dir is not None and depth_stem is not None:
            gt_path = os.path.join(gt_depth_dir, f"{depth_stem}.npy")
            if os.path.exists(gt_path):
                gt = np.load(gt_path).astype(np.float32)
                gt_t = torch.from_numpy(gt).unsqueeze(0)
                gt_max = gt_t.max()
                if gt_max > 0:
                    gt_t = gt_t / gt_max
                return gt_t
        # Fall back to predicted if GT not available
        return depth_tensor

    raise ValueError(f"Unknown depth mode: {mode}")


# ---------------------------------------------------------------------------
# Answer extraction & matching
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """Extract the model's answer from generated text, cleaning whitespace."""
    text = text.strip()
    # If the model wraps the answer in tags like <answer>...</answer>, extract it
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def compute_match(predicted: str, reference: str) -> dict:
    """
    Compute exact-match and keyword-match between predicted and reference answers.

    Returns:
        dict with 'exact_match' (bool) and 'keyword_match' (bool).
    """
    pred_clean = predicted.strip().lower()
    ref_clean = reference.strip().lower()

    exact = pred_clean == ref_clean

    # Keyword matching: all significant words in the reference appear in predicted
    ref_keywords = set(re.findall(r"[a-z0-9]+", ref_clean))
    pred_words = set(re.findall(r"[a-z0-9]+", pred_clean))
    # Remove stop words for a fairer keyword comparison
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "of", "in",
                  "to", "for", "on", "it", "and", "or", "that", "this", "with"}
    ref_keywords -= stop_words
    keyword = ref_keywords.issubset(pred_words) if ref_keywords else exact

    return {"exact_match": exact, "keyword_match": keyword}


# ---------------------------------------------------------------------------
# Dataset item access helpers
# ---------------------------------------------------------------------------

def get_qa_type(dataset: SpatialQADataset, idx: int) -> str:
    """Return the qa_type field from the raw dataset item, or 'unknown'."""
    raw = dataset.data[idx]
    return raw.get("qa_type", "unknown")


def get_reference_answer(dataset: SpatialQADataset, idx: int) -> str | None:
    """
    Extract the reference (expected) answer from the dataset item's messages.
    Convention: the last 'assistant' message contains the expected answer.
    """
    raw = dataset.data[idx]
    messages = raw.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def get_prompt_for_generation(dataset: SpatialQADataset, idx: int) -> str:
    """
    Build a generation prompt from the dataset item: include all messages
    except the final assistant turn, then apply a generation prompt.
    """
    raw = dataset.data[idx]
    messages = list(raw.get("messages", []))
    # Remove the final assistant message so the model generates it
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    text = dataset.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return text


def get_depth_stem(dataset: SpatialQADataset, idx: int) -> str:
    """Return the depth map filename stem for a dataset item."""
    raw = dataset.data[idx]
    from pathlib import Path
    return Path(raw["images"][0]).stem


# ---------------------------------------------------------------------------
# Model loading (multi-method)
# ---------------------------------------------------------------------------

def load_model(method: str, cfg: dict):
    """Load base model and build the appropriate model variant."""
    geolora_cfg = GeoLoRAConfig(
        **{k: tuple(v) if isinstance(v, list) else v
           for k, v in cfg["model"]["geolora"].items()}
    )
    base_name = cfg["model"]["base_model"]
    kwargs = dict(torch_dtype=torch.bfloat16)

    if method == "geolora":
        print("Loading Qwen2.5-VL + GeoLoRA...")
        model = Qwen2VLWithGeoLoRA.from_pretrained(base_name, geolora_cfg, **kwargs)

    elif method == "static_lora":
        print("Loading Qwen2.5-VL + StaticLoRA...")
        model = Qwen2VLWithStaticLoRA.from_pretrained(base_name, geolora_cfg, **kwargs)

    elif method == "uniform_alpha":
        print("Loading Qwen2.5-VL + GeoLoRA (uniform alpha)...")
        model = Qwen2VLWithGeoLoRA.from_pretrained(base_name, geolora_cfg, **kwargs)
        patch = make_uniform_alpha_geolora(geolora_cfg)
        patch(model)

    elif method == "depth_tokens":
        print("Loading Qwen2.5-VL + DepthTokens...")
        model = Qwen2VLWithDepthTokens.from_pretrained(base_name, geolora_cfg, **kwargs)

    elif method == "depth_gate":
        print("Loading Qwen2.5-VL + DepthGate (Level 1)...")
        model = Qwen2VLWithDepthGate.from_pretrained(base_name, geolora_cfg, **kwargs)

    elif method == "depth_film":
        print("Loading Qwen2.5-VL + DepthFiLM (Level 2)...")
        model = Qwen2VLWithDepthFiLM.from_pretrained(base_name, geolora_cfg, **kwargs)

    elif method == "question_conditioned":
        print("Loading Qwen2.5-VL + GeoLoRA (question-conditioned router)...")
        qc_cfg = GeoLoRAConfig(
            **{k: getattr(geolora_cfg, k) for k in geolora_cfg.__dataclass_fields__
               if k != "router_type"},
            router_type="question_conditioned",
        )
        model = Qwen2VLWithGeoLoRA.from_pretrained(base_name, qc_cfg, **kwargs)
        geolora_cfg = qc_cfg

    else:
        raise ValueError(f"Unknown method: {method}")

    return model, geolora_cfg


def load_checkpoint(model, method: str, checkpoint_path: str):
    """Load method-specific weights from a checkpoint directory."""
    if method in ("geolora", "uniform_alpha", "question_conditioned"):
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
        state = torch.load(
            os.path.join(checkpoint_path, "static_lora.pt"), map_location="cpu",
        )
        lora_sd = state["static_lora"]
        for key, val in lora_sd.items():
            # key format: layer_{li}_{pn}_{A|B}
            parts = key.split("_")
            li = int(parts[1])
            ab = parts[-1]  # "A" or "B"
            pn = "_".join(parts[2:-1])
            wrapper = model._wrapped_layers[li][pn]
            if ab == "A":
                wrapper.lora_A.data.copy_(val)
            else:
                wrapper.lora_B.data.copy_(val)

    elif method == "depth_tokens":
        state = torch.load(
            os.path.join(checkpoint_path, "depth_tokens.pt"), map_location="cpu",
        )
        model.depth_net.load_state_dict(state["depth_net"])
        model.token_proj.load_state_dict(state["token_proj"])

    elif method in ("depth_gate", "depth_film"):
        state = torch.load(
            os.path.join(checkpoint_path, f"{method}.pt"), map_location="cpu",
        )
        model.depth_net.load_state_dict(state["depth_net"])
        wrapper_sd = state["wrappers"]
        for li, lw in model._wrapped_layers.items():
            for pn, wrapper in lw.items():
                wrapper.lora_A.data.copy_(wrapper_sd[f"layer_{li}_{pn}_A"])
                wrapper.lora_B.data.copy_(wrapper_sd[f"layer_{li}_{pn}_B"])
                if method == "depth_gate":
                    wrapper.gate_proj.load_state_dict(wrapper_sd[f"layer_{li}_{pn}_gate"])
                else:
                    wrapper.film_proj.load_state_dict(wrapper_sd[f"layer_{li}_{pn}_film"])

    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Generation (multi-method)
# ---------------------------------------------------------------------------

def _prepare_generation_inputs(model, dataset, idx):
    """Build tokenized generation inputs and load the image."""
    gen_prompt = get_prompt_for_generation(dataset, idx)
    img_path = dataset.image_dir / dataset.data[idx]["images"][0]
    from PIL import Image
    image = Image.open(img_path).convert("RGB")

    gen_inputs = dataset.processor(
        text=[gen_prompt], images=[image], return_tensors="pt",
        max_length=dataset.max_length, truncation=True,
    )
    gen_inputs = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v
        for k, v in gen_inputs.items()
    }
    return gen_inputs


def _decode_generated(gen_ids, prompt_len, tokenizer):
    """Decode only the generated tokens (skip the prompt)."""
    gen_tokens = gen_ids[0, prompt_len:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return extract_answer(gen_text)


def run_generation_geolora(model, dataset, idx, depth_maps, max_new_tokens):
    """Generate with GeoLoRA / uniform_alpha / question_conditioned hook-based injection."""
    gen_inputs = _prepare_generation_inputs(model, dataset, idx)

    hooks = []
    if depth_maps is not None:
        z_geo = model.geolora.depth_net(depth_maps)

        # Question-conditioned router needs text embeddings
        if model.config.router_type == "question_conditioned":
            with torch.no_grad():
                text_embeds = model.base.model.embed_tokens(gen_inputs["input_ids"])
                q_embed = text_embeds.mean(dim=1)  # (B, d_lm)
            alphas = model.geolora.router(z_geo, q_embed=q_embed)
        else:
            alphas = model.geolora.router(z_geo)

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

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_ids = model.base.generate(
            **gen_inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )

    for hook in hooks:
        hook.remove()
    for layer_wrappers in model._wrapped_layers.values():
        for wrapper in layer_wrappers.values():
            wrapper.clear_delta()

    prompt_len = gen_inputs["input_ids"].shape[1]
    return _decode_generated(gen_ids, prompt_len, dataset.processor.tokenizer)


def run_generation_static_lora(model, dataset, idx, max_new_tokens):
    """Generate with StaticLoRA (no hooks needed, LoRA is always active)."""
    gen_inputs = _prepare_generation_inputs(model, dataset, idx)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_ids = model.base.generate(
            **gen_inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )

    prompt_len = gen_inputs["input_ids"].shape[1]
    return _decode_generated(gen_ids, prompt_len, dataset.processor.tokenizer)


def run_generation_depth_tokens(model, dataset, idx, depth_maps, max_new_tokens):
    """Generate with DepthTokens by manually prepending depth tokens to inputs_embeds."""
    gen_inputs = _prepare_generation_inputs(model, dataset, idx)

    input_ids = gen_inputs.pop("input_ids")
    attention_mask = gen_inputs.pop("attention_mask")

    # Build inputs_embeds from input_ids
    inputs_embeds = model.base.model.embed_tokens(input_ids)

    if depth_maps is not None:
        z_geo = model.depth_net(depth_maps)
        depth_tokens = model.token_proj(z_geo)  # (1, N_depth, d_lm)

        # Prepend depth tokens
        inputs_embeds = torch.cat([depth_tokens, inputs_embeds], dim=1)

        # Extend attention mask
        batch = attention_mask.size(0)
        n_depth = depth_tokens.size(1)
        depth_mask = torch.ones(
            batch, n_depth,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([depth_mask, attention_mask], dim=1)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_ids = model.base.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **{k: v for k, v in gen_inputs.items()
               if k not in ("input_ids", "attention_mask")},
        )

    prompt_len = inputs_embeds.shape[1]
    return _decode_generated(gen_ids, prompt_len, dataset.processor.tokenizer)


def run_generation_depth_conditioned_lora(model, dataset, idx, depth_maps, max_new_tokens):
    """Generate with DepthGate/DepthFiLM — set z_geo on wrappers, then generate."""
    gen_inputs = _prepare_generation_inputs(model, dataset, idx)

    # Set z_geo on all wrappers so gating/FiLM is active during generation
    if depth_maps is not None:
        z_geo = model.depth_net(depth_maps)
        for lw in model._wrapped_layers.values():
            for wrapper in lw.values():
                wrapper.set_z_geo(z_geo)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen_ids = model.base.generate(
            **gen_inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )

    # Clear z_geo
    for lw in model._wrapped_layers.values():
        for wrapper in lw.values():
            wrapper.clear_z_geo()

    prompt_len = gen_inputs["input_ids"].shape[1]
    return _decode_generated(gen_ids, prompt_len, dataset.processor.tokenizer)


def run_generation(method, model, dataset, idx, depth_maps, max_new_tokens):
    """Dispatch generation to the appropriate method-specific function."""
    if method in ("geolora", "uniform_alpha", "question_conditioned"):
        return run_generation_geolora(
            model, dataset, idx, depth_maps, max_new_tokens,
        )
    if method == "static_lora":
        return run_generation_static_lora(
            model, dataset, idx, max_new_tokens,
        )
    if method == "depth_tokens":
        return run_generation_depth_tokens(
            model, dataset, idx, depth_maps, max_new_tokens,
        )
    if method in ("depth_gate", "depth_film"):
        return run_generation_depth_conditioned_lora(
            model, dataset, idx, depth_maps, max_new_tokens,
        )
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    method: str,
    model,
    dataset: SpatialQADataset,
    cfg: dict,
    *,
    depth_mode: str = "predicted",
    corruption_sigma: float = 0.2,
    gt_depth_dir: str | None = None,
    max_new_tokens: int = 128,
) -> list[dict]:
    """
    Run full evaluation over the dataset.

    For each sample:
      1. Compute loss using teacher-forced labels.
      2. Generate an answer and compare against the reference.

    Returns a list of per-sample result dicts.
    """
    results = []

    for i in tqdm(range(len(dataset)), desc=f"Evaluating {method} (depth={depth_mode})"):
        item = dataset[i]
        qa_type = get_qa_type(dataset, i)
        reference = get_reference_answer(dataset, i)
        depth_stem = get_depth_stem(dataset, i)

        # --- Prepare inputs for loss computation ---
        inputs = {
            k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
            for k, v in item.items()
        }
        if "depth_map" in inputs:
            inputs["depth_maps"] = inputs.pop("depth_map")

        # Apply depth ablation
        if "depth_maps" in inputs:
            inputs["depth_maps"] = apply_depth_ablation(
                inputs["depth_maps"],
                depth_mode,
                corruption_sigma=corruption_sigma,
                gt_depth_dir=gt_depth_dir,
                depth_stem=depth_stem,
            ).to(inputs["depth_maps"].device)

        # --- Loss computation (teacher-forced) ---
        labels = inputs["input_ids"].clone()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs, labels=labels)
        loss_val = outputs.loss.item()

        # --- Generation-based accuracy ---
        gen_text = ""
        match_result = {"exact_match": False, "keyword_match": False}

        if reference is not None:
            gen_text = run_generation(
                method, model, dataset, i,
                depth_maps=inputs.get("depth_maps"),
                max_new_tokens=max_new_tokens,
            )
            match_result = compute_match(gen_text, reference)

        result = {
            "index": i,
            "qa_type": qa_type,
            "loss": loss_val,
            "generated": gen_text,
            "reference": reference or "",
            "exact_match": match_result["exact_match"],
            "keyword_match": match_result["keyword_match"],
        }
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------

def aggregate_results(results: list[dict]) -> dict:
    """
    Compute overall and per-question-type metrics.

    Returns a summary dict suitable for JSON serialization.
    """
    n = len(results)
    avg_loss = sum(r["loss"] for r in results) / n if n else 0.0
    exact_acc = sum(r["exact_match"] for r in results) / n if n else 0.0
    keyword_acc = sum(r["keyword_match"] for r in results) / n if n else 0.0

    overall = {
        "num_samples": n,
        "avg_loss": round(avg_loss, 6),
        "exact_match_accuracy": round(exact_acc, 4),
        "keyword_match_accuracy": round(keyword_acc, 4),
    }

    # Per qa_type breakdown
    by_type = defaultdict(list)
    for r in results:
        by_type[r["qa_type"]].append(r)

    per_type = {}
    for qa_type, group in sorted(by_type.items()):
        k = len(group)
        t_loss = sum(r["loss"] for r in group) / k
        t_exact = sum(r["exact_match"] for r in group) / k
        t_keyword = sum(r["keyword_match"] for r in group) / k
        per_type[qa_type] = {
            "num_samples": k,
            "avg_loss": round(t_loss, 6),
            "exact_match_accuracy": round(t_exact, 4),
            "keyword_match_accuracy": round(t_keyword, 4),
        }

    return {
        "overall": overall,
        "per_type": per_type,
    }


def print_summary(summary: dict, method: str, depth_mode: str):
    """Pretty-print the summary table to stdout."""
    print("\n" + "=" * 72)
    print(f"  {method} Results  (depth_mode={depth_mode})")
    print("=" * 72)
    ov = summary["overall"]
    print(f"  Samples:   {ov['num_samples']}")
    print(f"  Avg Loss:  {ov['avg_loss']:.6f}")
    print(f"  Exact Acc: {ov['exact_match_accuracy']:.4f}")
    print(f"  KW Acc:    {ov['keyword_match_accuracy']:.4f}")
    print("-" * 72)
    print(f"  {'qa_type':<30s} {'N':>5s} {'Loss':>10s} {'Exact':>8s} {'KW':>8s}")
    print("-" * 72)
    for qt, m in summary["per_type"].items():
        print(f"  {qt:<30s} {m['num_samples']:>5d} "
              f"{m['avg_loss']:>10.6f} "
              f"{m['exact_match_accuracy']:>8.4f} "
              f"{m['keyword_match_accuracy']:>8.4f}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GeoLoRA and baselines with depth ablation and per-type analysis",
    )
    parser.add_argument("--config", default="configs/geolora.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--method", default="geolora", choices=METHODS,
                        help="Model variant to evaluate (default: geolora)")
    parser.add_argument("--depth-mode", default="predicted",
                        choices=["predicted", "random", "zeros", "corrupted", "gt"],
                        help="Depth ablation mode (default: predicted)")
    parser.add_argument("--corruption-sigma", type=float, default=0.2,
                        help="Gaussian noise sigma for 'corrupted' mode (default: 0.2)")
    parser.add_argument("--gt-depth-dir", default=None,
                        help="Directory for ground-truth depth maps (for 'gt' mode)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens to generate per sample (default: 128)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <checkpoint>/results_<method>_<depth-mode>.json)")
    args = parser.parse_args()

    # --- Seed ---
    set_seed(args.seed)

    # --- Config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- Model ---
    print(f"Loading model from {cfg['model']['base_model']}...")
    model, geolora_cfg = load_model(args.method, cfg)

    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(model, args.method, args.checkpoint)

    model.to("cuda")
    model.eval()

    # --- Dataset ---
    test_ds = SpatialQADataset(
        cfg["data"]["val_file"],
        cfg["data"]["image_dir"],
        cfg["data"]["depth_dir"],
        model.processor,
    )
    print(f"Test set: {len(test_ds)} samples")

    # Resolve GT depth dir for 'gt' mode
    gt_depth_dir = args.gt_depth_dir
    if args.depth_mode == "gt" and gt_depth_dir is None:
        gt_depth_dir = cfg["data"].get("gt_depth_dir", cfg["data"]["depth_dir"])
        print(f"No --gt-depth-dir specified; using {gt_depth_dir}")

    # --- Run ---
    print(f"Method: {args.method} | Depth mode: {args.depth_mode} | Seed: {args.seed}")
    results = run_evaluation(
        args.method,
        model,
        test_ds,
        cfg,
        depth_mode=args.depth_mode,
        corruption_sigma=args.corruption_sigma,
        gt_depth_dir=gt_depth_dir,
        max_new_tokens=args.max_new_tokens,
    )

    # --- Aggregate & save ---
    summary = aggregate_results(results)
    print_summary(summary, args.method, args.depth_mode)

    output_data = {
        "config": {
            "method": args.method,
            "depth_mode": args.depth_mode,
            "corruption_sigma": args.corruption_sigma if args.depth_mode == "corrupted" else None,
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": summary,
        "per_sample": results,
    }

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(
            args.checkpoint, f"results_{args.method}_{args.depth_mode}.json",
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
