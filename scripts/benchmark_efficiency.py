"""
Usage: python scripts/benchmark_efficiency.py --config configs/geolora.yaml
       python scripts/benchmark_efficiency.py --config configs/geolora.yaml --methods geolora static_lora
       python scripts/benchmark_efficiency.py --config configs/geolora.yaml --output results/benchmark.json
"""
import argparse
import json
import os
import time
from dataclasses import asdict

import torch
import torch.nn as nn
import yaml

from geolora.config import GeoLoRAConfig
from geolora.depth_geometry import DepthGeometryNet
from geolora.lora_bank import LoRABasisBank
from geolora.router import GeometryRouter
from geolora.injection import DynamicLoRALinear
from geolora.baselines import _compute_static_lora_rank


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_params(module: nn.Module, only_trainable: bool = True) -> int:
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def fmt_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def fmt_mem(bytes_val: float) -> str:
    return f"{bytes_val / (1024 ** 2):.1f} MB"


def fmt_flops(flops: float) -> str:
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops:.0f} FLOPs"


# ---------------------------------------------------------------------------
# FLOPs estimation helpers (simple analytical, no profiler needed)
# ---------------------------------------------------------------------------

def linear_flops(in_feat: int, out_feat: int, seq_len: int, bias: bool = False) -> int:
    """FLOPs for a linear layer: 2 * in * out * seq (multiply-add)."""
    flops = 2 * in_feat * out_feat * seq_len
    if bias:
        flops += out_feat * seq_len
    return flops


def conv2d_flops(in_c: int, out_c: int, k: int, h_out: int, w_out: int) -> int:
    """FLOPs for a Conv2d: 2 * in_c * k * k * out_c * h_out * w_out."""
    return 2 * in_c * k * k * out_c * h_out * w_out


def estimate_geolora_flops(config: GeoLoRAConfig, seq_len: int, depth_h: int = 224, depth_w: int = 224) -> dict:
    """Estimate FLOPs for the GeoLoRA adapter path only (not the base LLM)."""
    num_layers = len(config.target_layers)
    K = config.num_bases
    r = config.lora_rank

    # --- DepthGeometryNet ---
    # Sobel: 2 x (1*1*3*3) convolutions over H*W -> negligible
    sobel_flops = 2 * (2 * 1 * 1 * 9 * depth_h * depth_w)

    # conv1: 3->64, k=7, stride=4
    h1, w1 = depth_h // 4, depth_w // 4
    f_conv1 = conv2d_flops(3, 64, 7, h1, w1)
    # conv2: 64->128, k=3, stride=2
    h2, w2 = h1 // 2, w1 // 2
    f_conv2 = conv2d_flops(64, 128, 3, h2, w2)
    # conv3: 128->256, k=3, stride=2
    h3, w3 = h2 // 2, w2 // 2
    f_conv3 = conv2d_flops(128, 256, 3, h3, w3)
    # LayerNorm(256) + Linear(256, d_geo)
    f_proj = 2 * 256 * config.d_geo

    depth_net_flops = sobel_flops + f_conv1 + f_conv2 + f_conv3 + f_proj

    # --- GeometryRouter ---
    # Per layer: Linear(d_geo, d_geo) + Linear(d_geo, K) = 2*d_geo*d_geo + 2*d_geo*K
    router_per_layer = 2 * config.d_geo * config.d_geo + 2 * config.d_geo * K
    router_flops = num_layers * router_per_layer

    # --- LoRA basis mixing + delta computation (per layer, per projection) ---
    bank_flops = 0
    for proj in config.target_projections:
        d_in, d_out = config.proj_dims(proj)
        # A_mixed = einsum("bk, kri -> bri"): K * r * d_in multiplies
        mix_a = K * r * d_in
        # B_mixed = einsum("bk, kor -> bor"): K * d_out * r multiplies
        mix_b = K * d_out * r
        # z = einsum("bri, bsi -> bsr"): r * d_in * seq_len
        proj_a = r * d_in * seq_len
        # delta = einsum("bor, bsr -> bso"): d_out * r * seq_len
        proj_b = d_out * r * seq_len
        bank_flops += 2 * (mix_a + mix_b + proj_a + proj_b)  # x2 for multiply-add
    bank_flops *= num_layers

    # --- Gate scaling (per layer, per projection, per token) ---
    gate_flops = 0
    for proj in config.target_projections:
        _, d_out = config.proj_dims(proj)
        gate_flops += seq_len * d_out  # element-wise multiply
    gate_flops *= num_layers

    total = depth_net_flops + router_flops + bank_flops + gate_flops
    return {
        "depth_net": depth_net_flops,
        "router": router_flops,
        "lora_banks": bank_flops,
        "gates": gate_flops,
        "total": total,
    }


def estimate_static_lora_flops(config: GeoLoRAConfig, seq_len: int, matched_rank: int = None) -> dict:
    """Estimate FLOPs for static LoRA at the matched-budget rank."""
    num_layers = len(config.target_layers)
    r = matched_rank if matched_rank is not None else _compute_static_lora_rank(config)
    total = 0
    for proj in config.target_projections:
        d_in, d_out = config.proj_dims(proj)
        total += 2 * seq_len * d_in * r + 2 * seq_len * r * d_out
    total *= num_layers
    return {"lora_forward": total, "total": total, "rank_used": r}


# ---------------------------------------------------------------------------
# Standalone module builders (no base model needed)
# ---------------------------------------------------------------------------

def build_geolora_modules(config: GeoLoRAConfig, device: torch.device, dtype: torch.dtype):
    """Build GeoLoRA adapter modules (no base model weights)."""
    depth_net = DepthGeometryNet(d_geo=config.d_geo).to(device, dtype=dtype)
    router = GeometryRouter(
        d_geo=config.d_geo,
        num_bases=config.num_bases,
        num_layers=len(config.target_layers),
    ).to(device, dtype=dtype)

    banks = nn.ModuleDict()
    for i in range(len(config.target_layers)):
        layer_banks = nn.ModuleDict()
        for proj in config.target_projections:
            d_in, d_out = config.proj_dims(proj)
            layer_banks[proj] = LoRABasisBank(
                num_bases=config.num_bases,
                lora_rank=config.lora_rank,
                d_in=d_in,
                d_out=d_out,
            )
        banks[str(i)] = layer_banks
    banks = banks.to(device, dtype=dtype)

    # Gate parameters (one per layer per projection)
    gates = []
    for _ in range(len(config.target_layers)):
        for _ in config.target_projections:
            gates.append(nn.Parameter(torch.zeros(1, device=device, dtype=dtype)))

    return depth_net, router, banks, gates


class StaticLoRALinear(nn.Module):
    """Minimal static LoRA for benchmarking: delta = B @ A @ h."""

    def __init__(self, d_in: int, d_out: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.xavier_uniform_(self.A)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (h @ self.A.T @ self.B.T)


def build_static_lora_modules(config: GeoLoRAConfig, device: torch.device, dtype: torch.dtype):
    """Build static LoRA modules at matched-budget rank for fair comparison."""
    matched_rank = _compute_static_lora_rank(config)
    modules = nn.ModuleDict()
    for i in range(len(config.target_layers)):
        layer_loras = nn.ModuleDict()
        for proj in config.target_projections:
            d_in, d_out = config.proj_dims(proj)
            layer_loras[proj] = StaticLoRALinear(d_in, d_out, matched_rank)
        modules[str(i)] = layer_loras
    modules = modules.to(device, dtype=dtype)
    return modules, matched_rank


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def benchmark_geolora(config: GeoLoRAConfig, device: torch.device, dtype: torch.dtype,
                      seq_len: int, num_warmup: int, num_iters: int):
    """Benchmark the GeoLoRA adapter path."""
    depth_net, router, banks, gates = build_geolora_modules(config, device, dtype)
    num_layers = len(config.target_layers)

    depth_map = torch.randn(1, 1, 224, 224, device=device, dtype=dtype)
    hidden = torch.randn(1, seq_len, config.d_lm, device=device, dtype=dtype)

    # Trainable param count
    trainable = (count_params(depth_net) + count_params(router)
                 + count_params(banks) + len(gates))

    # FLOPs
    flops_info = estimate_geolora_flops(config, seq_len)

    # Warmup
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(num_warmup):
        z_geo = depth_net(depth_map)
        alphas = router(z_geo)
        for i in range(num_layers):
            for j, proj in enumerate(config.target_projections):
                delta = banks[str(i)][proj](hidden, alphas[i])
                _ = gates[i * len(config.target_projections) + j] * delta
    torch.cuda.synchronize()

    # Timed iterations
    start_mem = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(num_iters):
        z_geo = depth_net(depth_map)
        alphas = router(z_geo)
        for i in range(num_layers):
            for j, proj in enumerate(config.target_projections):
                delta = banks[str(i)][proj](hidden, alphas[i])
                _ = gates[i * len(config.target_projections) + j] * delta
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated(device) - start_mem

    latency_ms = (elapsed / num_iters) * 1000
    throughput = num_iters / elapsed

    # Cleanup
    del depth_net, router, banks, gates, depth_map, hidden
    torch.cuda.empty_cache()

    return {
        "method": "GeoLoRA",
        "trainable_params": trainable,
        "trainable_params_str": fmt_params(trainable),
        "adapter_flops": flops_info["total"],
        "adapter_flops_str": fmt_flops(flops_info["total"]),
        "flops_breakdown": {k: fmt_flops(v) for k, v in flops_info.items()},
        "latency_ms": round(latency_ms, 2),
        "peak_vram": peak_mem,
        "peak_vram_str": fmt_mem(peak_mem),
        "throughput_sps": round(throughput, 1),
    }


def benchmark_static_lora(config: GeoLoRAConfig, device: torch.device, dtype: torch.dtype,
                           seq_len: int, num_warmup: int, num_iters: int):
    """Benchmark static LoRA at matched-budget rank for fair comparison."""
    modules, matched_rank = build_static_lora_modules(config, device, dtype)
    num_layers = len(config.target_layers)
    hidden = torch.randn(1, seq_len, config.d_lm, device=device, dtype=dtype)

    trainable = count_params(modules)
    flops_info = estimate_static_lora_flops(config, seq_len, matched_rank=matched_rank)
    print(f"  Static LoRA matched-budget rank: {matched_rank} (GeoLoRA rank: {config.lora_rank})")

    # Warmup
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(num_warmup):
        for i in range(num_layers):
            for proj in config.target_projections:
                _ = modules[str(i)][proj](hidden)
    torch.cuda.synchronize()

    # Timed iterations
    start_mem = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(num_iters):
        for i in range(num_layers):
            for proj in config.target_projections:
                _ = modules[str(i)][proj](hidden)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated(device) - start_mem

    latency_ms = (elapsed / num_iters) * 1000
    throughput = num_iters / elapsed

    del modules, hidden
    torch.cuda.empty_cache()

    return {
        "method": f"Static LoRA (rank={matched_rank})",
        "trainable_params": trainable,
        "trainable_params_str": fmt_params(trainable),
        "adapter_flops": flops_info["total"],
        "adapter_flops_str": fmt_flops(flops_info["total"]),
        "flops_breakdown": {k: fmt_flops(v) for k, v in flops_info.items() if isinstance(v, (int, float))},
        "latency_ms": round(latency_ms, 2),
        "peak_vram": peak_mem,
        "peak_vram_str": fmt_mem(peak_mem),
        "throughput_sps": round(throughput, 1),
    }


def benchmark_base_model(config: GeoLoRAConfig, device: torch.device, dtype: torch.dtype,
                          seq_len: int, num_warmup: int, num_iters: int):
    """
    Benchmark the base model path: no adapter, just frozen q_proj/v_proj linears.
    Uses standalone Linear layers to simulate the frozen projections only.
    """
    num_layers = len(config.target_layers)

    # Build frozen linears matching target projections
    linears = nn.ModuleDict()
    for i in range(num_layers):
        layer_linears = nn.ModuleDict()
        for proj in config.target_projections:
            d_in, d_out = config.proj_dims(proj)
            lin = nn.Linear(d_in, d_out, bias=True)
            lin.requires_grad_(False)
            layer_linears[proj] = lin
        linears[str(i)] = layer_linears
    linears = linears.to(device, dtype=dtype)

    hidden = torch.randn(1, seq_len, config.d_lm, device=device, dtype=dtype)

    # FLOPs for frozen linears only
    total_flops = 0
    for proj in config.target_projections:
        d_in, d_out = config.proj_dims(proj)
        total_flops += linear_flops(d_in, d_out, seq_len, bias=True)
    total_flops *= num_layers

    # Warmup
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(num_warmup):
            for i in range(num_layers):
                for proj in config.target_projections:
                    _ = linears[str(i)][proj](hidden)
    torch.cuda.synchronize()

    # Timed iterations
    start_mem = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            for i in range(num_layers):
                for proj in config.target_projections:
                    _ = linears[str(i)][proj](hidden)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated(device) - start_mem

    latency_ms = (elapsed / num_iters) * 1000
    throughput = num_iters / elapsed

    del linears, hidden
    torch.cuda.empty_cache()

    return {
        "method": "Base (no adapter)",
        "trainable_params": 0,
        "trainable_params_str": "0",
        "adapter_flops": 0,
        "adapter_flops_str": "0 FLOPs",
        "flops_breakdown": {},
        "latency_ms": round(latency_ms, 2),
        "peak_vram": peak_mem,
        "peak_vram_str": fmt_mem(peak_mem),
        "throughput_sps": round(throughput, 1),
    }


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_table(results: list):
    """Print a formatted comparison table."""
    header = ["Method", "Trainable Params", "Adapter FLOPs", "Latency (ms)", "Peak VRAM", "Throughput (s/s)"]
    rows = []
    for r in results:
        rows.append([
            r["method"],
            r["trainable_params_str"],
            r["adapter_flops_str"],
            f"{r['latency_ms']:.2f}",
            r["peak_vram_str"],
            f"{r['throughput_sps']:.1f}",
        ])

    col_widths = [max(len(header[i]), *(len(row[i]) for row in rows)) + 2 for i in range(len(header))]

    def fmt_row(cols):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cols, col_widths)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    print("\n" + sep)
    print(fmt_row(header))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)

    # Print relative overhead vs base
    if len(results) >= 2:
        base_latency = results[-1]["latency_ms"]  # base model is last
        print("\nRelative adapter overhead vs base (no adapter):")
        for r in results:
            if r["method"] == "Base (no adapter)":
                continue
            if base_latency > 0:
                overhead = ((r["latency_ms"] - base_latency) / base_latency) * 100
                print(f"  {r['method']}: +{overhead:.1f}% latency")
            else:
                print(f"  {r['method']}: base latency too small to compare")

    # Print GeoLoRA FLOPs breakdown
    for r in results:
        if r["method"] == "GeoLoRA" and r.get("flops_breakdown"):
            print(f"\nGeoLoRA FLOPs breakdown:")
            for component, val in r["flops_breakdown"].items():
                if component != "total":
                    print(f"  {component}: {val}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

AVAILABLE_METHODS = {
    "geolora": benchmark_geolora,
    "static_lora": benchmark_static_lora,
    "base": benchmark_base_model,
}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GeoLoRA adapter efficiency vs baselines."
    )
    parser.add_argument(
        "--config", default="configs/geolora.yaml",
        help="Path to GeoLoRA config YAML.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["geolora", "static_lora", "base"],
        choices=list(AVAILABLE_METHODS.keys()),
        help="Methods to benchmark (default: all).",
    )
    parser.add_argument(
        "--output", default="results/benchmark_efficiency.json",
        help="Output JSON path.",
    )
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for dummy inputs.")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--num-iters", type=int, default=100, help="Timed iterations.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        print("Please run on a machine with a CUDA-capable GPU.")
        return

    device = torch.device("cuda")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    geolora_cfg = GeoLoRAConfig(
        **{k: tuple(v) if isinstance(v, list) else v
           for k, v in cfg["model"]["geolora"].items()}
    )

    print(f"Benchmark config:")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Warmup iters:    {args.num_warmup}")
    print(f"  Timed iters:     {args.num_iters}")
    print(f"  Dtype:           {args.dtype}")
    print(f"  Device:          {torch.cuda.get_device_name(device)}")
    print(f"  Methods:         {', '.join(args.methods)}")
    print(f"  LoRA rank:       {geolora_cfg.lora_rank}")
    print(f"  Num bases (K):   {geolora_cfg.num_bases}")
    print(f"  Target layers:   {geolora_cfg.target_layers}")
    print()

    # Run benchmarks (base last so overhead can be computed)
    ordered_methods = [m for m in args.methods if m != "base"] + (["base"] if "base" in args.methods else [])
    results = []
    for method_name in ordered_methods:
        print(f"Benchmarking {method_name}...")
        bench_fn = AVAILABLE_METHODS[method_name]
        result = bench_fn(
            geolora_cfg, device, dtype,
            args.seq_len, args.num_warmup, args.num_iters,
        )
        results.append(result)
        print(f"  Done: {result['latency_ms']:.2f} ms, {result['peak_vram_str']} VRAM")

    # Print table
    print_table(results)

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        "config": {
            "seq_len": args.seq_len,
            "num_warmup": args.num_warmup,
            "num_iters": args.num_iters,
            "dtype": args.dtype,
            "device": torch.cuda.get_device_name(device),
            "geolora": {k: v for k, v in asdict(geolora_cfg).items()
                        if k in ("d_geo", "num_bases", "lora_rank", "target_layers", "target_projections")},
        },
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
