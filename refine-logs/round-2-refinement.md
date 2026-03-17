# Round 2 Refinement

## Problem Anchor
[Verbatim — same as round 0]

## Anchor Check
- Original bottleneck: VLM processes all scenes identically; depth unused for adaptation
- Revised method still addresses it: YES — depth conditions LoRA adaptation at all three levels
- Drift: NONE

## Simplicity Check
- Dominant contribution: Depth-conditioned PEFT — a systematic study, not a single complex mechanism
- Components: Kept tight (DepthGeometryNet shared, three LoRA conditioning levels)
- Reviewer accepted framing; asked for precision, not restructuring
- Smallest adequate route: DepthGate may be the final answer (cleanest); higher levels justify only if empirically needed

## Changes Made

### 1. Added exact equations with target modules, ranks, dimensions
- Reviewer said: Missing insertion details, gate granularity, matched budgets
- Action: Full mathematical specification for all three levels
- Impact: Implementable from the proposal alone

### 2. Added category-stratified analysis design
- Reviewer said: "Degree of conditioning depends on complexity" not yet provable
- Action: Pre-define depth-sensitive vs non-depth-sensitive QA type split
- Impact: Directly supports Claim 3

### 3. Softened novelty claim
- Reviewer said: "No prior work" claim is brittle
- Action: Changed to "to our knowledge" + emphasis on study design
- Impact: Less attackable positioning

## Revised Proposal

# DepthPEFT: Depth-Conditioned Parameter-Efficient Fine-Tuning for VLM Spatial Reasoning

## Problem Anchor
- Bottom-line problem: Qwen2.5-VL-7B lacks spatial reasoning in driving scenes. Depth maps from P1 exist but no method uses them to condition the VLM's adaptation mechanism.
- Must-solve bottleneck: Depth-aware method must clearly outperform zero-shot and standard LoRA.
- Constraints: 1× RTX 5880 48GB, Qwen2.5-VL-7B, SpatialQA (132K, splits done)
- Success condition: Beats standard LoRA, depth matters causally, simple enough to explain

## Technical Gap

Existing depth-VLM methods treat depth as passive input features. To our knowledge, no prior work conditions the parameter-efficient fine-tuning mechanism itself on scene depth geometry for spatial reasoning tasks.

## Method Thesis

Depth-conditioned PEFT is the right abstraction for incorporating scene geometry into VLM spatial reasoning. Simple conditioning may already suffice; richer conditioning (basis mixing) is justified only if empirically necessary.

## Contribution

- **Dominant**: Systematic study of depth-conditioned PEFT for VLM spatial reasoning
- **Supporting**: Evidence on HOW MUCH conditioning is needed via progressive framework
- **Paper logic**: If DepthGate wins → "simple conditioning suffices"; if DepthBasis wins → "scene-specific strategies matter"; both publishable

## Method Specification

### Notation

- Base VLM: Qwen2.5-VL-7B-Instruct
- LLM hidden dim: d = 3584
- GQA: 28 Q-heads × 128 dim, 4 KV-heads × 128 dim
- q_proj: R^{d → d} (3584 → 3584)
- v_proj: R^{d → d_kv} (3584 → 512)
- Target layers: L = {20, 21, ..., 27} (upper 8 of 28)
- Target projections: {q_proj, v_proj}
- LoRA rank: r = 16 for all methods
- Depth map: D ∈ R^{1×H×W} from P1

### Shared: DepthGeometryNet θ_depth (~445K params)

```
z_geo = f_depth(D) ∈ R^{d_geo=256}

f_depth(D):
  G = [D; Sobel_x(D); Sobel_y(D)]                    ∈ R^{3×H×W}
  h1 = GELU(Conv2d(3→64, k=7, s=4, p=3)(G))          ∈ R^{64×H/4×W/4}
  h2 = GELU(Conv2d(64→128, k=3, s=2, p=1)(h1))       ∈ R^{128×H/8×W/8}
  h3 = GELU(Conv2d(128→256, k=3, s=2, p=1)(h2))      ∈ R^{256×H/16×W/16}
  h4 = AvgPool(h3) → flatten                          ∈ R^{256}
  z_geo = GELU(Linear(LayerNorm(h4)))                  ∈ R^{256}
```

### Baseline: Standard LoRA (no depth) — ~9.0M trainable

For each layer l ∈ L, projection p ∈ {q, v}:

```
A_l^p ∈ R^{r × d_in},  B_l^p ∈ R^{d_out × r}      (Xavier/zeros init)
Δ_l^p = B_l^p · A_l^p · h                            (standard LoRA delta)
output = W_frozen^p · h + (1/r) · Δ_l^p
```

Params: |L| × |P| × (r × d_in + d_out × r) = 8 × 2 × (16×3584 + d_out×16)
= 8 × (16×3584 + 3584×16 + 16×3584 + 512×16) = 8 × (57344 + 57344 + 57344 + 8192) = 8 × 180,224 ≈ **1.44M per layer × 8 = ... let me compute correctly:**

Actually: per layer for q_proj: r×d_in + d_out×r = 16×3584 + 3584×16 = 114,688
per layer for v_proj: 16×3584 + 512×16 = 57,344 + 8,192 = 65,536
per layer total: 180,224
8 layers: **1,441,792 ≈ 1.44M**

To match GeoLoRA's ~9.17M budget, increase rank: r_matched ≈ 102 → **~9.19M**

### Level 1: DepthGate — Scalar depth gating (~9.5M trainable)

Standard LoRA (r=102, matched budget ~9.19M) + DepthGeometryNet (445K) + gating:

For each layer l, projection p:
```
g_l^p = σ(w_l^p · z_geo + b_l^p) ∈ R^1    (scalar gate, w ∈ R^{d_geo}, b ∈ R^1)
output = W_frozen^p · h + g_l^p · (1/r) · B_l^p · A_l^p · h
```

Gate params: 8 layers × 2 projections × (256 + 1) = 4,112
Total: 9,190K + 445K + 4K ≈ **9.64M**

**Gate granularity**: Scalar per (layer, projection) — the minimal signal. z_geo determines whether each layer's LoRA is activated or suppressed based on scene geometry.

### Level 2: DepthFiLM — Per-dimension scale+shift (~9.7M trainable)

Standard LoRA (r=102, ~9.19M) + DepthGeometryNet (445K) + FiLM:

For each layer l, projection p with d_out ∈ {3584, 512}:
```
[γ_l^p, β_l^p] = Linear(z_geo, 2 × d_out)   γ,β ∈ R^{d_out}
Δ_l^p = (1/r) · B_l^p · A_l^p · h            ∈ R^{B × seq × d_out}
output = W_frozen^p · h + γ_l^p ⊙ Δ_l^p + β_l^p
```

FiLM params: 8 × (256 × 2×3584 + 2×3584 + 256 × 2×512 + 2×512)
= 8 × (1,835,008 + 7,168 + 262,144 + 1,024) ≈ 8 × 2.1M ≈ **16.8M** ← TOO MANY

**CORRECTION**: FiLM with full d_out dimensions is too expensive. Use rank-space FiLM instead:

```
[γ_l^p, β_l^p] = Linear(z_geo, 2 × r)       γ,β ∈ R^r
z_l^p = A_l^p · h                             ∈ R^{B × seq × r}
z_conditioned = γ_l^p ⊙ z_l^p + β_l^p        (FiLM in rank space)
Δ_l^p = B_l^p · z_conditioned                 ∈ R^{B × seq × d_out}
output = W_frozen^p · h + Δ_l^p
```

FiLM params: 8 × 2 × (256 × 32 + 32) = 8 × 2 × 8,224 ≈ **132K**
Total: 9,190K + 445K + 132K ≈ **9.77M** ✓

**Why rank-space**: FiLM in the r=16 rank space (32 dims for γ+β) is far more parameter-efficient than in full d_out space, and modulates the latent representation that LoRA learns.

### Level 3: DepthBasis — Basis mixing (~9.17M trainable)

No standard LoRA. K=6 LoRA basis pairs per (layer, projection), mixed by depth:

```
α_l = softmax(MLP(z_geo + LayerEmbed(l)))     ∈ R^K    (K=6)
A_mixed = Σ_k α_l^k · A_l^{p,k}              ∈ R^{r × d_in}
B_mixed = Σ_k α_l^k · B_l^{p,k}              ∈ R^{d_out × r}
Δ_l^p = B_mixed · A_mixed · h
output = W_frozen^p · h + gate_l^p · Δ_l^p    (zero-init gate)
```

r = 16 (smaller rank, but K=6 bases = effective expressivity)
Basis params: 8 × 2 × 6 × (16×3584 + 3584×16) + 8 × 2 × 6 × (16×3584 + 512×16)
= 8 × 6 × (114,688 + 65,536) = 8 × 6 × 180,224 ≈ **8.65M**
Router: ~70K, DepthGeometryNet: 445K, Gates: <1K
Total: **~9.17M** ✓

### Parameter Budget Summary

| Method | LoRA params | Depth params | Conditioning params | Total |
|--------|------------|-------------|-------------------|-------|
| Standard LoRA (r=102) | 9.19M | — | — | 9.19M |
| DepthGate (r=102) | 9.19M | 445K | 4K (gates) | 9.64M |
| DepthFiLM (r=102) | 9.19M | 445K | 132K (rank FiLM) | 9.77M |
| DepthBasis (r=16, K=6) | 8.65M | 445K | 70K (router) | 9.17M |

All within ~9-10M total trainable parameters. DepthGate/FiLM have slightly MORE total params than DepthBasis — the comparison is fair, and any DepthBasis advantage cannot be explained by parameter count.

### Insertion Point

Target: q_proj and v_proj of self_attn in Qwen2.5-VL LLM layers 20-27.

Levels 1-2: Standard PEFT wrapping — replace each target Linear with a LoRA-augmented version. Depth conditioning applied to the LoRA output.

Level 3: Replace each target Linear with DynamicLoRALinear (hook-based injection, existing implementation).

All: ViT, Merger, LLM base weights, and layers 0-19 are frozen.

### z_geo Is Intentionally Global

z_geo is a single global scene geometry descriptor per image. This is an intentional design choice:
- Spatial reasoning questions about a scene are conditioned on the SCENE'S geometry, not individual object locations
- Object-level geometry is already in the visual tokens via the ViT
- A global descriptor is sufficient for scene-level conditioning (e.g., "this is a complex intersection" vs "this is an open highway")

We do NOT claim object-level geometry reasoning from z_geo.

## Validation Plan

### Analysis Design: Depth-Sensitive Category Split

SpatialQA has 24 question types. Pre-define two splits:

**Depth-sensitive types** (where geometric structure directly impacts the answer):
- distance estimation, relative depth, height estimation, occlusion reasoning, size estimation at depth

**Non-depth-sensitive types** (where geometry is less relevant):
- color identification, object counting, direction naming, lane identification

Hypothesis: Depth-conditioned PEFT helps more on depth-sensitive types. If Level 3 (basis mixing) is needed anywhere, it's on depth-sensitive types.

### Experiment Table

| Experiment | Method | Purpose |
|-----------|--------|---------|
| E1 | Zero-shot Qwen2.5-VL-7B | Floor baseline |
| E2 | Standard LoRA (r=102, matched) | Non-depth PEFT baseline |
| E3 | DepthGate (r=102 + gates) | Simplest depth conditioning |
| E4 | DepthFiLM (r=102 + rank FiLM) | Medium depth conditioning |
| E5 | DepthBasis (K=6, r=16) | Richest depth conditioning |
| E6 | DepthGate (shuffled depth) | Causality: is real depth needed? |
| E7 | DepthGate (zero depth) | Causality: is any depth needed? |

**Metrics**: Overall accuracy, per-type accuracy, depth-sensitive vs non-depth-sensitive split

### Expected Outcomes (Pre-Committed Logic)

| Outcome | Paper Narrative |
|---------|----------------|
| E3 > E2 clearly, E5 ≈ E3 | "Simple depth gating suffices. Geometric conditioning of PEFT is the key insight, not complex routing." |
| E5 > E3 clearly | "Scene-specific adaptation strategies are needed. Richer conditioning captures geometric nuances that simple gating misses." |
| E3 ≈ E2 | "Depth conditioning of PEFT does not help. Gap exists but this mechanism doesn't close it." (Negative result — still thesis-publishable) |
| E5 > E3 only on depth-sensitive types | "Richer conditioning is needed specifically for depth-dependent reasoning. A selective depth PEFT strategy is optimal." |

## Compute

| Run | GPU Hours |
|-----|-----------|
| E1 (zero-shot eval) | ~2h |
| E2 (Standard LoRA train+eval) | ~8h |
| E3 (DepthGate train+eval) | ~9h |
| E4 (DepthFiLM train+eval) | ~9h |
| E5 (DepthBasis train+eval) | ~10h |
| E6-E7 (ablation evals) | ~2h |
| **Total** | **~40h** |

Timeline: ~1 week on RTX 5880
