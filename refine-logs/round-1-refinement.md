# Round 1 Refinement

## Problem Anchor
[Verbatim from round 0]

- **Bottom-line problem**: Qwen2.5-VL-7B lacks spatial reasoning capability in autonomous driving scenes. Depth maps from P1 (PGA-Depth) exist as a rich geometric signal, but current methods either ignore depth or integrate it as passive features without changing how the VLM processes information.
- **Must-solve bottleneck**: Need a method that uses depth maps to MEANINGFULLY improve VLM spatial reasoning — must clearly outperform both zero-shot and standard LoRA fine-tuning.
- **Non-goals**: New depth estimator, new dataset, multi-model generalization, real-time deployment
- **Constraints**: 1× RTX 5880 48GB, Qwen2.5-VL-7B, NuScenes + SpatialQA (132K, already split)
- **Success condition**: Depth-aware method beats standard LoRA, depth quality matters, simple enough to explain

## Anchor Check

- **Original bottleneck**: VLM processes all scenes identically regardless of geometry; depth is available but unused
- **Does the revised method still address it?** YES — depth conditioning on the adapter is the core mechanism
- **Reviewer suggestions rejected as drift**: NONE — all feedback is well-anchored. The reviewer correctly identified that the problem is "depth-conditioned PEFT" not "dynamic basis mixing."

## Simplicity Check

- **Dominant contribution after revision**: Depth-conditioned PEFT for VLM spatial reasoning — the first systematic study of how depth geometry should condition LoRA adaptation for spatial understanding
- **Components removed or merged**: The "three options menu" is eliminated. Instead, we present a unified framework with progressive conditioning depth.
- **Reviewer suggestions rejected as unnecessary complexity**: NONE — reviewer recommended simplification, which we adopt
- **Why the mechanism is the smallest adequate route**: We start from the simplest depth conditioning (layerwise gating) and only justify escalation to basis mixing if empirically required

## Changes Made

### 1. Reframed contribution from "GeoLoRA" to "Depth-Conditioned PEFT Framework"
- Reviewer said: Contribution is unclear — two claims fighting each other
- Action: Reframe as a progressive study. Core contribution = depth-conditioned adaptation. GeoLoRA = one instantiation.
- Reasoning: This is a stronger paper — the contribution is the IDEA (depth should condition adaptation), not one specific mechanism
- Impact: Cleaner story, broader applicability, less fragile to GeoLoRA underperforming

### 2. Added IA3/FiLM-style depth conditioning as serious baseline
- Reviewer said: Missing the key control — simpler depth-conditioned PEFT
- Action: Add Depth-FiLM as the simplest depth-conditioned method (scale+shift from depth on layer norms)
- Reasoning: This directly separates "depth conditioning helps" from "basis mixing is needed"
- Impact: Much stronger ablation structure

### 3. Eliminated method menu, committed to single formulation
- Reviewer said: Proposal is a menu of three methods, not one committed design
- Action: Present a single framework with three conditioning levels (gating → FiLM → basis mixing)
- Reasoning: Progressive complexity justification, not three separate proposals
- Impact: Cleaner paper structure

### 4. Narrowed gap statement
- Reviewer said: "All prior methods keep attention fixed" is overbroad
- Action: Narrowed to: "Existing depth-VLM methods add depth as features but don't condition the adaptation mechanism itself"
- Reasoning: More precise, less easily attacked
- Impact: Sharper positioning

## Revised Proposal

# DepthPEFT: Depth-Conditioned Parameter-Efficient Fine-Tuning for VLM Spatial Reasoning

## Problem Anchor
[Same as above — verbatim]

## Technical Gap

Existing methods for injecting depth into VLMs add depth as passive features — additional tokens, positional encodings, or visual inputs. These approaches provide geometric information but leave the VLM's adaptation mechanism (LoRA weights) fixed regardless of scene geometry.

**Gap**: No prior work conditions the PEFT mechanism itself on depth geometry. The adaptation is scene-agnostic — the same LoRA weights process a flat parking lot and a complex urban intersection identically.

**Insight**: Depth should condition not just WHAT the VLM sees, but HOW it adapts. Different scene geometries should activate different adaptation strategies.

## Method Thesis

- **One-sentence thesis**: Conditioning LoRA adaptation on scene depth geometry significantly improves VLM spatial reasoning, and the degree of conditioning required depends on the geometric complexity of the task.

- **Why smallest adequate intervention**: We add depth conditioning to existing LoRA fine-tuning — no new architectures, no extra tokens, no modified attention. The base model and standard PEFT infrastructure remain unchanged.

## Contribution Focus

- **Dominant contribution**: First systematic study of depth-conditioned PEFT for VLM spatial reasoning, demonstrating that conditioning the adaptation mechanism on depth geometry is more effective than treating depth as passive input
- **Supporting contribution**: Progressive conditioning framework — from gating (simplest) through FiLM (medium) to basis mixing (richest) — revealing how much depth conditioning is actually needed
- **Explicit non-contributions**: New depth estimator, new dataset, new VLM architecture

## Proposed Method: DepthPEFT Framework

### Complexity Budget

- **Frozen / reused**: Qwen2.5-VL-7B (ViT + Merger + LLM base), standard PEFT infrastructure
- **Shared across all variants**: DepthGeometryNet (~445K params) — depth map → z_geo ∈ R^256
- **New trainable per variant**: See variant descriptions
- **Intentionally excluded**: Question conditioning, multi-scale depth, auxiliary losses

### System Overview

```
Image → Qwen2.5-VL ViT (frozen) → visual tokens → Merger (frozen)
                                                        ↓
Depth map (P1) → DepthGeometryNet → z_geo ∈ R^256     LLM input
                        ↓                               ↓
              Depth Conditioning Module          LLM layers 20-27
                        ↓                        (frozen base weights
                 Condition the LoRA              + conditioned LoRA)
                                                        ↓
                                                     Answer
```

### Shared Component: DepthGeometryNet (~445K params)

```
depth_map (B, 1, H, W)
  → Sobel_x, Sobel_y → concat → (B, 3, H, W)
  → Conv2d(3, 64, 7, stride=4) + GELU
  → Conv2d(64, 128, 3, stride=2) + GELU
  → Conv2d(128, 256, 3, stride=2) + GELU
  → AdaptiveAvgPool2d(1, 1) → flatten
  → LayerNorm(256) → Linear(256, 256) → GELU
  → z_geo (B, 256)
```

### Level 1: DepthGate — Depth-Conditioned LoRA Gating (~9.5M trainable)

The simplest depth conditioning. Standard fixed LoRA weights, but their output is modulated by per-layer gate scalars derived from depth geometry.

```
Standard LoRA: delta_l = B_l @ A_l @ h    (fixed weights, learned normally)

Depth gating:
  g_l = sigmoid(Linear(z_geo, 1))           per target layer, per projection

output_l = frozen_proj(h) + g_l * delta_l

Total: Standard LoRA (~9M) + DepthGeometryNet (445K) + gating linears (16 × 257 ≈ 4K)
```

**What it tests**: Does scene-specific activation scaling of LoRA suffice?

### Level 2: DepthFiLM — Depth-Conditioned Scale+Shift (~9.6M trainable)

Richer conditioning via FiLM-style (Feature-wise Linear Modulation). Depth generates per-layer scale and shift for LoRA output.

```
Standard LoRA: delta_l = B_l @ A_l @ h    (fixed weights)

FiLM conditioning:
  γ_l, β_l = Linear(z_geo, 2 × d_out).chunk(2)   per layer, per projection

output_l = frozen_proj(h) + γ_l * delta_l + β_l

Total: Standard LoRA (~9M) + DepthGeometryNet (445K) + FiLM projectors (~100K)
```

**What it tests**: Does richer per-dimension depth conditioning help beyond gating?

### Level 3: DepthBasis (GeoLoRA) — Depth-Conditioned Basis Mixing (~9.17M trainable)

Richest conditioning. No fixed LoRA — K=6 basis LoRA pairs per layer, mixed by depth-derived routing weights to create scene-specific adapter weights.

```
LoRA Basis Bank: {(A_k, B_k)}_{k=1}^K per layer per projection

Routing:
  α_l = softmax(RouterMLP(z_geo + LayerEmbed(l)))   # (B, K)

Mixing:
  A_mixed = Σ_k α_k * A_k
  B_mixed = Σ_k α_k * B_k
  delta_l = B_mixed @ A_mixed @ h                    # per-sample different

Gated output:
  output_l = frozen_proj(h) + gate_l * delta_l

Total: Basis Banks (8.6M) + DepthGeometryNet (445K) + Router (70K) + Gates (<1K)
```

**What it tests**: Do multiple learned spatial processing strategies improve over single-LoRA conditioning?

### Training Plan (Same for All Levels)

- **Data**: P3 SpatialQA train split (NuScenes images + P1 depth maps)
- **Loss**: Standard next-token prediction on answers
- **Optimizer**: AdamW, weight_decay=0.01
- **LR**:
  - Level 1-2: depth_net: 1e-4, LoRA: 5e-5, gating/FiLM: 1e-3
  - Level 3: depth_net+router: 1e-4, bank: 5e-5, gates: 1e-3
- **Schedule**: Cosine decay, 5% warmup, 3 epochs
- **Precision**: bf16, effective batch size 16

### Failure Modes and Diagnostics

1. **DepthGate gates near 0.5 for all scenes** → conditioning signal too weak → increase depth_net LR
2. **DepthBasis routing collapse** → all scenes get same α → add entropy regularization on α
3. **All levels perform similarly** → depth conditioning too simple to need basis mixing → paper story becomes "simple gating suffices" (still publishable)
4. **No method beats standard LoRA** → depth maps not informative enough → try GT depth, check P1 quality

### Novelty and Elegance Argument

**Closest work**: SHINE (text-conditioned LoRA), Video2LoRA (visual-conditioned LoRA for generation)

**Exact difference**:
1. **Conditioning signal**: Geometric structure from depth (not text, not visual reference)
2. **Task domain**: Spatial understanding (not generation, not in-context learning)
3. **Progressive study**: First work to systematically compare levels of depth conditioning on PEFT

**Why focused**: One idea (depth conditions adaptation), three controlled levels of implementation, direct head-to-head comparison. Not a module pile-up.

## Claim-Driven Validation Sketch

### Claim 1: Depth-conditioned PEFT outperforms standard PEFT for spatial reasoning
- Experiment: DepthGate vs. Standard LoRA (matched params) on SpatialQA
- Baselines: Zero-shot Qwen2.5-VL-7B, Standard LoRA (already run)
- Metric: Overall accuracy + per-type accuracy (24 types)
- Expected: DepthGate > Standard LoRA, especially on depth-dependent types (distance, height, occlusion)

### Claim 2: Depth geometry signal is causally important
- Experiment: DepthGate(predicted) vs DepthGate(shuffled) vs DepthGate(zero)
- Metric: Accuracy degradation
- Expected: Predicted > Shuffled > Zero

### Claim 3: Richer conditioning provides diminishing or clear returns
- Experiment: DepthGate vs DepthFiLM vs DepthBasis (GeoLoRA)
- Metric: Accuracy delta over Standard LoRA baseline
- Expected: One of two outcomes:
  - (A) Gating is enough → paper story is "simple conditioning suffices"
  - (B) Basis mixing clearly wins → paper story is "scene-specific strategies are needed"
  - Both outcomes are publishable.

## Experiment Handoff Inputs

- Must-prove claims: Depth conditioning > no conditioning; depth signal is causal
- Must-run ablations: 3 conditioning levels, depth quality ablation
- Critical datasets: SpatialQA (primary), optional CV-Bench (transfer)
- Highest-risk assumption: That DepthGeometryNet produces a useful z_geo

## Compute & Timeline Estimate

| Experiment | GPU Hours |
|-----------|-----------|
| DepthGate training | ~8h |
| DepthFiLM training | ~8h |
| DepthBasis (GeoLoRA) training | ~10h |
| Standard LoRA (if re-run needed) | ~6h |
| Evaluation suite (all methods × depth modes) | ~6h |
| **Total** | **~38h** on RTX 5880 |

Timeline: ~1 week of continuous server time
