# Research Proposal: Depth-Conditioned VLM Spatial Reasoning Enhancement

## Problem Anchor

- **Bottom-line problem**: Qwen2.5-VL-7B lacks spatial reasoning capability in autonomous driving scenes. Depth maps from P1 (PGA-Depth) exist as a rich geometric signal, but current methods either ignore depth or integrate it as passive features without changing how the VLM processes information.

- **Must-solve bottleneck**: Need a method that uses depth maps to MEANINGFULLY improve VLM spatial reasoning — not just fuse depth as features, but change the VLM's processing in a depth-aware way. The method must clearly outperform both (a) zero-shot VLM and (b) standard LoRA fine-tuning without depth.

- **Non-goals**:
  - NOT building a new depth estimator (P1 already provides depth)
  - NOT creating a new dataset (P3 SpatialQA already exists with 132K QA, 24 types)
  - NOT multi-model generalization (single VLM sufficient for thesis)
  - NOT real-time deployment optimization

- **Constraints**:
  - Hardware: 1× RTX 5880 48GB, single GPU
  - Base model: Qwen2.5-VL-7B-Instruct (already on server)
  - Data: NuScenes images + P1 depth maps + P3 SpatialQA (already split, baselines done)
  - Existing baselines: zero-shot Qwen2.5-VL-7B + standard LoRA fine-tuning (already run)
  - Venue: Master's thesis Chapter 4, potentially standalone paper
  - Time: needs to complete within weeks, not months
  - Preference: LoRA-based fine-tuning approach

- **Success condition**: A depth-aware method that:
  1. Clearly beats standard LoRA (without depth) on SpatialQA
  2. Shows depth map quality matters (random/zero depth degrades performance)
  3. Is simple enough to implement, train, and explain convincingly
  4. Connects naturally to P1 (uses its depth) and P3 (evaluated on its dataset)

## Technical Gap

### Current Methods and Why They Fail

All existing depth-VLM integration methods share one limitation: **the VLM's attention weights remain fixed regardless of scene geometry**. Whether the scene is a flat parking lot or a complex urban intersection with occluded objects, the VLM processes them identically.

| Method | How depth is used | VLM processing | Problem |
|--------|------------------|----------------|---------|
| No depth (standard LoRA) | Not used | Fixed LoRA weights | Misses geometric cues |
| Depth as PE (SD-VLM) | Additive encoding | Unchanged | Signal too weak |
| Depth tokens (Spa3R-style) | Prepended tokens | Unchanged weights | Token budget cost, passive |
| Depth as second image | Multi-image input | Unchanged weights | VLM treats as visual, not geometric |
| Depth as text prompt | Numeric description | Unchanged weights | Loses spatial structure |

### Why Naive Fixes Are Insufficient

1. **Depth as extra image**: Qwen2.5-VL can take multiple images, but it treats the depth map as another visual input — it doesn't understand it geometrically. The ViT was not trained on depth maps.
2. **Depth as text**: "Object at 5.2m" loses the rich spatial structure of the full depth field.
3. **Depth tokens**: Prepending depth features as tokens adds information but doesn't change how the model attends to or processes the visual scene.
4. **Larger LoRA**: More parameters without geometric conditioning = memorize more patterns, not reason better spatially.

### The Smallest Adequate Intervention

The key insight: instead of adding depth as passive information, use it to **condition the adaptation itself** — make the LoRA weights scene-specific based on geometry.

This is the minimum mechanism that changes the VLM's actual processing (not just its input) based on depth.

## Method Thesis

- **One-sentence thesis**: Scene depth geometry should determine scene-specific adapter weights for the VLM, so that different geometric structures activate different spatial processing strategies.

- **Why this is the smallest adequate intervention**: It adds depth conditioning to the adaptation mechanism (LoRA) rather than requiring new architectures, extra tokens, or modified attention. The base model stays frozen, only the adapter behavior changes per scene.

- **Why this route is timely**: Input-conditioned parameter adaptation is emerging in 2026 generative models (Video2LoRA, SHINE, FLOWER AdaLN-Zero). Applying it to VLM spatial understanding is natural and under-explored.

## Contribution Focus

- **Dominant contribution**: Depth-conditioned dynamic LoRA — depth maps determine per-scene adapter weights for VLM spatial reasoning
- **Optional supporting contribution**: Systematic evaluation framework showing depth signal is causally important (via ablation modes)
- **Explicit non-contributions**: New depth estimation method, new dataset, multi-VLM generalization

## Proposed Method

### Complexity Budget

- **Frozen / reused**: Qwen2.5-VL-7B ViT + Merger + LLM base weights (~7.6B frozen)
- **New trainable**: Depth encoder (~445K) + Routing/mixing mechanism + Gate scalars
- **Tempting additions intentionally excluded**:
  - Question-conditioned routing (adds complexity without clear benefit)
  - Multi-scale depth features (single global descriptor suffices)
  - Auxiliary losses (pure LM loss is simpler)

### System Overview

```
Image → Qwen2.5-VL ViT (frozen) → visual tokens → Merger (frozen) → LLM input

Depth map (P1) → Depth Encoder → scene geometry descriptor z_geo
                                    ↓
                          Dynamic LoRA weight generation
                                    ↓
                     LLM upper layers: frozen base + depth-conditioned LoRA (gated)
                                    ↓
                                 Answer
```

### Core Mechanism: GeoLoRA (Geometry-Conditioned Dynamic LoRA)

**OPTION A: Full Basis Mixing (Current GeoLoRA)**
- K=6 LoRA basis pairs per layer, mixed by routing weights
- GeometryRouter maps z_geo → per-layer softmax mixing weights α
- Dynamic per-sample: different scenes get different effective LoRA weights
- ~9.17M trainable params
- **Pros**: Most expressive, cleanest novelty story, learned vocabulary of strategies
- **Cons**: Most complex, more to debug, harder to explain

**OPTION B: Simplified Depth-Conditioned LoRA (DepthLoRA)**
- Single LoRA pair per target layer, but A/B matrices generated by a small hypernetwork from z_geo
- HyperNet: z_geo → (ΔA, ΔB) per layer (lightweight linear mapping)
- Standard LoRA + depth-conditioned perturbation
- ~5-7M trainable params
- **Pros**: Simpler, easier to implement, still depth-conditioned
- **Cons**: Less expressive, weaker novelty story

**OPTION C: Depth-Conditioned Gating of Standard LoRA (GateLoRA)**
- Standard fixed LoRA (trained normally), but output gated by depth-derived signal
- z_geo → per-layer gate scalars (separate from zero-init gates)
- LoRA learns general spatial patterns, depth gates modulate per-scene activation
- ~9M trainable (standard LoRA) + ~0.5M (depth gating)
- **Pros**: Simplest, most interpretable, easy to implement on top of existing LoRA
- **Cons**: Weakest conditioning (gate vs. full weight generation), less novel

### Recommended Route: OPTION A (Full Basis Mixing — GeoLoRA)

Despite higher complexity, Option A has the strongest paper story:
1. **Novelty**: Learned basis vocabulary + geometry routing is genuinely new for VLM spatial reasoning
2. **Ablations are clean**: Can ablate to Option B (K=1) and Option C (uniform α + gating) as baselines
3. **Expressiveness**: K=6 bases can represent distinct spatial strategies (depth/distance/occlusion processing)
4. **Thesis fit**: The most substantial Chapter 4 contribution

However, if Option A proves too complex or doesn't outperform B/C, the fallback is clear.

### Training Plan

- **Data**: P3 SpatialQA train split (NuScenes images + P1 depth maps)
- **Loss**: Standard next-token prediction on QA answers (no auxiliary losses)
- **Optimizer**: AdamW with 3 LR groups (depth/router: 1e-4, bank: 5e-5, gates: 1e-3)
- **Schedule**: Cosine decay, 5% warmup, 3 epochs
- **Precision**: bf16
- **Effective batch**: 16 (batch=2 × grad_accum=8)

### Failure Modes and Diagnostics

1. **Gates stay near zero**: Depth conditioning doesn't activate → check depth encoder gradients
2. **All scenes get same α**: Routing collapses → increase temperature / reduce weight decay on router
3. **Worse than standard LoRA**: Depth hurts → check depth map quality, try GT depth
4. **Training instability**: Dynamic weights fluctuate → reduce bank LR, increase gate warmup

### Novelty and Elegance Argument

**Closest work**: SHINE (text-conditioned LoRA for ICL), Video2LoRA (visual ref → LoRA for generation)
**Exact difference**: GeoLoRA uses GEOMETRIC STRUCTURE (not text/visual reference) to condition UNDERSTANDING LoRA (not generation). The conditioning signal (depth) and the task domain (VLM spatial reasoning) are both novel.

## Claim-Driven Validation Sketch

### Claim 1: Depth-conditioned dynamic LoRA improves spatial reasoning over static LoRA
- Experiment: GeoLoRA vs. Static LoRA (matched budget) on SpatialQA
- Metric: Exact match accuracy, per-type accuracy (24 types)
- Expected: GeoLoRA > Static LoRA, especially on distance/depth-dependent questions

### Claim 2: Depth geometry signal is causally important
- Experiment: GeoLoRA(predicted depth) vs. GeoLoRA(random depth) vs. GeoLoRA(zero depth)
- Metric: Accuracy degradation when depth is ablated
- Expected: Predicted > Corrupted > Random > Zero

### Claim 3: Dynamic routing adapts to scene geometry
- Experiment: GeoLoRA(full) vs. GeoLoRA(uniform α=1/K)
- Metric: Accuracy + visualization of α across scene types
- Expected: Full routing > Uniform, and different scene types show different α patterns

## Compute & Timeline Estimate

- Training GeoLoRA: ~8 GPU hours (3 epochs, 132K samples)
- Training baselines (Static LoRA, Uniform α): ~6 GPU hours each
- Evaluation suite: ~4 GPU hours
- Ablation experiments: ~20 GPU hours total
- **Total: ~40-50 GPU hours** on RTX 5880
- **Timeline**: 1-2 weeks of server time
