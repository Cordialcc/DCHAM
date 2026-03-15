# GeoLoRA: Geometry-Conditioned Dynamic LoRA for VLM Spatial Reasoning

## Design Specification for Master's Thesis P2

**Date**: 2026-03-16
**Status**: Approved by user, pending spec review
**Supersedes**: 2026-03-15-dcham-design.md

---

## 1. Problem Statement

Current VLMs (e.g., Qwen2.5-VL) lack spatial reasoning capability. Existing approaches to inject depth information into VLMs all share a fundamental limitation: the VLM's processing parameters remain **fixed** regardless of the scene's geometric structure. A flat indoor scene and a complex outdoor driving scene get processed by identical attention weights.

| Approach | Limitation |
|----------|-----------|
| Depth as positional encoding (SD-VLM, SpaceDrive) | Additive signal, processing parameters unchanged |
| Feature alignment (VIRAL, SURGE) | Regularization loss, processing parameters unchanged |
| Dual-stream MoT (G²VLM, CVPR'26) | Fixed expert weights per stream |
| 3D feature injection (VGGDrive, CVPR'26) | Enriches visual tokens, processing parameters unchanged |
| Spatial field cross-attention (Spa3R) | Enriches visual tokens, processing parameters unchanged |
| Geometric imagination (3DThinker, CVPR'26) | Adds mental tokens, processing parameters unchanged |
| Separate HyperAttention module (DCHAM) | Generates side-module params, LLM itself unchanged |

**Our key insight**: Depth should not be a feature to fuse — it should **modulate the VLM itself**. Different scenes with different geometric structures should activate different processing strategies within the LLM. The geometry of each scene should determine HOW the VLM processes visual information, not just WHAT it sees.

**Novelty position**: This work draws from the "input-conditioned parameter adaptation" paradigm emerging in 2026's generative models (Video2LoRA, SHINE, FLOWER's AdaLN-Zero) and applies it to VLM spatial reasoning — the first geometry-conditioned dynamic adapter modulation for spatial understanding.

---

## 2. Core Innovation

**Geometry-Conditioned LoRA Basis Mixing**: A raw depth map (from P1) is processed into a scene geometry descriptor, which then dynamically mixes a bank of learned LoRA bases to create scene-specific adapter weights for the LLM's upper layers. Each scene's geometry literally determines a different "spatial processing strategy" from a learned vocabulary.

Three key distinctions:

| Dimension | Prior Work | GeoLoRA |
|-----------|-----------|---------|
| What depth modulates | Input features / side modules | The LLM's own attention layers |
| Adaptation granularity | Fixed for all inputs | Dynamic per scene geometry |
| Conditioning signal | Text (SHINE) / visual reference (Video2LoRA) | Geometric structure (depth + surface normals) |

**Contribution claim**: *"First geometry-conditioned dynamic model modulation for VLM spatial reasoning — depth maps from monocular estimation determine scene-specific adapter weights, enabling the VLM to adapt its processing strategy to each scene's geometric structure."*

---

## 3. Architecture

### 3.1 Overall Integration with Qwen2.5-VL-7B

```
Image ──→ Qwen2.5-VL ViT (frozen) ──→ visual_tokens
                                            │
                                            ↓
                                      Standard Merger (frozen)
                                            │
                                            ↓
                                     merged_visual_tokens
                                            │
Depth map (P1) ──→ DepthGeometryNet ──→ z_geo ∈ R^{d_geo}
                                            │
                                            ↓
                                   ┌─────────────────┐
                                   │ Geometry Router  │
                                   │ z_geo → α_l     │
                                   │ per-layer mixing │
                                   │ weights          │
                                   └────────┬────────┘
                                            │
                     ┌──────────────────────┤
                     ↓                      ↓
              ┌─────────────┐        ┌─────────────┐
              │ LoRA Bank   │        │ LoRA Bank   │
              │ Layer 20    │  ...   │ Layer 27    │
              │ K=6 bases   │        │ K=6 bases   │
              │ Mix by α_20 │        │ Mix by α_27 │
              └──────┬──────┘        └──────┬──────┘
                     │                      │
                     ↓                      ↓
              Qwen2.5 LLM (frozen base weights)
              Layers 0-19: unchanged
              Layers 20-27: + geometry-mixed LoRA (gated)
                            │
                         Answer
```

GeoLoRA operates inside the LLM itself. No separate module, no extra tokens. Depth modulates the LLM's own attention layers.

### 3.2 Qwen2.5-VL-7B Dimensions Reference

| Parameter | Value |
|-----------|-------|
| ViT hidden_size (d_vit) | 1280 |
| ViT num_heads | 16 |
| ViT depth | 32 layers |
| ViT patch_size | 14 × 14 |
| Merger: spatial_merge_size | 2 (2×2 = 4× reduction) |
| LLM hidden_size (d_lm) | 3584 |
| LLM num_attention_heads | 28 |
| LLM num_key_value_heads | 4 (GQA) |
| LLM head_dim | 128 |
| LLM layers | 28 |
| q_proj / o_proj dim | 3584 → 3584 |
| k_proj / v_proj dim | 3584 → 512 (4 × 128) |

### 3.3 GeoLoRA Internal Dimensions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_geo | 256 | Scene geometry descriptor dimension |
| K | 6 | Number of LoRA bases (spatial processing strategies) |
| r | 16 | LoRA rank per basis |
| L | 8 | Number of target LLM layers (20-27) |
| Target projections | q_proj, v_proj | Q determines what to attend to, V determines what to extract |

---

## 4. Component Details

### 4.1 DepthGeometryNet (~445K params)

**Input**: P1 depth map, single channel, R^{1 × H_img × W_img}
**Output**: Scene geometry descriptor z_geo ∈ R^{d_geo}

**Architecture**: Depth gradient augmentation + 3 convolutional layers + global pooling + MLP

```python
depth_map (1ch)
    → Sobel_x, Sobel_y (differentiable, no params)       # Surface normal approx
    → Concatenate: R^{3 × H × W}  (depth + ∂d/∂x + ∂d/∂y)
    → Conv2d(3, 64, 7, stride=4, pad=3) + GELU            # /4
    → Conv2d(64, 128, 3, stride=2, pad=1) + GELU           # /8
    → Conv2d(128, 256, 3, stride=2, pad=1) + GELU          # /16
    → feature_map R^{256 × h' × w'}
    → AdaptiveAvgPool2d(1, 1) → flatten                    # R^{256}
    → LayerNorm(256) → Linear(256, d_geo) → GELU
    → z_geo R^{d_geo=256}
```

**Parameter breakdown:**

| Layer | Params |
|-------|--------|
| Conv2d(3, 64, 7×7) | 9.5K |
| Conv2d(64, 128, 3×3) | 73.9K |
| Conv2d(128, 256, 3×3) | 295.2K |
| LayerNorm(256) | 0.5K |
| Linear(256, 256) | 65.8K |
| **Total** | **~445K** |

**Why gradients**: Raw depth encodes "how far." Gradients encode "surface orientation" — a flat floor vs. vertical wall vs. sloped surface all have distinct gradient patterns. This provides structured geometric information that goes beyond scalar depth values.

**Depth map resolution**: Resolution-agnostic due to AdaptiveAvgPool2d. The depth map should match the original image resolution (e.g., 900×1600 for NuScenes). Depth maps are loaded as separate single-channel tensors, bypassing the ViT entirely, and fed directly to DepthGeometryNet.

### 4.2 LoRA Basis Bank (~8.6M params)

The core learnable component. K=6 basis LoRA pairs per target layer, per target projection.

Each basis represents a different learned "spatial processing strategy." The bank forms a vocabulary of processing behaviors that the router selects from based on scene geometry.

```
For each target layer l ∈ {20, 21, ..., 27} (L=8 layers):
  For each target projection p ∈ {q_proj, v_proj}:
    A_bank[l][p] ∈ R^{K × r × d_in}       # Down projection bases
    B_bank[l][p] ∈ R^{K × d_out × r}       # Up projection bases
```

**Per-layer parameter breakdown:**

| Projection | A per basis | B per basis | Per basis total | K=6 total |
|-----------|------------|------------|----------------|-----------|
| q_proj (3584→3584) | 16 × 3584 = 57K | 3584 × 16 = 57K | 114K | 688K |
| v_proj (3584→512) | 16 × 3584 = 57K | 512 × 16 = 8K | 65K | 393K |
| **Per layer** | | | | **1.08M** |
| **8 layers** | | | | **8.6M** |

**Initialization**: Xavier uniform for A, zeros for B (so initial LoRA output is zero, combined with zero-init gates this ensures the model starts from base behavior).

### 4.3 Geometry Router (~70K params)

Maps z_geo to per-layer mixing weights using a shared MLP with layer-specific embeddings.

```python
# Shared router with per-layer specialization
layer_embed = nn.Embedding(L, d_geo)          # L=8, d_geo=256

# Shared MLP (applied per layer with different input)
router_mlp = nn.Sequential(
    Linear(d_geo, d_geo), GELU,
    Linear(d_geo, K),                         # K=6
)

For each target layer l:
  input_l = z_geo + layer_embed(l)            # R^{B × d_geo}
  α_l = softmax(router_mlp(input_l), dim=-1)  # R^{B × K}
```

**Parameters**: Embedding(8, 256)=2K + Linear(256,256)=66K + Linear(256,6)=1.5K ≈ **70K**

Softmax ensures convex combination — coefficients sum to 1 per layer. Each scene's geometry produces a different mixture of the K bases, and different layers can receive different mixtures.

### 4.4 Dynamic LoRA Application

Efficient batched computation — no weight copying, no LoRA-Switch kernel needed.

```python
For each target layer l, projection p (e.g., q_proj or v_proj):

  # Step 1: Mix bases per sample
  # α_l: R^{B × K}, A_bank: R^{K × r × d_in}
  A_mixed = einsum('bk, kri -> bri', α_l, A_bank[l][p])   # R^{B × r × d_in}
  B_mixed = einsum('bk, kor -> bor', α_l, B_bank[l][p])   # R^{B × d_out × r}

  # Step 2: Apply mixed LoRA to hidden states (h = LLM hidden state at this layer)
  # h: R^{B × seq_len × d_in}, where d_in = 3584 for both q_proj and v_proj
  z = einsum('bri, bsi -> bsr', A_mixed, h)                # R^{B × seq × r}
  delta = einsum('bor, bsr -> bso', B_mixed, z)            # R^{B × seq × d_out}

  # Step 3: Add delta to the FROZEN projection output (not to h itself!)
  # frozen_out = frozen_proj_p(h)                           # R^{B × seq × d_out}
  # proj_out = frozen_out + gate_l_p * delta
  # gate_l_p: learnable scalar per (layer, projection), initialized to 0
```

**Injection point**: The LoRA delta is added to the **output of the frozen projection** (e.g., `frozen_q_proj(h) + gate * delta`), not to the hidden state h. This is standard LoRA behavior. For v_proj, d_out=512 (GQA), so delta ∈ R^{B × seq × 512} is added to the 512-dim v_proj output.

**Implementation**: Subclass each target attention layer's q_proj/v_proj Linear. The subclass wraps the frozen Linear and adds the dynamic LoRA delta in its `forward()`. Standard HuggingFace PEFT cannot be used directly since LoRA weights differ per sample.

**Why efficient**: The einsum naturally handles per-sample different LoRA across the batch dimension. No CUDA kernel fragmentation. Overhead per layer: 2 basis-mixing einsum + 2 matmul einsum + 1 gated addition.

**Zero-init gates**: gate_l_p is a learnable nn.Parameter initialized to 0.0. This means at training start, GeoLoRA is completely transparent (delta × 0 = 0), and the model starts from base Qwen2.5-VL behavior. Gates gradually open during training as the model learns to use geometry modulation.

---

## 5. Parameter Budget

| Component | Parameters | % of GeoLoRA |
|-----------|-----------|-------------|
| DepthGeometryNet (3 conv + MLP + norms) | 445K | 4.9% |
| LoRA Basis Bank (6 bases × 8 layers × q,v) | 8.65M | 94.4% |
| Geometry Router (shared MLP + layer embed) | 70K | 0.8% |
| Gates (16 scalars) | <1K | ~0% |
| **GeoLoRA Total** | **~9.17M** | **100%** |

Frozen: Qwen2.5-VL ViT (all 32 layers) + Merger + LLM base weights (~7.6B frozen)

---

## 6. GPU Memory Estimate (RTX 5880 48GB)

```
Assumptions: batch_size=2, max_seq_len=2048 (incl. ~500 visual tokens + text)

Qwen2.5-VL-7B frozen (bf16):       ~14 GB
GeoLoRA trainable (bf16, ~9.2M):     0.02 GB
Optimizer (AdamW, 2× trainable):     0.04 GB
Gradients (bf16):                    0.02 GB
ViT activations (frozen, no grad):  ~2 GB
LLM activations + LoRA gradients:  ~20-24 GB
DepthGeometryNet activations:       ~0.1 GB
Dynamic LoRA intermediates:         ~0.05 GB
─────────────────────────────────────────────
Estimated total:                   ~36-40 GB  ✓ comfortable fit

Note: With larger images (e.g., 1344×1344 → 2304 visual tokens after merge),
seq_len can reach 3000+. Use gradient checkpointing if needed — the dynamic
LoRA computation is compatible (basis mixing is re-computable from α and banks).

Recommended training config:
  batch_size_per_gpu: 2-4
  gradient_accumulation: 4-8
  precision: bf16
  optimizer: AdamW (or 8-bit Adam for extra headroom)
```

---

## 7. Training Strategy

### 7.1 Data

- **Primary**: P3 SpatialQA dataset (132K QA pairs, 24 spatial reasoning types)
- **Depth maps**: Generated by P1's depth estimator (paired with each image)
- **Format**: image + depth_map + spatial_list + multi-turn QA (GR3D format)

### 7.2 Loss

```
L_total = L_LM    (standard next-token prediction on QA answers)
```

No auxiliary losses. Geometry routing learns implicitly from the language modeling signal.

### 7.3 Training Schedule

Single-phase end-to-end with zero-init gates (gates provide natural warmup):

```
Trainable: DepthGeometryNet + Router + LoRA Bank + Gates
Frozen: ViT, Merger, LLM base weights

Learning rates:
  DepthGeometryNet + Router: 1e-4
  LoRA Bank: 5e-5
  Gates: 1e-3 (faster — needs to "open" during training)

Schedule: cosine decay with 5% linear warmup
Optimizer: AdamW, weight_decay=0.01
Precision: bf16
Batch: 2-4 per GPU, gradient_accumulation=8
Epochs: 3
```

### 7.4 Frozen Components

- Qwen2.5-VL ViT: all layers frozen
- Qwen2.5-VL Merger: frozen
- Qwen2.5-VL LLM: base weights frozen (only dynamic LoRA trains)

---

## 8. Thesis Narrative

**Chapter 3 (P1)**: Self-supervised monocular depth estimation in dynamic scenes (PGA-Depth)
→ Contribution: How to obtain accurate depth maps from monocular images

**Chapter 4 (P2)**: Geometry-Conditioned Dynamic LoRA for VLM Spatial Reasoning (GeoLoRA)
→ Contribution: How scene geometry should modulate the VLM's processing — depth determines HOW the model thinks
→ Connection to P1: Uses P1's depth maps as the conditioning signal for dynamic LoRA generation
→ Connection to P3: Trained and evaluated on P3's spatial reasoning dataset

**Chapter 5 (P3)**: Spatial understanding dataset for autonomous driving VLMs (PGDG/SpatialQA)
→ Contribution: Systematic dataset and evaluation framework for VLM spatial reasoning

**Overall thesis title suggestion**:
"面向自动驾驶的深度感知视觉理解：从单目深度估计到几何条件化视觉语言模型空间推理"

---

## 9. Paper Contributions

1. **Paradigm**: "Depth as model modulation" — depth maps dynamically condition the VLM's own adapter weights, rather than serving as features to fuse or parameters for a side module. Fundamentally different from all prior depth-VLM integration methods.

2. **Mechanism**: Geometry-conditioned LoRA basis mixing — a bank of K learned spatial processing strategies dynamically combined per scene via depth-derived mixing coefficients. Parameter-efficient yet expressive.

3. **Architecture**: Scene-adaptive VLM with zero-init gated application — the same base model exhibits different processing behavior for different scene geometries, with stable training via gradual gate opening.

4. **Evaluation**: Comprehensive experiments on P3's SpatialQA and standard VLM spatial benchmarks.

---

## 10. Key Distinctions from Related Work

| Method | Depth's Role | What Changes | Dynamic? | Domain |
|--------|-------------|-------------|----------|--------|
| SD-VLM | Additive PE | Input features | No | VLM understanding |
| SpaceDrive | 3D coordinate PE | Input features | No | VLM understanding |
| VIRAL | Alignment target | Training loss | No | VLM understanding |
| G²VLM (CVPR'26) | Geometric expert input | Fixed expert weights | No | VLM understanding |
| VGGDrive (CVPR'26) | 3D feature injection | Visual tokens | No | VLA driving |
| Spa3R | Spatial field | Visual tokens | No | VLM understanding |
| 3DThinker (CVPR'26) | None (2D imagination) | Mental tokens | No | VLM understanding |
| SHINE | Text context → LoRA | LLM parameters | Yes | In-context learning |
| Video2LoRA | Visual ref → LoRA | Diffusion backbone | Yes | Video generation |
| GST-VLA | Gaussian tokens from depth | VLA action expert | Yes | Robotic action |
| **GeoLoRA (ours)** | **Depth → LoRA basis mix** | **LLM attention layers** | **Yes** | **VLM spatial reasoning** |

---

## 11. Ablation Study Plan

| # | Experiment | What it tests |
|---|------------|--------------|
| 1 | Remove geometry routing, use fixed uniform α = 1/K | Core: does scene-specific modulation matter? |
| 2 | Remove Sobel gradients, use raw depth only | Surface orientation information value |
| 3 | K=1,2,4,6,8 bases | Basis bank capacity sweet spot |
| 4 | r=4,8,16,32 rank | Per-basis expressivity |
| 5 | Layers 24-27 vs 20-27 vs 12-27 | How many layers need modulation? |
| 6 | q_proj only vs v_proj only vs q+v vs q+k+v | Which projections benefit most? |
| 7 | Replace with AdaLN-Zero only (scale+shift on LayerNorms) | LoRA mixing vs norm modulation |
| 8 | Replace depth with random noise | Does depth geometry actually matter? |
| 9 | Standard fixed LoRA (same total param budget) | Dynamic vs static adaptation |
| 10 | Replace with standard depth token injection (Spa3R-style) | Modulation vs token injection paradigm |
| 11 | Visualize α coefficients across scene types | Do different geometries get different routing? |
| 12 | Visualize gate values during training | Training dynamics of zero-init gates |
