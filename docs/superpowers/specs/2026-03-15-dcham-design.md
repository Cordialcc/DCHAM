# DCHAM: Depth-Conditioned HyperAttention Module

## Design Specification for Master's Thesis P2

**Date**: 2026-03-15
**Status**: Approved by user, pending implementation plan

---

## 1. Problem Statement

Current VLMs (e.g., Qwen2.5-VL) lack spatial reasoning capability. Existing approaches to inject depth information into VLMs all share a fundamental limitation: the network parameters that process visual features are **fixed** regardless of the scene's geometric structure.

| Approach | Limitation |
|----------|-----------|
| Depth as positional encoding (SD-VLM, SpaceDrive) | Additive signal, doesn't change processing |
| Feature alignment (VIRAL, SURGE) | Regularization loss, fixed processing params |
| Dual-stream MoT (G2VLM) | Fixed expert weights per stream |
| Depth as separate modality (DeepSight) | Modified encoder, still fixed attention params |

**Our key insight**: Depth should not be a feature to fuse — it should **generate** the processing parameters themselves. Different scenes require different attention behaviors.

---

## 2. Core Innovation

**Depth-Conditioned HyperAttention**: A HyperNetwork uses depth map features to dynamically generate attention projection matrices (W_K, W_V) from a learnable basis kernel library. Combined with a tri-modal Q/K/V design where each comes from a different modality.

Three modalities, three distinct roles:

| Role | Source | Function |
|------|--------|----------|
| **Q** (Query) | Text-conditioned learnable tokens | "What spatial information to extract" |
| **W_K, W_V** (Processing) | Depth-generated via HyperNetwork | "How to process visual information" |
| **K, V input** (Content) | RGB visual tokens from ViT | "Visual content to be processed" |

---

## 3. Architecture

### 3.1 Overall Integration with Qwen2.5-VL-7B

```
Image ──→ Qwen2.5-VL ViT (frozen) ──→ visual_tokens ∈ R^{N × 1280}
                                            │
                    ┌───────────────────────┤
                    │                       │
                    ↓                       ↓
              ┌──────────┐          ┌───────────────┐
              │  DCHAM    │          │ Standard      │
              │ (new,     │          │ Merger        │
              │  trained) │          │ (frozen)      │
              │           │          │ 2×2 → MLP     │
              │ +depth_map│          │ 5120→3584     │
              │ +text_emb │          │               │
              └────┬─────┘          └───────┬───────┘
                   │                        │
          spatial_tokens             merged_visual_tokens
          R^{16 × 3584}             R^{N/4 × 3584}
                   │                        │
                   └──────────┬─────────────┘
                              ↓
                        Concatenate
                   R^{(N/4 + 16) × 3584}
                              ↓
                   Qwen2.5 LLM (+ LoRA rank=64)
                              ↓
                           Answer
```

DCHAM operates on ViT output (before merger), produces 16 spatial tokens projected to LLM dimension, concatenated with standard merged visual tokens.

### 3.2 DCHAM Internal Structure

```
DCHAM Module
├── Component 1: DepthFeatureNet (~77K params)
│   Input:  P1 depth map R^{1 × H × W}
│   Output: fine features, coarse features, global vector
│   Role:   Provide input signal for HyperNetwork
│
├── Component 2: HyperAttentionHead × 2 (~1.75M params)
│   ├── Fine scale: local boundary/surface spatial relationships
│   └── Coarse scale: global scene layout relationships
│   Each contains:
│     ├── Context Encoder (global depth → context vector)
│     ├── Coefficient Generator (context → basis kernel weights α)
│     └── Basis Kernel Library (n=12 low-rank basis matrices, r=24)
│   Role:   THE CORE INNOVATION — depth generates attention params
│
├── Component 3: Spatial Queries + Text Conditioning (~2.37M params)
│   ├── 16 learnable query tokens (8 fine + 8 coarse)
│   └── Pooled text → MLP → additive modulation
│   Role:   Define what spatial information to extract
│
└── Output: Scale embedding + Linear projection (~1.84M params)
    Role:   Project to LLM dimension
```

### 3.3 Qwen2.5-VL-7B Dimensions Reference

| Parameter | Value |
|-----------|-------|
| ViT hidden_size (d_vit) | 1280 |
| ViT num_heads | 16 |
| ViT depth | 32 layers |
| ViT patch_size | 14 × 14 |
| Merger: spatial_merge_size | 2 (2×2 = 4× reduction) |
| LLM hidden_size (d_lm) | 3584 |
| LLM layers | 28 |

### 3.4 DCHAM Internal Dimensions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 512 | DCHAM internal dim (H × d_head) |
| d_head | 64 | Per-head attention dimension |
| H | 8 | Number of attention heads |
| n | 12 | Number of basis kernels |
| r | 24 | Rank of each basis kernel |
| M | 16 | Spatial query tokens (8 fine + 8 coarse) |
| d_depth | 128 | Depth feature dimension from CNN |

---

## 4. Component Details

### 4.1 DepthFeatureNet

**Input**: P1 depth map, single channel, R^{1 × H_img × W_img}

**Architecture**: 2 convolutional layers + adaptive pooling

```python
depth_map (1ch)
    → Conv2d(1, 64, 7, stride=4, pad=3) + GELU     # /4
    → Conv2d(64, 128, 3, stride=2, pad=1) + GELU    # /8
    → feature_map R^{128 × h/8 × w/8}
    ├── AdaptiveAvgPool2d(match_vit_grid) → flatten → fine_feats   R^{N_f × 128}
    ├── AdaptiveAvgPool2d(4, 4)           → flatten → coarse_feats R^{16 × 128}
    └── AdaptiveAvgPool2d(1, 1)           → flatten → global_feat  R^{128}
```

**Parameters**: ~77K (2 Conv2d layers + biases)
**Why learnable (not Sobel)**: Learns task-relevant depth features (curvature, multi-scale context) that Sobel + statistics cannot capture. Still takes P1's depth map as direct input — maintains P1 connection.

### 4.2 HyperAttentionHead (× 2 instances: fine + coarse)

This is the core innovation. Each instance has independent parameters.

#### 4.2.1 Context Encoder

Maps global depth feature to a context vector.

```
global_feat R^{128}
    → Linear(128, 128) → GELU → LayerNorm
    → Linear(128, 256) → GELU → LayerNorm
    → context R^{256}
```

Parameters per instance: ~50K

#### 4.2.2 Coefficient Generator

Maps context to per-head basis kernel combination coefficients.

```
context R^{256}
    → Linear(256, H × n) → reshape(H, n) → softmax(dim=-1) → α_K R^{H × n}
    → Linear(256, H × n) → reshape(H, n) → softmax(dim=-1) → α_V R^{H × n}
```

Softmax ensures convex combination (coefficients sum to 1 per head).

Parameters per instance: ~50K

#### 4.2.3 Basis Kernel Library

The learnable component. n=12 basis matrices, each low-rank (r=24).

Each basis kernel i represents a projection R^{d_vit} → R^{d_head}:
- B_K[i] = U_K[i] · V_K[i]^T where U_K[i] ∈ R^{d_head × r}, V_K[i] ∈ R^{d_vit × r}

Dynamic assembly for head h:
```
W_K^h = Σ_{i=1}^{n} α_K[h,i] · (U_K[i] · V_K[i]^T)   ∈ R^{d_head × d_vit}
```

Efficient computation (never materializes full W_K^h):
```
For visual_tokens X ∈ R^{N × d_vit}:

Step 1: Z_K = einsum('nd, kdr -> nkr', X, V_K)        # R^{N × n × r}
Step 2: K_all = einsum('nkr, kfr -> nkf', Z_K, U_K)   # R^{N × n × d_head}  (f indexes d_head)
Step 3: K = einsum('hk, nkf -> nhf', α_K, K_all)      # R^{N × H × d_head}  (h indexes H heads, f indexes d_head)
```

Same for V projection with U_V, V_V, α_V.

Parameters per instance:
- U_K: n × d_head × r = 12 × 64 × 24 = 18,432
- V_K: n × d_vit × r = 12 × 1280 × 24 = 368,640
- U_V, V_V: same
- Basis kernels per instance: ~774K

Total per HyperAttentionHead instance (all sub-components):
- Basis kernels: 774K, Context Encoder: ~50K, Coefficient Generator: ~50K
- **Per instance total: ~874K**
- **Total for 2 instances: ~1.75M**

Note: V_K (368K) is the largest single parameter tensor because it projects from d_vit=1280 into the rank space. This is inherent to the low-rank factorization and cannot be reduced without lowering expressivity.

#### 4.2.4 Tri-Modal Attention

```
Q: R^{M × H × d_head}  ← text-conditioned spatial queries
K: R^{N × H × d_head}  ← depth-generated projection of visual tokens
V: R^{N × H × d_head}  ← depth-generated projection of visual tokens

Attention:
  scores = einsum('mhd, nhd -> hmn', Q, K) / sqrt(d_head)
  weights = softmax(scores, dim=-1)        # R^{H × M × N}
  output = einsum('hmn, nhd -> mhd', weights, V)   # R^{M × H × d_head}
  output = reshape(M, H * d_head)          # R^{M × d_model}
```

### 4.3 Spatial Queries + Text Conditioning

```python
# 16 learnable query tokens
Q_base = nn.Parameter(randn(16, d_model))           # R^{16 × 512}

# Lightweight text conditioning
text_pool = text_embeddings.mean(dim=0)               # R^{3584}
text_mod = MLP(text_pool)                              # R^{512}
    # MLP: Linear(3584, 512) → GELU → Linear(512, 512)

Q = Q_base + text_mod.unsqueeze(0)                    # R^{16 × 512} (broadcast)
Q_fine, Q_coarse = Q[:8], Q[8:]                       # 8 each

# Project to multi-head
Q = head_proj(Q).view(M, H, d_head)                  # R^{M × H × 64}
```

Parameters: ~2.37M (dominated by Linear(3584, 512) = 1.84M)

### 4.4 Output

```python
# Scale embedding (2 scales)
S_fine   += scale_embed(0)     # R^{8 × 512}
S_coarse += scale_embed(1)     # R^{8 × 512}

# Concatenate (no cross-scale attention needed — LLM handles this)
S = cat([S_fine, S_coarse])    # R^{16 × 512}

# Project to LLM dimension
spatial_tokens = LayerNorm(Linear(512, 3584)(S))  # R^{16 × 3584}
```

---

## 5. Parameter Budget

| Component | Parameters | % of DCHAM |
|-----------|-----------|------------|
| DepthFeatureNet (2 conv layers) | 77K | 1.3% |
| HyperAttentionHead × 2 (basis kernels + context enc + coeff gen) | 1.75M | 29.0% |
| Spatial Queries + Text Conditioning (incl. Linear 3584→512) | 2.37M | 39.3% |
| Output Projection + Scale Embed + Norms | 1.84M | 30.5% |
| **DCHAM Total** | **~6M** | **100%** |
| LoRA (rank=64, all 28 LLM layers, q/k/v/o) | ~40-51M | — |
| **All Trainable** | **~46-57M** | — |

Note: LoRA param count depends on GQA configuration. Qwen2.5-VL-7B uses num_key_value_heads=4 (GQA). LoRA on Q/O projections uses full 3584 dim; on K/V uses 512 dim. Exact count depends on PEFT library conventions.

Frozen: Qwen2.5-VL ViT (all 32 layers) + LLM base weights

## 6. GPU Memory Estimate (RTX 5880 48GB)

```
Qwen2.5-VL-7B frozen (bf16):      ~14 GB
DCHAM (bf16, 6M):                    0.01 GB
LoRA (bf16, ~45M):                   0.09 GB
Optimizer (AdamW bf16, 2× trainable): 0.2 GB
Gradients (bf16):                    0.1 GB
ViT activations (frozen, no grad):  ~2 GB
DCHAM activations + gradients:      ~1 GB
LLM activations + LoRA gradients:  ~20-24 GB
────────────────────────────────────────────
Estimated total:                   ~38-42 GB  ✓ fits 48GB

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
- **Depth maps**: Generated by P1's depth estimator (or DepthAnythingV2 as fallback)
- **Format**: image + depth_map + spatial_list + multi-turn QA (GR3D format)

### 7.2 Loss

```
L_total = L_LM                          (standard language modeling loss on QA)
        + λ · L_depth_aux  (optional)   (auxiliary depth relationship prediction)
```

L_depth_aux (optional): predict relative depth ordering between object pairs from spatial tokens. Provides direct geometric supervision to DCHAM. λ = 0.1-0.5.

### 7.3 Training Schedule

1. **Warmup** (first 5% steps): Train only DCHAM module (LLM LoRA frozen)
   - Lets depth feature net and HyperNetwork initialize before LLM adaptation
2. **Joint** (remaining 95%): Train DCHAM + LoRA jointly
3. Learning rate: 1e-4 (DCHAM), 2e-5 (LoRA), cosine decay

### 7.4 Frozen Components

- Qwen2.5-VL ViT: all layers frozen
- Qwen2.5-VL Merger: frozen
- Qwen2.5-VL LLM: base weights frozen, LoRA adapters trained

---

## 8. Thesis Narrative

**Chapter 3 (P1)**: Self-supervised monocular depth estimation in dynamic scenes (PGA-Depth)
→ Contribution: How to obtain accurate depth maps from monocular images

**Chapter 4 (P2)**: Depth-Conditioned HyperAttention for VLM spatial reasoning (DCHAM)
→ Contribution: How depth information can **generate** (not just inform) the VLM's visual processing
→ Connection to P1: Uses P1's depth maps as direct input to DCHAM
→ Connection to P3: Trained and evaluated on P3's spatial reasoning dataset

**Chapter 5 (P3)**: Spatial understanding dataset for autonomous driving VLMs (PGDG/SpatialQA)
→ Contribution: Systematic dataset and evaluation framework for VLM spatial reasoning

**Overall thesis title suggestion**:
"面向自动驾驶的深度感知视觉理解：从单目深度估计到视觉语言模型空间推理"

---

## 9. Paper Contributions

1. **Paradigm**: "Depth as processing generator" — depth maps generate attention parameters rather than serving as features. Fundamentally different from all prior depth-VLM integration methods.

2. **Mechanism**: Factorized HyperAttention with basis kernel library — depth-conditioned dynamic combination of low-rank basis matrices to generate attention projections. Parameter-efficient yet expressive.

3. **Architecture**: Tri-modal Q/K/V functional decomposition — Text generates queries, Depth generates processing (K/V projections), RGB provides content. Each modality has a distinct, non-interchangeable role.

4. **Evaluation**: Comprehensive experiments on P3's SpatialQA and standard VLM benchmarks.

---

## 10. Key Distinctions from Related Work

| Method | Depth Role | Attention Params | Our Distinction |
|--------|-----------|-----------------|-----------------|
| SD-VLM | Additive PE | Fixed W_Q/W_K/W_V | We GENERATE W_K/W_V from depth |
| SpaceDrive | 3D coordinate PE | Fixed | Same as above |
| VIRAL | Alignment target | Fixed | Depth is not a regularizer but a generator |
| G2VLM (CVPR'26) | Geometric expert input | Fixed per expert | Our params are dynamic per scene |
| DeepSight | Modified encoder input | Fixed encoder | We add a module, don't modify encoder |
| FiLM methods | Generate scale+shift (2 params/channel) | Fixed base params | We generate ENTIRE projection matrices |
| **DCHAM** | **Generate W_K, W_V** | **Dynamic per scene** | — |

---

## 11. Ablation Study Plan

| Experiment | What it tests |
|------------|--------------|
| Remove HyperNetwork, use fixed W_K/W_V | Core innovation value |
| Remove text conditioning on queries | Text-guidance necessity |
| Single scale vs two scales | Multi-scale value |
| n=4,8,12,16 basis kernels | Basis library capacity |
| r=8,16,24,32 rank | Per-basis expressivity |
| M=8,16,24 spatial tokens | Token count sweet spot |
| Remove DepthFeatureNet, use Sobel features | Learned vs hand-crafted depth features |
| Replace depth with random noise | Depth information necessity |
| Visualize α coefficients across scenes | Basis kernel specialization analysis |
| Visualize attention maps | Spatial reasoning interpretability |
