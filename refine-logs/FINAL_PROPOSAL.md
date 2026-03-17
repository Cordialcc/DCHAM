# DepthPEFT: Depth-Conditioned Parameter-Efficient Fine-Tuning for VLM Spatial Reasoning

**Final Proposal (Post-Refinement v4)**
**Score: 8.6/10 | Verdict: REVISE (ceiling without empirical results)**
**Thread: 019cf9b0-098f-7061-af24-fd9d8c30bba9**

---

## 1. Problem Anchor

**Bottom-line problem.** Qwen2.5-VL-7B lacks spatial reasoning capability in autonomous driving scenes. Depth maps from P1 (PGA-Depth) provide rich geometric signal, but existing methods treat depth as passive input --- additional tokens, positional encodings, or visual features --- without changing how the VLM processes information.

**Must-solve bottleneck.** A depth-aware method must clearly outperform both (a) zero-shot Qwen2.5-VL-7B and (b) standard LoRA fine-tuning without depth on the SpatialQA benchmark.

**Non-goals.**
- New depth estimator (P1 already provides depth)
- New dataset (P3 SpatialQA already exists: 132K QA, 24 types)
- Multi-model generalization (single VLM sufficient for thesis)
- Real-time deployment optimization

**Constraints.**
- Hardware: 1x RTX 5880 48GB, single GPU
- Base model: Qwen2.5-VL-7B-Instruct (already on server)
- Data: NuScenes images + P1 depth maps + P3 SpatialQA (already split, baselines done)
- Venue: Master's thesis Chapter 4 (potentially standalone paper)
- Time: weeks, not months

**Success condition.** Depth-aware method that:
1. Clearly beats standard LoRA on SpatialQA
2. Shows depth quality matters (random/zero depth degrades performance)
3. Is simple enough to implement, train, and explain convincingly
4. Connects naturally to P1 (uses its depth) and P3 (evaluated on its dataset)

---

## 2. Technical Gap

Existing depth-VLM integration methods add depth as passive features but leave the VLM's adaptation mechanism (LoRA weights) fixed regardless of scene geometry. The same LoRA weights process a flat parking lot and a complex urban intersection identically.

| Method | How depth is used | VLM adaptation | Limitation |
|--------|-------------------|----------------|------------|
| No depth (standard LoRA) | Not used | Fixed LoRA weights | Misses geometric cues |
| Depth as PE (SD-VLM) | Additive encoding | Unchanged | Signal too weak |
| Depth tokens (Spa3R-style) | Prepended tokens | Unchanged weights | Token budget cost, passive |
| Depth as second image | Multi-image input | Unchanged weights | ViT not trained on depth |
| Depth as text prompt | Numeric description | Unchanged weights | Loses spatial structure |

**Gap.** To our knowledge, no prior work conditions the PEFT mechanism itself on scene depth geometry for spatial reasoning tasks.

**Insight.** Depth should condition not just WHAT the VLM sees, but HOW it adapts. Different scene geometries should activate different adaptation behaviors.

---

## 3. Method Thesis

Conditioning LoRA adaptation on scene depth geometry significantly improves VLM spatial reasoning. Simple conditioning may already suffice; richer conditioning is justified only if empirically necessary. The study design --- progressive levels of conditioning richness --- is as important as any single method.

**Why this is the smallest adequate intervention.** We add depth conditioning to existing LoRA fine-tuning. No new architectures, no extra tokens, no modified attention. The base model and standard PEFT infrastructure remain unchanged.

**Why this route is timely.** Input-conditioned parameter adaptation is emerging in 2026 generative models (Video2LoRA, SHINE, FLOWER AdaLN-Zero). Applying it to VLM spatial understanding is natural and under-explored.

---

## 4. Contribution

- **Dominant.** Systematic study of depth-conditioned PEFT for VLM spatial reasoning, demonstrating that conditioning the adaptation mechanism on depth geometry is more effective than treating depth as passive input.
- **Supporting.** Progressive conditioning framework --- gating (simplest) through FiLM (medium) to basis mixing (richest) --- revealing how much depth conditioning is actually needed.
- **Explicit non-contributions.** New depth estimator, new dataset, new VLM architecture.

**Paper logic.** If DepthGate wins, the story is "simple conditioning suffices." If DepthBasis wins, the story is "scene-specific strategies matter." If none beats standard LoRA, the negative result is still publishable in the thesis.

---

## 5. Method Specification

### 5.1 Notation

| Symbol | Definition |
|--------|-----------|
| d | LLM hidden dim = 3584 |
| d_kv | KV projection dim = 512 (4 KV-heads x 128 dim) |
| L | Target layers = {20, 21, ..., 27} (upper 8 of 28) |
| P | Target projections = {q_proj, v_proj} |
| r | LoRA rank (level-dependent) |
| D | Depth map from P1, in R^{1 x H x W} |
| z_geo | Scene geometry descriptor, in R^{d_geo=256} |
| K | Number of basis LoRA pairs (Level 3 only) = 6 |

### 5.2 Shared Component: DepthGeometryNet (~445K params)

Extracts a global scene geometry descriptor from the depth map.

```
z_geo = f_depth(D) in R^256

f_depth(D):
  G = [D; Sobel_x(D); Sobel_y(D)]                    in R^{3 x H x W}
  h1 = GELU(Conv2d(3 -> 64, k=7, s=4, p=3)(G))       in R^{64 x H/4 x W/4}
  h2 = GELU(Conv2d(64 -> 128, k=3, s=2, p=1)(h1))    in R^{128 x H/8 x W/8}
  h3 = GELU(Conv2d(128 -> 256, k=3, s=2, p=1)(h2))   in R^{256 x H/16 x W/16}
  h4 = AdaptiveAvgPool2d(1,1)(h3) -> flatten           in R^{256}
  z_geo = GELU(Linear(256, 256)(LayerNorm(h4)))        in R^{256}
```

z_geo is intentionally a single global descriptor per image. Scene-level conditioning (e.g., "complex intersection" vs. "open highway") is the target. Object-level geometry is already captured in the visual tokens via the ViT.

### 5.3 Baseline: Standard LoRA (no depth) --- ~9.19M params

For each layer l in L, projection p in {q, v}:

```
A_l^p in R^{r x d_in},  B_l^p in R^{d_out x r}       (Kaiming/zero init)
Delta_l^p = (1/r) * B_l^p * A_l^p * h
output = W_frozen^p * h + Delta_l^p
```

Rank r = 102 to match ~9.19M total trainable parameters.

### 5.4 Level 1: DepthGate --- Scalar Depth Gating (~9.64M params)

Standard LoRA (r=102, ~9.19M) + DepthGeometryNet (445K) + scalar gating (4K).

The simplest depth conditioning. Standard fixed LoRA weights, with their output modulated by a per-layer, per-projection gate scalar derived from z_geo.

```
g_l^p = sigma(w_l^p . z_geo + b_l^p)     in R^1     (w in R^{256}, b in R^1)
output = W_frozen^p * h + g_l^p * (1/r) * B_l^p * A_l^p * h
```

Gate params: 8 layers x 2 projections x (256 + 1) = **4,112**

**What it tests.** Does scene-specific on/off scaling of LoRA output suffice? This is IA3-like gating conditioned on depth geometry.

### 5.5 Level 2: DepthFiLM --- Rank-Space FiLM (~9.77M params)

Standard LoRA (r=102, ~9.19M) + DepthGeometryNet (445K) + rank-space FiLM (132K).

Richer conditioning: depth generates per-layer scale and shift vectors that modulate the LoRA computation in its rank space (r-dimensional), not in the full d_out space.

```
[gamma_l^p, beta_l^p] = Linear(z_geo, 2*r)     gamma, beta in R^r
z_l^p = A_l^p * h                               in R^{B x seq x r}
z_conditioned = gamma_l^p . z_l^p + beta_l^p    (FiLM in rank space)
Delta_l^p = (1/r) * B_l^p * z_conditioned       in R^{B x seq x d_out}
output = W_frozen^p * h + Delta_l^p
```

FiLM params: 8 x 2 x (256 x 2*102 + 2*102) = **~132K**

**Why rank-space.** FiLM in the full d_out space costs ~16.8M parameters (budget-busting). Operating in the r-dimensional rank space is far more efficient and modulates the latent representation that LoRA learns.

**What it tests.** Does per-dimension modulation in the adaptation's latent space improve over scalar gating?

### 5.6 Level 3: DepthBasis --- Basis Mixing (~9.17M params)

No standard LoRA. K=6 LoRA basis pairs per (layer, projection), mixed by depth-derived routing weights.

The richest conditioning. Each basis pair represents a learned spatial processing strategy; the depth descriptor selects and blends strategies per scene.

```
{(A_l^{p,k}, B_l^{p,k})}_{k=1}^K     per layer l, projection p     (K=6, r=16)

alpha_l = softmax(MLP(z_geo + LayerEmbed(l)))     in R^K

For each basis k:
  delta_k = B_l^{p,k} * A_l^{p,k} * h            in R^{B x seq x d_out}

Delta_l^p = sum_k alpha_l^k * delta_k             (adapter-output mixing)
output = W_frozen^p * h + gate_l^p * Delta_l^p    (zero-init gate)
```

**Critical: adapter-output mixing, not weight mixing.** The mixing coefficients combine the outputs of each basis pair, not the weight matrices. This avoids the rank-collapse issue of weight-space mixing (sum of rank-16 matrices remains rank-16) and allows the effective adapter to represent richer transformations.

Basis params: 8 x 2 x 6 x (16x3584 + d_out x 16) = **~8.65M**
Router: ~70K | DepthGeometryNet: 445K | Gates: <1K
Total: **~9.17M**

**What it tests.** Do multiple learned spatial processing strategies, mixed per-scene by depth geometry, improve over single-LoRA conditioning?

### 5.7 Parameter Budget Summary

| Method | LoRA params | Depth params | Conditioning params | Total |
|--------|------------|-------------|-------------------|-------|
| Standard LoRA (r=102) | 9.19M | --- | --- | 9.19M |
| DepthGate (r=102) | 9.19M | 445K | 4K (gates) | 9.64M |
| DepthFiLM (r=102) | 9.19M | 445K | 132K (rank FiLM) | 9.77M |
| DepthBasis (r=16, K=6) | 8.65M | 445K | 70K (router) | 9.17M |

All methods are within ~9--10M total trainable parameters. DepthGate and DepthFiLM have slightly MORE total parameters than DepthBasis. Any DepthBasis advantage cannot be attributed to parameter count.

### 5.8 Insertion Point

Target: q_proj and v_proj of self_attn in Qwen2.5-VL LLM layers 20--27.

- Levels 1--2: Standard PEFT wrapping --- replace each target Linear with a LoRA-augmented version. Depth conditioning is applied to the LoRA output.
- Level 3: Replace each target Linear with DynamicLoRALinear (hook-based injection, existing GeoLoRA implementation).

Frozen: ViT, Merger, LLM base weights, layers 0--19.

---

## 6. Training

Identical for all levels:

| Setting | Value |
|---------|-------|
| Data | P3 SpatialQA train split (NuScenes images + P1 depth maps) |
| Loss | Standard next-token prediction on QA answers |
| Optimizer | AdamW, weight_decay=0.01 |
| LR (Levels 1--2) | depth_net: 1e-4, LoRA: 5e-5, gating/FiLM: 1e-3 |
| LR (Level 3) | depth_net+router: 1e-4, bank: 5e-5, gates: 1e-3 |
| Schedule | Cosine decay, 5% warmup |
| Epochs | 3 |
| Precision | bf16 |
| Effective batch size | 16 (batch=2 x grad_accum=8) |

---

## 7. Validation Plan

### 7.1 Category-Stratified Analysis

SpatialQA has 24 question types. Pre-defined split:

**Depth-sensitive types (10):** distance estimation, relative depth, height estimation, occlusion reasoning, size estimation at depth, spatial relation (depth-dependent), object-behind reasoning, near-far judgment, depth ordering, 3D position estimation.

**Non-depth-sensitive types (14):** color identification, object counting, direction naming, lane identification, traffic sign reading, weather/lighting, vehicle type, road marking, signal state, pedestrian action, general scene description, object presence, lateral position, heading direction.

**Hypothesis.** Depth-conditioned PEFT helps more on depth-sensitive types. If Level 3 (basis mixing) is needed anywhere, it is on depth-sensitive types.

### 7.2 Experiment Table

| ID | Method | Purpose |
|----|--------|---------|
| E1 | Zero-shot Qwen2.5-VL-7B | Floor baseline |
| E2 | Standard LoRA (r=102, ~9.19M) | Non-depth PEFT baseline |
| E3 | DepthGate (r=102 + scalar gates) | Simplest depth conditioning |
| E4 | DepthFiLM (r=102 + rank-space FiLM) | Medium depth conditioning |
| E5 | DepthBasis (K=6, r=16, adapter-output mixing) | Richest depth conditioning |
| E6 | Best-depth-method (shuffled depth) | Causality: is real depth needed? |
| E7 | Best-depth-method (zero depth) | Causality: is any depth needed? |

**Metrics.** Overall accuracy (exact match), per-type accuracy (24 types), depth-sensitive vs. non-depth-sensitive aggregate accuracy.

E6 and E7 use whichever depth-conditioned level performs best in E3--E5. This avoids wasting compute on causality ablations of suboptimal methods.

### 7.3 Pre-Committed Outcome Logic

| Outcome | Paper Narrative |
|---------|----------------|
| E3 > E2 clearly, E5 approx E3 | "Simple depth gating suffices. Geometric conditioning of PEFT is the key insight, not complex routing." |
| E5 > E3 clearly | "Scene-specific adaptation strategies are needed. Richer conditioning captures geometric nuances that simple gating misses." |
| E5 > E3 only on depth-sensitive types | "Richer conditioning is needed specifically for depth-dependent reasoning. Selective depth PEFT strategy is optimal." |
| E3 approx E2 | "Depth conditioning of PEFT does not help for this task. Negative result." (Still thesis-publishable.) |

---

## 8. Failure Modes and Diagnostics

| Failure | Symptom | Diagnostic |
|---------|---------|-----------|
| Depth conditioning inactive | DepthGate gates all near 0.5 | Check depth_net gradient norms; increase depth_net LR |
| Routing collapse | DepthBasis alpha uniform for all scenes | Add entropy regularization; visualize alpha across scene types |
| Depth hurts | All depth methods < Standard LoRA | Check P1 depth map quality; try GT depth |
| Training instability | Loss spikes or NaN | Reduce bank LR; clip gradients; increase gate warmup |

---

## 9. Compute Estimate

| Run | GPU Hours |
|-----|-----------|
| E1: Zero-shot eval | ~2h |
| E2: Standard LoRA train + eval | ~8h |
| E3: DepthGate train + eval | ~9h |
| E4: DepthFiLM train + eval | ~9h |
| E5: DepthBasis train + eval | ~10h |
| E6--E7: Causality ablation evals | ~2h |
| **Total** | **~40h on RTX 5880** |

Timeline: ~1 week of continuous server time.

---

## 10. Novelty Positioning

**Closest work.**
- SHINE: text-conditioned LoRA for in-context learning
- Video2LoRA: visual-reference-conditioned LoRA for video generation

**Exact differentiation.**
1. **Conditioning signal**: Geometric structure from depth maps (not text, not visual reference)
2. **Task domain**: Spatial understanding in VLMs (not generation, not ICL)
3. **Study design**: First systematic comparison of progressive depth conditioning levels on PEFT

**Positioning.** Sell on study design and empirical insight, not solely on mechanism novelty. The contribution is the finding (how much conditioning is needed), not just the framework.
