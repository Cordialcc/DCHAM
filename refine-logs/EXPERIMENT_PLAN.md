# Experiment Plan

**Problem**: Qwen2.5-VL-7B lacks spatial reasoning in driving scenes; depth maps exist but unused for adaptation
**Method Thesis**: Depth-conditioned PEFT improves VLM spatial reasoning; progressive study reveals how much conditioning is needed
**Date**: 2026-03-17
**Hardware**: 1× RTX 5880 48GB, conda env `nuscenes-spatialqa`
**Workdir**: `/home/xmu/djd/experiments`

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|----------------------------|---------------|
| C1: Depth-conditioned PEFT > Standard PEFT | Core contribution | DepthGate > Standard LoRA on overall accuracy AND on depth-sensitive subset, statistically significant | B1, B2 |
| C2: Depth signal is causally important | Rules out "any extra signal helps" | Predicted depth > Shuffled depth > Zero depth, monotonic degradation | B3 |
| C3: Conditioning richness matters (or doesn't) | Distinguishes paper narratives | Clear ordering or clear equivalence among Gate/FiLM/Basis, especially on depth-sensitive types | B4 |

Anti-claims to rule out:
- "The gain comes from more parameters" → DepthGate/FiLM have MORE params than DepthBasis; Standard LoRA is param-matched
- "The gain comes from the DepthGeometryNet architecture, not depth conditioning" → shuffled/zero depth ablations use same architecture
- "Any additional signal helps, not specifically depth" → shuffled depth ablation

---

## Paper Storyline

**Main paper must prove:**
1. Depth-conditioned LoRA > Standard LoRA for spatial reasoning (Table 1: main results)
2. Depth signal is causally necessary (Table 2: ablation)
3. How much conditioning is needed (Table 1 comparison across levels)
4. Results stratified by depth-sensitive vs non-depth-sensitive types (Table 3 or Figure)

**Appendix can support:**
- Sobel vs raw depth ablation
- Per-type breakdown (24 types detailed)
- Gate value / routing alpha visualizations
- Training loss curves

**Experiments intentionally cut:**
- External benchmark (CV-Bench) — nice-to-have, not essential for thesis
- Second model scale (Qwen2.5-VL-3B) — nice-to-have
- Question-conditioned routing — not needed unless DepthBasis clearly wins

---

## Experiment Blocks

### Block 1: Main Anchor Result — Does depth-conditioned PEFT work?

- **Claim tested**: C1 — Depth-conditioned PEFT > Standard PEFT
- **Why this block exists**: Core paper claim. Without this, nothing else matters.
- **Dataset / split / task**: SpatialQA val split (P3), all 24 question types
- **Compared systems**:
  - E1: Zero-shot Qwen2.5-VL-7B (floor baseline, already run on server)
  - E2: Standard LoRA r=102 (non-depth PEFT, already run on server — verify or re-run)
  - E3: DepthGate r=102 + scalar gates (simplest depth conditioning)
- **Metrics**: Overall exact-match accuracy, keyword-match accuracy
- **Setup details**:
  - Base: Qwen2.5-VL-7B-Instruct from `~/djd/qwen2.5vl_lora/Qwen2.5-VL-7B-Instruct/`
  - Data: `~/djd/qwen2.5vl_lora/spatialqa_gtads/` (train/val splits)
  - Depth: P1 depth maps paired with NuScenes images
  - Training: AdamW, cosine 5% warmup, 3 epochs, bf16, batch=2, grad_accum=8
  - Seed: 42 (run 3 seeds only if margins are tight)
- **Success criterion**: DepthGate accuracy > Standard LoRA accuracy by >= 2 points overall
- **Failure interpretation**: If DepthGate ≈ Standard LoRA → depth conditioning of PEFT doesn't help → paper becomes negative result (C1 falsified)
- **Table / figure target**: Table 1 (main results), rows 1-3
- **Priority**: MUST-RUN

### Block 2: Progressive Conditioning — How much is needed?

- **Claim tested**: C3 — Conditioning richness ordering
- **Why this block exists**: Distinguishes "simple suffices" from "complex needed" narrative
- **Dataset / split / task**: SpatialQA val split, stratified by depth-sensitive (10 types) vs non-depth-sensitive (14 types)
- **Compared systems**:
  - E3: DepthGate (from Block 1)
  - E4: DepthFiLM r=102 + rank-space FiLM
  - E5: DepthBasis K=6, r=16, adapter-output mixing
- **Metrics**: Overall accuracy + depth-sensitive subset accuracy + non-depth-sensitive accuracy
- **Setup details**: Same training setup as Block 1; matched param budgets (~9-10M each)
- **Success criterion**: Either clear ordering (Gate < FiLM < Basis) OR clear equivalence (all within 1 point)
- **Failure interpretation**:
  - If Gate ≈ FiLM ≈ Basis → simple gating suffices, paper narrative A
  - If Basis >> Gate on depth-sensitive only → selective conditioning, narrative D
  - If FiLM is the sweet spot → paper recommends FiLM as the practical choice
- **Table / figure target**: Table 1 (rows 4-5), Table 3 (stratified results)
- **Priority**: MUST-RUN (but AFTER Block 1 confirms depth conditioning works)

### Block 3: Depth Causality — Is the depth signal actually important?

- **Claim tested**: C2 — Depth signal causality
- **Why this block exists**: Rules out "any extra signal helps" and "architecture is the gain, not depth"
- **Dataset / split / task**: SpatialQA val split (same as above)
- **Compared systems** (all use the BEST depth-conditioned method from Block 1-2):
  - E_best(predicted depth) — real depth from P1
  - E6: E_best(shuffled depth) — randomly permute depth maps across images
  - E7: E_best(zero depth) — all-zero depth tensor
- **Metrics**: Accuracy degradation relative to predicted depth
- **Setup details**: No re-training. Use the same checkpoint; only change depth input at eval time.
- **Success criterion**: Predicted > Shuffled > Zero, with at least 2-point drop from predicted to zero
- **Failure interpretation**: If shuffled ≈ predicted → DepthGeometryNet is using texture, not geometry → investigate z_geo quality
- **Table / figure target**: Table 2 (depth ablation)
- **Priority**: MUST-RUN

### Block 4: Qualitative Analysis — What does depth conditioning learn?

- **Claim tested**: Supporting evidence for C1/C3
- **Why this block exists**: Adds interpretability and reviewer appeal
- **Analysis**:
  1. **Gate values** (DepthGate): Distribution of g_l^p across val set. Do different scenes get different gate activations?
  2. **Alpha coefficients** (DepthBasis, if run): Routing patterns across scene types (e.g., highway vs intersection vs parking)
  3. **Per-type breakdown**: Full 24-type accuracy table for best method vs Standard LoRA
  4. **Error analysis**: 10-20 examples where depth-conditioned method succeeds but Standard LoRA fails (and vice versa)
- **Setup details**: Post-hoc analysis on evaluation outputs. No additional training.
- **Table / figure target**: Figure 1 (gate/alpha visualization), Table 3 (per-type), Figure 2 (qualitative examples)
- **Priority**: MUST-RUN (but runs after Blocks 1-3, using saved checkpoints)

### Block 5: Sobel Ablation — Is gradient augmentation needed? (Appendix)

- **Claim tested**: Supporting — DepthGeometryNet design choice
- **Why this block exists**: Reviewer may question why Sobel. Quick to run.
- **Compared systems**:
  - DepthGate (with Sobel): default
  - DepthGate (without Sobel): raw depth only, Conv2d(1→64,...) instead of Conv2d(3→64,...)
- **Metrics**: Accuracy delta
- **Success criterion**: Sobel helps by >= 0.5 points, or can be dropped if not
- **Table / figure target**: Appendix Table A1
- **Priority**: NICE-TO-HAVE

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0: Sanity | Verify data pipeline, metrics, depth loading | Overfit on 100 samples with DepthGate | Loss decreases; depth maps load correctly | ~1h | Low: existing code mostly works |
| M1: Baseline | Confirm Standard LoRA baseline | E1 (zero-shot, re-verify), E2 (Standard LoRA, re-verify or re-run) | Accuracy matches prior numbers | ~2h (verify) or ~10h (re-run) | Low: already run before |
| M2: Core | Does depth conditioning work? | E3 (DepthGate train + eval) | **STOP/GO**: If E3 > E2 by >= 2pts → GO to M3. If E3 ≈ E2 → DIAGNOSE (check depth maps, try GT depth, check gradients) | ~9h | Medium: novel mechanism, may not help |
| M3: Progressive | How much conditioning? | E4 (DepthFiLM), E5 (DepthBasis) train + eval | Compare E3 vs E4 vs E5. Determines paper narrative. | ~19h | Low: mechanism is clear, just needs training |
| M4: Causality | Is depth signal causal? | E6 (shuffled depth), E7 (zero depth) — eval only, no training | Predicted > Shuffled > Zero | ~2h | Low: eval-only |
| M5: Analysis | Qualitative + per-type | Gate vis, alpha vis, per-type table, error examples | Results are interpretable | ~2h | Low: post-hoc analysis |
| M6: Polish | Appendix + optional | Sobel ablation, optional CV-Bench, optional multi-seed | Robustness | ~4-8h | Low |

**Total estimated: ~35-45 GPU hours**
**Critical path: M0 → M1 → M2 (STOP/GO) → M3 → M4 → M5**

---

## Compute and Data Budget

| Resource | Estimate |
|----------|----------|
| Total GPU hours | 35-45h |
| Data preparation | Minimal — SpatialQA already split, depths already generated |
| Depth map generation | Not needed — P1 depths already exist for NuScenes |
| Human evaluation | Not needed for thesis (exact-match + keyword-match suffice) |
| Biggest bottleneck | M2 decision gate — if DepthGate doesn't work, need to diagnose |
| Storage | ~2-3 GB per checkpoint (5 methods × 3 GB = ~15 GB) |
| Server access | `ssh -p 8003 xmu@120.24.31.117`, env `nuscenes-spatialqa` |

### Data Paths on Server

| Item | Path |
|------|------|
| Qwen2.5-VL-7B | `~/djd/qwen2.5vl_lora/Qwen2.5-VL-7B-Instruct/` |
| SpatialQA dataset | `~/djd/qwen2.5vl_lora/spatialqa_gtads/` |
| NuScenes images | `~/djd/datasets/nuscenes/samples/` |
| Depth maps (DAv2) | Generate with `scripts/generate_depth_maps.py` using DepthAnythingV2 |
| Existing LoRA output | `~/djd/qwen2.5vl_lora/output/` |
| Experiment workdir | `~/djd/experiments/` |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DepthGate ≈ Standard LoRA | Medium | High (paper weakened) | 1) Check depth map quality with GT depth. 2) Increase depth_net capacity. 3) Paper becomes negative-result study. |
| All depth methods equivalent | Medium | Medium (still publishable) | Paper narrative: "simple conditioning suffices" — elegant result |
| Training OOM | Low | Medium | bf16 + batch=2 + grad_accum=8 fits in 48GB. Monitor with `nvidia-smi`. |
| Server freeze during training | Low | High | Use `nohup` + lazy loading. Follow CLAUDE.md dangerous operations rules. |
| Depth maps not yet generated | Medium | Medium | Run `scripts/generate_depth_maps.py` with DepthAnythingV2. NuScenes has LiDAR but sparse; DAv2 gives dense depth. Swap to P1 for final thesis. |
| Standard LoRA baseline not reproducible | Low | Low | Re-run E2 with exact config from existing code |

---

## Implementation Notes

### What Code Exists vs What's Needed

| Component | Status | Location | Action |
|-----------|--------|----------|--------|
| DepthGeometryNet | EXISTS | `geolora/depth_geometry.py` | Reuse as-is |
| LoRABasisBank (Level 3) | EXISTS | `geolora/lora_bank.py` | Reuse as-is |
| GeometryRouter (Level 3) | EXISTS | `geolora/router.py` | Reuse as-is |
| DynamicLoRALinear (Level 3) | EXISTS | `geolora/injection.py` | Reuse as-is |
| Qwen2VLWithGeoLoRA (Level 3) | EXISTS | `geolora/model.py` | Reuse as-is |
| SpatialQADataset | EXISTS | `geolora/dataset.py` | Reuse, verify paths |
| StaticLoRA baseline | EXISTS | `geolora/baselines.py` | Reuse as-is |
| **DepthGate (Level 1)** | **NEEDS IMPL** | — | New: ~50 lines (LoRA + scalar gate from z_geo) |
| **DepthFiLM (Level 2)** | **NEEDS IMPL** | — | New: ~80 lines (LoRA + rank-space FiLM) |
| Training script | EXISTS | `scripts/train_geolora.py` | Add DepthGate/FiLM methods |
| Eval script | EXISTS | `scripts/evaluate_geolora.py` | Add DepthGate/FiLM methods |
| Depth-sensitive category split | **NEEDS IMPL** | — | Add to eval script config |

### Key Implementation Tasks Before Server Run

1. **Implement DepthGateModel** in `geolora/baselines.py` (~50 lines)
   - Standard LoRA r=102 on q_proj/v_proj layers 20-27
   - DepthGeometryNet → z_geo → per-(layer,proj) sigmoid gate
   - `forward()`: frozen(h) + g * (1/r) * B @ A @ h

2. **Implement DepthFiLMModel** in `geolora/baselines.py` (~80 lines)
   - Standard LoRA r=102 on q_proj/v_proj layers 20-27
   - DepthGeometryNet → z_geo → per-(layer,proj) γ,β ∈ R^r
   - `forward()`: z = A @ h, z_cond = γ⊙z + β, δ = B @ z_cond, output = frozen(h) + δ

3. **Add DepthGate/FiLM to training script** — extend `build_model()` dispatch
4. **Add DepthGate/FiLM to eval script** — extend `load_model()` + `load_checkpoint()` + generation
5. **Add depth-sensitive type split** to eval output
6. **Verify data paths on server** — confirm P1 depths exist, SpatialQA format

---

## Final Checklist

- [ ] Main paper table covered (E1-E5: zero-shot, LoRA, Gate, FiLM, Basis)
- [ ] Novelty isolated (E3 vs E2: depth conditioning vs no depth)
- [ ] Simplicity defended (E3 vs E4 vs E5: progressive levels)
- [ ] Depth causality proven (E6-E7: shuffled/zero depth)
- [ ] Category-stratified results (depth-sensitive vs non-depth-sensitive)
- [ ] Qualitative analysis (gate vis, error examples)
- [ ] Nice-to-have separated from must-run (Sobel ablation, CV-Bench)
- [ ] Code for DepthGate and DepthFiLM implemented
- [ ] Data paths verified on server
- [ ] P1 depth maps confirmed available
