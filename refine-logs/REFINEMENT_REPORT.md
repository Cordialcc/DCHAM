# Refinement Report

**Proposal: DepthPEFT (formerly GeoLoRA)**
**Refinement Thread: 019cf9b0-098f-7061-af24-fd9d8c30bba9**
**Date: 2026-03-17**
**Rounds: 4 | Final Score: 8.6/10 | Verdict: REVISE (ceiling without results)**

---

## 1. Score Evolution

```
Round 1: 6.2  ████████████░░░░░░░░  (REVISE)
Round 2: 7.1  ██████████████░░░░░░  (REVISE)  +0.9
Round 3: 8.1  ████████████████░░░░  (REVISE)  +1.0
Round 4: 8.6  █████████████████░░░  (REVISE)  +0.5
```

| Round | PF | MS | CQ | FL | Feas | VF | VR | Overall | Verdict |
|-------|----|----|----|----|------|----|----|---------|---------|
| 1 | 7 | 6 | 5 | 7 | 8 | 6 | 5 | 6.2 | REVISE |
| 2 | 8 | 6 | 7 | 8 | 8 | 6 | 6 | 7.1 | REVISE |
| 3 | 9 | 8 | 8 | 8 | 8 | 8 | 7 | 8.1 | REVISE |
| 4 | 9 | 9 | 8 | 8 | 9 | 9 | 8 | 8.6 | REVISE |

PF = Problem Fidelity, MS = Method Specificity, CQ = Contribution Quality,
FL = Frontier Leverage, Feas = Feasibility, VF = Validation Focus, VR = Venue Readiness.

**Verdict rationale.** The proposal has reached its theoretical ceiling. Score 8.6 with "REVISE" reflects that the remaining 1.4 points are locked behind empirical results. No amount of further specification will raise the score. The next improvement comes from running experiments.

---

## 2. Round-by-Round Record

### Round 0 -> Round 1 (Initial Submission: 6.2)

**What was submitted.** GeoLoRA proposal with three method options (A: full basis mixing, B: hypernetwork LoRA, C: gated LoRA). Recommended Option A as the main method. Options B and C presented as fallbacks.

**Reviewer response.** "GeoLoRA complexity not justified. Simplify. Add simpler depth-conditioned baseline."

**Critical issues identified:**
- The proposal is a menu, not a committed design.
- No simpler depth-conditioned baseline exists to prove basis mixing is necessary.
- Contribution Quality scored 5/10 --- two claims fighting each other.
- The "ablation nesting" claim (A ablates to B ablates to C) is mathematically false.

### Round 1 -> Round 2 (Score: 7.1, +0.9)

**Changes made:**
1. Reframed from "GeoLoRA mechanism" to "DepthPEFT systematic study."
2. Eliminated the method menu. Three levels presented as progressive conditioning depth.
3. Added IA3/FiLM-style depth conditioning as a serious intermediate level.
4. Narrowed the gap statement from "all prior methods keep attention fixed" to "no prior work conditions PEFT on depth geometry."

**What moved:** Contribution Quality +2 (5->7), Problem Fidelity +1, Frontier Leverage +1, Venue Readiness +1.

**Reviewer response.** "Framing is paper-shaped. Need exact equations and category-stratified analysis."

### Round 2 -> Round 3 (Score: 8.1, +1.0)

**Changes made:**
1. Full mathematical specification for all three levels with concrete dimensions.
2. Discovered FiLM budget issue: full d_out FiLM costs ~16.8M params (budget-busting). Corrected to rank-space FiLM (~132K).
3. Matched parameter budgets across all methods (~9-10M). Gate/FiLM have MORE params than Basis.
4. Pre-defined depth-sensitive vs. non-depth-sensitive QA category split.
5. Softened "no prior work" claim to "to our knowledge."

**What moved:** Method Specificity +2 (6->8), Validation Focus +2 (6->8), Venue Readiness +1.

**Reviewer response.** "Precise enough to implement. Fix DepthBasis mixing definition."

**Key discovery:** The DepthBasis weight-mixing formulation (sum_k alpha_k * A_k, then multiply) keeps the result at rank-16 regardless of K. Adapter-output mixing (sum_k alpha_k * B_k * A_k * h) avoids this rank collapse.

### Round 3 -> Round 4 (Score: 8.6, +0.5)

**Changes made:**
1. Fixed DepthBasis from weight-space mixing to adapter-output mixing.
2. Clarified E6/E7 causality ablations apply to best-performing depth method.
3. Added "depth-sensitive only" outcome path to pre-committed logic.

**What moved:** Method Specificity +1 (8->9), Feasibility +1 (8->9), Validation Focus +1 (8->9), Venue Readiness +1 (7->8).

**Reviewer response.** "Strong enough to execute. Contribution strength now depends on empirical results."

---

## 3. Method Evolution Highlights

### 3.1 From GeoLoRA to DepthPEFT

The most significant evolution was not technical but conceptual:

| Aspect | Round 0 | Round 4 |
|--------|---------|---------|
| Framing | "GeoLoRA is a novel mechanism" | "DepthPEFT is a systematic study" |
| Core claim | "Basis mixing is needed" | "How much conditioning is needed?" |
| Contribution | One mechanism | Study design + empirical finding |
| Robustness | Fragile (GeoLoRA must win) | Robust (any outcome is publishable) |

### 3.2 DepthBasis Mixing Fix

The weight-mixing formulation was mathematically flawed:
```
WRONG:  A_mixed = sum_k alpha_k * A_k    (result is still rank-16)
        B_mixed = sum_k alpha_k * B_k
        delta = B_mixed * A_mixed * h

RIGHT:  delta = sum_k alpha_k * (B_k * A_k * h)    (adapter-output mixing)
```

Adapter-output mixing allows the effective transformation to exceed rank-16, which is the whole point of having K=6 bases.

### 3.3 Rank-Space FiLM Discovery

Original FiLM design modulated in d_out space (3584 or 512 dimensions), costing ~16.8M parameters --- nearly double the budget. Rank-space FiLM modulates in the r-dimensional latent space, costing only ~132K. This was a non-obvious but critical budget fix.

### 3.4 Parameter Budget Fairness

Final budget design ensures DepthGate (9.64M) and DepthFiLM (9.77M) have slightly MORE parameters than DepthBasis (9.17M). Any DepthBasis advantage is attributable to its richer conditioning mechanism, not parameter count.

---

## 4. Pushback Log

Reviewer pushback that changed the proposal:

| Round | Pushback | Action Taken | Impact |
|-------|----------|-------------|--------|
| 1 | "Method menu, not committed design" | Eliminated options, made progressive framework | +2 CQ |
| 1 | "No simpler conditioned baseline" | Added DepthGate and DepthFiLM as serious levels | +2 CQ |
| 1 | "Ablation nesting claim is mathematically false" | Dropped nesting claim, honest about independence | Credibility |
| 2 | "Missing exact equations" | Full spec with dimensions for all levels | +2 MS |
| 2 | "Category-stratified analysis needed" | Pre-defined depth-sensitive/non-depth split | +2 VF |
| 2 | "'No prior work' is brittle" | Softened to "to our knowledge" + sell on study design | +1 VR |
| 3 | "Weight-mixing collapses rank" | Switched to adapter-output mixing | +1 MS |

Reviewer pushback that was noted but not acted on (no action needed):

| Round | Pushback | Reason for non-action |
|-------|----------|-----------------------|
| 4 | "Score ceiling without results" | Acknowledged. Next step is experiments, not more refinement. |

---

## 5. Remaining Weaknesses

### 5.1 Empirical Risk (High)

The entire contribution depends on depth conditioning actually helping. If E3 (DepthGate) does not beat E2 (Standard LoRA), the paper becomes a negative result. This is acceptable for a thesis chapter but limits standalone publication potential.

**Mitigation:** Pre-committed outcome logic covers all cases. The negative result path is explicitly acknowledged.

### 5.2 DepthGeometryNet Quality (Medium)

The shared DepthGeometryNet must produce a z_geo that actually captures useful scene geometry. If the 3-layer CNN + global pool is too lossy, all three levels will underperform.

**Mitigation:** This is the highest-risk component and should be validated first. Check gradient flow through depth_net in early training. If z_geo is uninformative, try (a) deeper CNN, (b) skip AvgPool and use spatial features, (c) pre-trained depth encoder.

### 5.3 z_geo Granularity (Low-Medium)

z_geo is a single global vector per scene. If spatial reasoning requires object-level or region-level geometric conditioning, the global descriptor will be insufficient.

**Mitigation:** Explicitly scoped as a non-claim. The paper does not promise object-level geometry reasoning from z_geo.

### 5.4 Single VLM, Single Dataset (Low)

Results are on Qwen2.5-VL-7B + SpatialQA only. Generalization to other VLMs or benchmarks is not tested.

**Mitigation:** Acceptable scope for a thesis chapter. If pursuing standalone publication, add CV-Bench or SpatialBench as transfer evaluation.

---

## 6. Next Steps

### Immediate: Experiment Plan

Run `/experiment-plan` to generate a detailed execution roadmap covering:
- Server environment setup and dependency verification
- Data pipeline validation (depth maps + SpatialQA loading)
- DepthGeometryNet sanity check (gradient flow, z_geo diversity)
- Training scripts for all 5 methods (E1--E5)
- Evaluation pipeline with category-stratified metrics
- Causality ablation protocol (E6--E7)
- Checkpoint management and logging

### Execution Order

1. **E1 (Zero-shot):** Quick baseline, validates evaluation pipeline.
2. **E2 (Standard LoRA):** Validates training pipeline, establishes non-depth baseline.
3. **E3 (DepthGate):** Simplest depth method. If this already wins clearly, the story is strong.
4. **E4 (DepthFiLM):** Only if E3 shows promise (depth conditioning works).
5. **E5 (DepthBasis):** Only if E3/E4 show that richer conditioning helps.
6. **E6--E7 (Causality):** Applied to whichever depth method performs best.

### Decision Points

- After E2+E3: If E3 < E2, investigate DepthGeometryNet before running E4/E5.
- After E3+E4+E5: Determine paper narrative based on pre-committed outcome logic.
- After E6+E7: Confirm causality. If shuffled depth matches predicted depth, z_geo is not using real geometry.

### Code Status

The existing GeoLoRA codebase (`geolora/` package) implements Level 3 (DepthBasis). Levels 1 (DepthGate) and 2 (DepthFiLM) need new implementations, which are straightforward extensions of the standard LoRA wrapping pattern.

Files to create or modify:
- `geolora/depth_gate.py` --- DepthGate LoRA wrapper (new)
- `geolora/depth_film.py` --- DepthFiLM LoRA wrapper (new)
- `scripts/train_depthpeft.py` --- Unified training script for all levels (modify existing)
- `scripts/evaluate_depthpeft.py` --- Unified evaluation with category stratification (modify existing)

### Estimated Timeline

| Phase | Duration | GPU Hours |
|-------|----------|-----------|
| Environment setup + data validation | 1 day | 0 |
| E1 (zero-shot eval) | 0.5 day | 2h |
| E2 (Standard LoRA) | 1 day | 8h |
| E3 (DepthGate) | 1 day | 9h |
| E4 (DepthFiLM) | 1 day | 9h |
| E5 (DepthBasis) | 1.5 days | 10h |
| E6--E7 (causality) | 0.5 day | 2h |
| Analysis + writing | 2 days | 0 |
| **Total** | **~8 days** | **~40h** |
