# Review Summary

**Proposal: DepthPEFT**
**Rounds: 4 | Final Score: 8.6/10 | Verdict: REVISE (ceiling without results)**

---

## Score Trajectory

| Round | Score | Delta | Verdict |
|-------|-------|-------|---------|
| 1 | 6.2 | --- | REVISE |
| 2 | 7.1 | +0.9 | REVISE |
| 3 | 8.1 | +1.0 | REVISE |
| 4 | 8.6 | +0.5 | REVISE |

---

## Round 1 (Score: 6.2)

**Reviewer verdict:** "GeoLoRA complexity not justified. Simplify. Add simpler depth-conditioned baseline."

**Key feedback:**
1. GeoLoRA's basis mixing complexity is not justified --- must prove it is needed, not just helpful.
2. Option C (gating) should be the main method; keep GeoLoRA only if it clearly wins.
3. Missing the key control: a simpler depth-conditioned PEFT baseline (IA3/FiLM-style).
4. Contribution is unclear --- two claims fighting ("depth helps" vs. "basis mixing helps").
5. Narrow the claim to "depth-conditioned PEFT for spatial reasoning."

**Honest assessment (verbatim):** "GeoLoRA's complexity is not justified yet. My honest bet is that a simpler depth-conditioned method will recover most of the gain over static LoRA. Use GeoLoRA only if it shows a clear, stable margin over the simpler conditioned baseline on the depth-sensitive parts of SpatialQA."

**Dimension breakdown:**

| Dimension | Score | Note |
|-----------|-------|------|
| Problem Fidelity | 7 | Right problem. Risk: redefining bottleneck as "attention must be dynamic" |
| Method Specificity | 6 | High-level clear but presents a menu. Ablation nesting claim mathematically false |
| Contribution Quality | 5 | Two claims fighting each other |
| Frontier Leverage | 7 | Qwen2.5-VL + PEFT + depth conditioning is appropriate |
| Feasibility | 8 | Very feasible. Risk is scientific overdesign |
| Validation Focus | 6 | Missing simpler depth-conditioned PEFT baseline |
| Venue Readiness | 5 | Novelty not sharp unless GeoLoRA clearly beats simpler conditioned baseline |

---

## Round 2 (Score: 7.1)

**Reviewer verdict:** "Framing is paper-shaped. Need exact equations and category-stratified analysis."

**Key feedback:**
1. Method Specificity still lacking: exact equations, rank values, gate granularity, matched parameter counts.
2. Validation needs category-stratified analysis: depth-sensitive vs. non-depth-sensitive QA types.
3. Soften "no prior work" novelty claim --- it is brittle. Sell on study design + empirical insight.

**What improved (+0.9):**
- Reframed from "GeoLoRA is the answer" to "systematic study of depth conditioning levels" (+2 Contribution Quality).
- Eliminated the method menu; committed to a single progressive framework.
- Added IA3/FiLM-style baseline as a serious conditioning level.
- Narrowed the gap statement.

**Dimension breakdown:**

| Dimension | Score | Delta | Note |
|-----------|-------|-------|------|
| Problem Fidelity | 8 | +1 | Much better aligned |
| Method Specificity | 6 | 0 | Still missing insertion details |
| Contribution Quality | 7 | +2 | Sharper --- focused study, not oversized mechanism |
| Frontier Leverage | 8 | +1 | Appropriate, no trend chasing |
| Feasibility | 8 | 0 | Very plausible |
| Validation Focus | 6 | 0 | Needs category-stratified analysis |
| Venue Readiness | 6 | +1 | Pseudo-novelty risk in "no prior work" claim |

---

## Round 3 (Score: 8.1)

**Reviewer verdict:** "Precise enough to implement. Fix DepthBasis mixing definition."

**Key feedback:**
1. DepthBasis mixing equation needs clarification: weight-space mixing (summing rank-16 matrices) collapses rank. Use adapter-output mixing instead.
2. Otherwise, the proposal is now at implementation-ready specificity.
3. Category-stratified analysis design is solid.

**What improved (+1.0):**
- Full mathematical specification: exact equations for all three levels with concrete dimensions.
- Matched parameter budgets: all methods ~9-10M, with Gate/FiLM having MORE params than Basis.
- Rank-space FiLM correction: discovered that full d_out FiLM costs ~16.8M (budget-busting), switched to rank-space modulation.
- Pre-defined depth-sensitive vs. non-depth-sensitive QA type split.

**Dimension breakdown:**

| Dimension | Score | Delta | Note |
|-----------|-------|-------|------|
| Problem Fidelity | 9 | +1 | Locked in |
| Method Specificity | 8 | +2 | Exact equations, implementable |
| Contribution Quality | 8 | +1 | Clear progressive study |
| Frontier Leverage | 8 | 0 | Appropriate |
| Feasibility | 8 | 0 | Very feasible |
| Validation Focus | 8 | +2 | Category split addresses stratified claim |
| Venue Readiness | 7 | +1 | Improved but still needs results |

---

## Round 4 (Score: 8.6)

**Reviewer verdict:** "Strong enough to execute. Contribution strength now depends on empirical results."

**Key feedback:**
1. Adapter-output mixing for DepthBasis is now correctly specified. Avoids rank collapse.
2. The proposal has reached its ceiling without results --- no further refinement will increase the score.
3. Remaining risk is entirely empirical: will depth conditioning actually help?

**What improved (+0.5):**
- Fixed DepthBasis mixing from weight-space to adapter-output mixing.
- Clarified that E6/E7 causality ablations apply to whichever depth method performs best.
- Pre-committed outcome table strengthened with the depth-sensitive-only outcome path.

**Dimension breakdown:**

| Dimension | Score | Delta | Note |
|-----------|-------|-------|------|
| Problem Fidelity | 9 | 0 | Locked |
| Method Specificity | 9 | +1 | Adapter-output mixing correctly defined |
| Contribution Quality | 8 | 0 | Depends on results now |
| Frontier Leverage | 8 | 0 | Stable |
| Feasibility | 9 | +1 | Implementation-ready |
| Validation Focus | 9 | +1 | Complete experiment design |
| Venue Readiness | 8 | +1 | Ceiling without results |

---

## Key Evolution

**Starting point (Round 0):** "GeoLoRA is a novel geometry-conditioned dynamic LoRA mechanism. It uses K=6 basis pairs mixed by a router. Options B and C are fallbacks."

**Ending point (Round 4):** "DepthPEFT is a systematic study of how much depth conditioning is needed for PEFT in VLM spatial reasoning. Three precisely specified levels, fairly budgeted, with pre-committed outcome logic for any empirical result."

The fundamental shift: from **"GeoLoRA is the answer"** to **"DepthPEFT asks the right question."**
