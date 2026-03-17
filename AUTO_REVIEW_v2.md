# DepthPEFT Auto Review Loop v2 — Method Design

**Focus**: Finding the right depth injection architecture
**Started**: 2026-03-17
**Max Rounds**: 4
**Reviewer**: GPT-5.4 (xhigh reasoning)

---

## Round 1 (2026-03-17)

### Assessment (Summary)
- Topic: What depth injection approach is both innovative and feasible?
- Key finding: **Stop trying to make PEFT itself dynamic. Operate BEFORE the LLM.**

### Approach Scores

| Approach | Innovation | Feasibility | Impact | Verdict |
|----------|-----------|------------|--------|---------|
| A. Cross-attention (depth→visual) | 7 | 7 | 8 | **Best among A-E** |
| B. Token reweighting | 5 | 9 | 6 | Weak main method |
| C. Fusion adapter (MLP) | 6 | 8 | 7 | Safe fallback |
| D. Structured depth prompt | 4 | 8 | 4 | Don't bother |
| E. Mid-layer injection | 8 | 3 | 7 | Too risky for 1 week |
| **F. Relative depth bias** | **8** | **6** | **8** | **Best overall if time allows** |

### Recommended Architecture

**A-lite: Depth-Aware Visual Token Adapter (post-merger, pre-LLM)**

```
Image → Qwen ViT+Merger (frozen) → visual_tokens (B, N, d)
                                         ↓
Depth map → DepthEncoder → depth_features (B, Nd, d) + pos_embed
                                         ↓
                              Cross-Attention adapter
                              Q=ln(visual), K=ln(depth), V=depth
                                         ↓
                          visual_enriched = visual + gate * proj(attn_out)
                                         ↓
                    Replace image tokens in inputs_embeds with enriched
                                         ↓
                         LLM + Standard LoRA (gradient checkpointing OK)
```

### Key Insight
All depth conditioning happens **before** the checkpointed LLM — no hooks, no PEFT modification, no wrapper needed.

### Reviewer Raw Response

<details>
<summary>Click to expand</summary>

Stop trying to make PEFT itself dynamic. The hook-heavy paths are where gradient checkpointing fights you.

Best approach: A (post-merger cross-attention), with F (relative depth bias) as upgrade.

Biggest risk: plumbing error in image-token replacement. Mitigation: zero-init gate + equivalence test with gate=0.

</details>

### Status
- Proceeding with implementation of A-lite
