# Experiment Tracker

**Last updated**: 2026-03-17

## Implementation Tasks (Before Server Runs)

| ID | Task | Status | Notes |
|----|------|--------|-------|
| I1 | Implement DepthGateModel in baselines.py | DONE | Qwen2VLWithDepthGate + DepthGatedLoRALinear |
| I2 | Implement DepthFiLMModel in baselines.py | DONE | Qwen2VLWithDepthFiLM + DepthFiLMLoRALinear |
| I3 | Add depth_gate/depth_film to train script | DONE | build_model() + _save_checkpoint() |
| I4 | Add depth_gate/depth_film to eval script | DONE | load_model() + load_checkpoint() + run_generation() |
| I5 | Add depth-sensitive type split to eval | TODO | 10 depth-sensitive + 14 non-depth types |
| I6 | Generate depth maps with DAv2 on server | TODO | `scripts/generate_depth_maps.py` on NuScenes CAM_FRONT |
| I7 | Verify existing baselines (E1, E2) on server | TODO | Results exist at `~/djd/qwen2.5vl_lora/output/` |

## Experiment Runs

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | GPU Hours | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-----------|-------|
| R001 | M0 | Sanity check | DepthGate, 100 samples | train mini | loss↓ | MUST | TODO | ~1h | Verify pipeline works |
| R002 | M1 | Verify floor baseline | Zero-shot Qwen2.5-VL-7B | val | acc | MUST | TODO | ~2h | May already exist on server |
| R003 | M1 | Verify PEFT baseline | Standard LoRA r=102 | val | acc | MUST | TODO | ~8h | Re-run if prior run used different rank |
| R004 | M2 | **Core: depth works?** | DepthGate r=102 | train→val | acc, per-type | MUST | TODO | ~9h | **STOP/GO gate** |
| R005 | M3 | Medium conditioning | DepthFiLM r=102 | train→val | acc, per-type | MUST | TODO | ~9h | After M2 confirms |
| R006 | M3 | Rich conditioning | DepthBasis K=6 r=16 | train→val | acc, per-type | MUST | TODO | ~10h | After M2 confirms |
| R007 | M4 | Causality: shuffled | Best method + shuffled depth | val | acc drop | MUST | TODO | ~1h | Eval only, no training |
| R008 | M4 | Causality: zero | Best method + zero depth | val | acc drop | MUST | TODO | ~1h | Eval only, no training |
| R009 | M5 | Gate visualization | DepthGate checkpoint | val | gate distribution | MUST | TODO | ~0.5h | Post-hoc analysis |
| R010 | M5 | Per-type breakdown | All methods | val | 24-type table | MUST | TODO | ~0.5h | From eval outputs |
| R011 | M5 | Error analysis | Best vs Standard LoRA | val | 20 examples | MUST | TODO | ~1h | Manual inspection |
| R012 | M6 | Sobel ablation | DepthGate no-Sobel | train→val | acc | NICE | TODO | ~9h | Appendix |
| R013 | M6 | Multi-seed (close comparisons) | 3 seeds for Gate/FiLM/Basis | val | variance | NICE | TODO | ~6h | Only if margins tight |
| R014 | M6 | Alpha visualization | DepthBasis checkpoint | val | alpha patterns | NICE | TODO | ~0.5h | If DepthBasis is trained |

## Decision Gates

| Gate | After Run | Question | GO | STOP / PIVOT |
|------|-----------|----------|----|----|
| G1 | R001 | Pipeline works? | Loss decreases, depth loads | Fix data pipeline |
| G2 | R003 | Baseline reproducible? | Matches prior numbers (±1pt) | Investigate config differences |
| **G3** | **R004** | **Depth conditioning helps?** | **DepthGate > Standard LoRA by ≥2pts** | **Diagnose: check depths, try GT, increase capacity** |
| G4 | R005+R006 | How much conditioning? | Clear ordering or equivalence | Paper narrative chosen based on outcome |
| G5 | R007+R008 | Depth signal causal? | Predicted > Shuffled > Zero | Investigate z_geo (texture vs geometry) |

## Result Summary (To Be Filled)

| Method | Overall Acc | Depth-Sensitive Acc | Non-Depth Acc | Params |
|--------|-----------|-------------------|--------------|--------|
| E1: Zero-shot | — | — | — | 0 |
| E2: Standard LoRA | — | — | — | 9.19M |
| E3: DepthGate | — | — | — | 9.64M |
| E4: DepthFiLM | — | — | — | 9.77M |
| E5: DepthBasis | — | — | — | 9.17M |
| E6: Best(shuffled) | — | — | — | same |
| E7: Best(zero) | — | — | — | same |
