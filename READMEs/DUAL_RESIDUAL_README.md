# Dual-Scale Coupling Model — Design

Two-component architecture for zero-shot pattern synthesis at 8×8 / 16×16 while training **only** on 2×2, 4×4, and 6×6 HFSS.

**Code:** `src/models/dual_residual.py`  
**Prep:** `scripts/prep_dual_residual_train.py`  
**Train:** `scripts/train_dual_residual.py`  
**Checkpoints:** `checkpoints/dual_residual/`

---

## Problem decomposition

Full-wave pattern at scale N:

```
hfss_N ≈ sub_block_N + δ_coupling + δ_scale(N)
```

| Term | Meaning | Who learns it |
|------|---------|---------------|
| `sub_block_N` | Phase-aware (or magnitude) compose from HFSS sub-blocks | **No ML** — physics + data |
| `δ_coupling` | Mutual-coupling correction (sidelobes/nulls; main beam mostly OK) | **Coupling head** (v4 regional multi-head) |
| `δ_scale(N)` | Extra compose error that grows with array size | **Scale head** (lightweight, scale-conditioned) |

At **6×6** (max training scale), `δ_scale → 0` by design — the coupling head owns the problem where sub-block data is richest. At **2×2 / 4×4**, the scale head absorbs weaker compose / fewer sub-block types.

This avoids the matlab_rel failure mode (`hfss − matlab` is huge and scale-dependent) while keeping multiscale extrapolation.

---

## Targets (why not a single residual?)

**Primary target (both heads):** compose deficit in dB

```
δ* = hfss − sub_block
```

Peak-normalised like the rest of the pipeline. Easier than `hfss − matlab` because sub_block already includes measured coupling.

**Coupling head target:** `δ*`, with tiered null/sidelobe weighting (v4 loss style).

**Scale head target:** residual after coupling (stop-gradient through coupling):

```
δ_scale_target = δ* − δ_coupling
```

**Regularisation at N=6:** `L_zero = mean|δ_scale|` when `scale_token ≈ 1`.

**Optional future target** (not in v1): null-region log-amplitude ratio — only if compose MAE plateaus.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
  x (6 ch)          │  Coupling head (FocusedRegional v4) │
  matlab            │  • base + null_refine + E/H cuts     │
  sub_block    ───► │  • main-beam shield                  │──► δ_coupling
  fingerprint       │  • NO scale token (scale-invariant)  │
  dpx, dpy          └──────────────┬──────────────────────┘
  coupling_gap                     │ bottleneck
                                   ▼
                    ┌─────────────────────────────────────┐
  scale_token       │  ScaleCorrectionHead                 │
  log(N)/log(6) ──► │  • MLP(scale) + bottleneck fuse      │──► δ_scale
  δ_coupling (ctx)  │  • zero-init output conv               │
                    └─────────────────────────────────────┘

Inference:
  pred = sub_block + shield(δ_coupling) + α_scale · δ_scale
```

### Input channels (coupling head — same as v4)

| ch | Name | Notes |
|----|------|-------|
| 0 | matlab | Analytical AF; z-scored per-scale train stats |
| 1 | sub_block | Composed from HFSS sub-blocks |
| 2 | fingerprint | matlab − mean(sub-blocks at native scale) |
| 3–4 | dpx, dpy | /180 broadcast |
| 5 | coupling_gap | sub_block − matlab |

Scale token is **not** fed to the coupling head so it learns coupling physics independent of N.

### Scale token

```
scale_token = log(N) / log(6)
```

| N | Token |
|---|-------|
| 2 | 0.386 |
| 4 | 0.774 |
| 6 | 1.000 |
| 8 | 1.124 (zero-shot extrapolation) |
| 16 | 1.286 |

Gentler than `N/16` used in the old multiscale cGAN; training only sees tokens ≤ 1.

---

## Training data (2 / 4 / 6 only)

Aligned by steering to the stage-1 6×6 anchor set (~1000 samples × 3 scales = ~3000 rows):

| Scale | HFSS source | sub_block source |
|-------|-------------|------------------|
| 2×2 | `antenna_data_2to4.h5` (or measured HFSS when available) | 4 native 2×2 sub-blocks composed |
| 4×4 | `antenna_data.h5` | `subblock_compositions_v2` (6×6 corners) |
| 6×6 | `stage1_6x6_extras.npz` | 9-block compose / truth-calibrated |

Split: same train/val/test indices as `split_indices_stage1_6x6`, replicated per scale.

**No 8×8 or 16×8 HFSS in the loss.**

---

## Loss

```
L = L_compose
  + λ_scale · L_scale (weighted higher when scale_token < 1)
  + λ_zero · L_zero at N=6
  + λ_null · L_null (aux, ramped)
```

`L_compose` = tiered L1 + SSIM on `sub_block + δ_coupling + δ_scale` vs HFSS.

Recommended init: `--init-v4` warm-starts the coupling head from `stage1_6x6_multihead_v4/best_generator.pt`.

---

## Zero-shot inference (8×8 / 16×16)

1. Build `sub_block_N` from phase-aware compose NPZ (`phase_aware_8x8_compose.npz`, etc.).
2. Build matlab, fingerprint, coupling_gap at target scale (cached via `prep_zeroshot_eval_cache`).
3. Forward with `scale_token = log(N)/log(6)`.
4. Reconstruct:

```
pred = sub_block + shield(δ_coupling) + α · δ_scale
```

Fit scalar `α` (or regional α like v5) on a small train split at the target scale — same protocol as existing evaluators.

---

## Why not spatial quadrant heads?

Mutual coupling crosses sub-block boundaries; the repo already factorises spatially via **9 sub-block types** (corner/edge/interior). v4’s regional heads split by **radiation region** (main/null/E/H cuts), which matches where compose fails without forcing angular seams.

---

## Pipeline

```bash
# 1. Ensure stage-1 6×6 extras exist
python -m scripts.prep_stage1_6x6

# 2. Build multi-scale training cache (2/4/6)
python -m scripts.prep_dual_residual_train

# 3. Train (warm-start coupling from v4)
python -m scripts.train_dual_residual --init-v4

# 4. Zero-shot eval (TODO: evaluate_dual_residual_zeroshot.py)
#    Uses phase_aware compose + zeroshot_eval_cache
```

---

## Comparison to prior work in this repo

| Model | Train scales | Target | Zero-shot 8×8 MAE |
|-------|-------------|--------|-------------------|
| multihead v4 | 6×6 only | hfss − sub_block | ~2.9 dB |
| matlab_rel v4 | 6×6 only | hfss − matlab | ~5.5 dB (failed) |
| multiscale cGAN | 4+8 | hfss − matlab | N/A @ 8×8 |
| **dual_residual** | **2+4+6** | **compose deficit + scale split** | TBD |

Expected benefit: v4-quality coupling at 6×6 **plus** explicit scale correction learned from 2×2/4×4/6×6 progression, improving extrapolation to 8/16 without HFSS labels at those sizes.
