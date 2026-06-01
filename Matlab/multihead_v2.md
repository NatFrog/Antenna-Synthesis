# Multi-Head v2 — Physics-Informed Regional Residual Model

A **6×6-trained, zero-shot** mutual-coupling corrector that sits on top of position-aware sub-block composition. Unlike the baseline cGAN (uniform residual + PatchGAN), multi-head v2 applies correction **only where physics says it is needed** — sidelobes and nulls — while preserving the main beam that sub-block compose already predicts well.

**Checkpoint:** `checkpoints/stage1_6x6_multihead_v2/best_generator.pt`  
**Model code:** `src/models/multihead_v2.py`  
**Related:** parent pipeline in [README.md](README.md); future ESA surrogate in [esa/README_ESA.md](esa/README_ESA.md)

---

## TL;DR — headline result @ 8×8

Held-out test **n = 101** (`predicted_8x8_full_bundle.npz`), peak-normalised dB, LS α = **0.937** fit on train residuals.

| Metric | Sub-block baseline | **Multi-head v2** | MATLAB (no coupling) |
|--------|-------------------:|------------------:|---------------------:|
| Full MAE (dB) | 4.75 | **3.06** | 7.71 |
| Full RMSE (dB) | 6.75 | **4.22** | 11.90 |
| R² | 0.59 | **0.84** | −0.29 |
| Pearson | 0.83 | **0.91** | 0.78 |
| Null RMSE (dB) | 7.03 | **4.38** | 12.42 |
| Null fill accuracy | 33% | **44%** | 26% |
| MAE ≥ −10 dB (main beam) | **0.40** | 0.44 | 0.50 |

The model beats sub-block compose on every global and null-region metric, with main-beam error within ~0.04 dB of the baseline. Full comparison table: `results/stage2_multihead_v2/comparison.txt`.

---

## Method in one paragraph

Sub-block composition (`sub_block_NxN`) tiles nine pre-measured 6×6 HFSS sub-block patterns (corner / edge / interior) into larger arrays using array-factor phase — see [README.md](README.md). That baseline captures most of the main-lobe structure but misses mutual-coupling detail in sidelobes and nulls (~4.75 dB MAE at 8×8). Multi-head v2 learns the **residual** `hfss − sub_block` at 6×6 using a gated ResUNet with auxiliary PSL and null-depth heads. A **coupling-gap channel** (`sub_block − matlab`) tells the network where local coupling diverges from the isolated-element model (ESA-inspired). A **main-beam shield** zeros correction inside the HPBW at train and inference time. At 8×8 zero-shot, the frozen 6×6 model predicts a gated residual; a scalar **α** is least-squares fit on 8×8 train residuals to calibrate amplitude. **No 8×8 HFSS is used during stage-1 training.**

---

## Pipeline

```
  6×6 HFSS sub-blocks (9 classes)
              │
              ▼
  ┌───────────────────────────────────────┐
  │  sub_block_NxN  (position-aware tile) │
  └───────────────────────────────────────┘
              │
              │  + gated residual
              ▼
  ┌───────────────────────────────────────┐
  │  RegionalGatedFiLMMultiHeadResUNet     │
  │  • 6 input channels (see below)        │
  │  • Regional gate + main-beam shield    │
  │  • PSL + top-K null-depth aux heads    │
  │  • No PatchGAN (supervised only)       │
  │  • ~20 M params, EMA weights           │
  └───────────────────────────────────────┘
              │
              ▼
  pred = sub_block + α · shield( gated_residual )
  pred_pn = pred − max(pred)                    (peak-normalised dB)
```

### Reconstruction formula

```
residual_gated = regional_scale · coupling_gate · residual_raw

regional_scale = α_null · shield + α_main · (1 − shield)
coupling_gate  = σ(spatial_gate + aux_bias) · amplitude_gate(sub_block)
shield         ≈ 0 inside HPBW, ≈ 1 in sidelobes/nulls

pred_8x8 = sub_block_8x8 + α_LS · shield(residual_gated)   # α fit on train
```

Learned scalars at convergence (typical): `α_main ≈ 0.04`, `α_null ≈ 0.93`, `α_LS ≈ 0.94`.

---

## Input channels (6 × 181 × 360)

| ch | Name | 6×6 training | 8×8 inference |
|----|------|--------------|---------------|
| 0 | `matlab` | Analytical 6×6 AF, z-scored (6×6 train stats) | matlab_8×8, z-scored (**8×8 train** per-pixel stats) |
| 1 | `sub_block` | 6×6 sub-block compose | sub_block_8×8, z-scored (**8×8 train** stats) |
| 2 | `fingerprint` | `matlab − mean(9 HFSS sub-blocks)` | Same formula; fingerprint stats from **6×6 train** |
| 3 | `dphase_x / 180` | Steering scalar broadcast | Same |
| 4 | `dphase_y / 180` | Steering scalar broadcast | Same |
| 5 | **`coupling_gap`** | `(sub_block − matlab)`, z-scored on 6×6 train | Same difference; gap stats from **8×8 train** |

**Target (training):** `residual = hfss_6×6 − sub_block_6×6`, z-scored with 6×6 `residual_mean / residual_std`.

---

## Architecture highlights

| Component | Purpose |
|-----------|---------|
| **Coupling-gap channel** | Maps where sub-block compose ≠ isolated AF — proxy for local mutual coupling (Li et al. ESA idea) |
| **Amplitude gate** | High weight where sub-block amplitude is low (nulls / sidelobes) |
| **Spatial gate + aux bias** | Bottleneck PSL/null context steers per-pixel correction |
| **Main-beam shield** | Differentiable HPBW mask; blocks residual in main lobe |
| **Bounded α_main / α_null** | `α_main ≤ 0.12`, `α_null ≥ 0.65` — prevents main-beam drift |
| **PSL head** | Scalar sidelobe level (E-plane cut) |
| **Null-depth head** | Top-10 deepest null depths (monotonic, sorted) |
| **FiLM decoder** | Bottleneck modulates decoder features |
| **Self-attention bottleneck** | Same as EnhancedResUNet cGAN |

### Loss (composed pattern, not residual-only)

| Term | Weight (default) | Region |
|------|------------------:|--------|
| L1 + SSIM on residual | 100 + 100 | Tiered: nulls ×3, sidelobes ×2.5 |
| Main-beam preservation | 55 | Inside HPBW: composed ≈ sub_block |
| Composed null L1 | 22 | truth < peak − 20 dB |
| Sidelobe band L1 | 10 | −20 to −3 dB below peak |
| E-plane cut L1 | 6 | Sparse cut at φ_peak (ESA-inspired) |
| PSL head L1 | 3 | Scalar |
| Null-depth head L1 | 8 | Top-K scalars |

**Validation score** (checkpoint selection): `0.30·RMSE_main + 0.30·RMSE_main+SL + 0.40·RMSE_null` on 6×6 val split.

---

## Comparison to other models @ 8×8

Approximate comparison — test splits differ slightly (v2 bundle n=101 vs others n=100).

| Model | Full MAE | MAE ≥ −10 dB | Null fill | Null RMSE | SLL err | R² |
|-------|---------:|-------------:|----------:|----------:|--------:|---:|
| Sub-block baseline | 4.75 | **0.40** | 33% | 7.03 | 2.32 | 0.59 |
| cGAN ResUNet + LS α | 3.29 | **0.42** | **42%** | 4.70 | **1.88** | — |
| Multi-head v1 + LS α | 3.35 | 0.68 | 39% | 4.66 | 4.76 | 0.82 |
| **Multi-head v2 + LS α** | **3.06** | 0.44 | **44%** | **4.38** | 3.32 | **0.84** |

**Strengths of v2:** best full-pattern MAE/R², best null RMSE and null fill, main beam nearly as good as baseline.  
**Trade-off:** SLL error higher than cGAN (3.3 vs 1.9 dB) — auxiliary PSL head MAE ~4.2 dB suggests room to tighten sidelobe amplitude matching.

---

## Reproducibility

### Prerequisites

Same processed data as the main 6×6 pipeline:

```
processed/stage1_6x6_extras.npz
processed/norm_stats_stage1_6x6.npz
processed/split_indices_stage1_6x6.npz
processed/subblock_compositions_v2.npz
processed/antenna_data_4to8.h5
predicted_8x8_full_bundle.npz          # for 8×8 eval
datasets_6x6sub-block_hfss/            # fingerprint channel
```

### Train (stage 1 @ 6×6)

```bash
python -m scripts.train_stage1_6x6_multihead_v2
python -m scripts.train_stage1_6x6_multihead_v2 --resume   # continue
```

- Checkpoints: `checkpoints/stage1_6x6_multihead_v2/`
- `best_generator.pt` — EMA weights, selected by composed val score
- **Note:** 6-channel model; old 5-channel v2 checkpoints are incompatible (retrain required)

### Evaluate (stage 2 zero-shot @ 8×8)

```bash
python -m scripts.eval_multihead_v2
python -m scripts.eval_multihead_v2 --no-alpha    # skip LS α calibration
python -m scripts.eval_multihead_v2 --no-shield   # skip main-beam zeroing
```

Outputs in `results/stage2_multihead_v2/`:

| File | Content |
|------|---------|
| `metrics.txt` | Full / regional / antenna metrics |
| `comparison.txt` | Masked RMSE/MAE table vs MATLAB and baseline |
| `comparison_{best,median,worst}_sample_*.png` | 6-panel heatmaps |
| `cuts_00.png` … `cuts_09.png` | E/H plane cuts |
| `error_distribution.png` | Pointwise error histogram |
| `scatter_pred_vs_true.png` | 200k subsample scatter |
| `stage2_multihead_v2_test_metrics.npz` | Per-sample MAE, α, aux head outputs |

---

## Project structure (this model)

```
src/models/multihead_v2.py              # Model, loss, EMA, inference helpers
scripts/train_stage1_6x6_multihead_v2.py
scripts/eval_multihead_v2.py
checkpoints/stage1_6x6_multihead_v2/
results/stage2_multihead_v2/
esa/README_ESA.md                       # Future local AEP surrogate spec
```

---

## Limitations

1. **Magnitude-only sub-block floor (~3.8 dB at 6×6 self-compose)** — dominant irreducible error; complex AEP export from HFSS would help most.
2. **SLL vs cGAN** — v2 prioritises null fill and global fidelity over sidelobe-level accuracy.
3. **16×16 eval not yet run** for v2 — v1 multi-head reached 3.72 dB MAE at 16×16; v2 should be evaluated with the same `zeroshot_6x6_to_16x16.py` pattern adapted for 6 channels.
4. **Auxiliary head calibration** — PSL/null scalar heads are training guides; composed-pattern losses drive final quality.
5. **Fingerprint always from 6×6 sub-blocks** — no scale-specific HFSS prior at 8×8/16×16 except via composition.

---

## Changelog: v1 → v2

| Change | Why |
|--------|-----|
| +6th channel `coupling_gap` | ESA-style local mismatch signal |
| Main-beam shield (train + infer) | Sub-block compose already good in HPBW |
| Bounded α_main / α_null | v1 α_main drifted to ~0.44, hurt main beam |
| Fixed gating formula | Removed redundant double-gate |
| E-plane cut loss | Sparse angular supervision |
| Balanced val score | Protect main beam during early stopping |
| LS α + shield at 8×8 eval | Matches cGAN zero-shot protocol |

---

## Citation / lineage

Built on the 6×6 sub-block reconstruction branch (`6x6_recon_approach_052526`). Multi-head v2 extends the gated ResUNet v1 design with physics-informed regional correction inspired by equivalent-small-array (ESA) mutual-coupling modeling (Li et al., IEEE TAP 2025).
