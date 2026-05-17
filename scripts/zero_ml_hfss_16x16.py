"""
Zero-ML physics-decomposition baseline for hfss_16x16 prediction.

Idea
----
A pattern factors approximately as
    G(theta, phi; M, N, beta) = element_eff(theta, phi) * |AF(theta, phi; M, N, beta)|^2
where element_eff is the embedded element pattern (carries mutual coupling)
and AF is the analytic array factor for an MxN array steered to beta.

If element_eff is approximately scale-transferable (the bet of approach 3),
we can:
  1. Recover element_eff from a stack of (hfss_NxN, beta) pairs at a small
     scale where data is cheap. We use hfss_8x8 train split (1600 samples).
  2. Synthesise hfss at any larger scale via element_eff + AF^2_dB(M, N, beta),
     per-sample max-normalised.

Recovery
--------
For each training sample i with steering beta_i:
    element_per_sample_i(theta, phi) = hfss_8x8_dB_i - AF^2_8x8_dB(beta_i)
This recovers the element shape + a per-sample additive constant (because the
project max-normalises every pattern to peak=0 dB). The constant cancels at
synthesis time, since we re-max-normalise the predicted pattern.

We mask sky directions where AF^2_dB(beta_i) is too far below its own peak
(say <= -30 dB) — recovery there is dominated by numerical noise / nulls in
the AF — and then take the median across samples to denoise.

Synthesis + evaluation
----------------------
For each held-out 4-to-8 test beta we compute AF^2_16x16_dB(beta), add the
recovered element_eff, max-normalise, and compare to hfss_16x16 from
processed/antenna_data_8to16_subarray.h5.

The 200 4-to-8 test betas are split:
    val_16x16  = sorted(test_idx)[:100]   # would be used for ML model selection
    test_16x16 = sorted(test_idx)[100:]   # honest zero-shot test
We report metrics on each separately.

Usage:
    python -m scripts.zero_ml_hfss_16x16
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
from tqdm import tqdm
from scipy.ndimage import minimum_filter
from scipy.signal import find_peaks

from src.config import N_THETA, N_PHI, PROCESSED_DIR, RESULTS_DIR, NULL_THRESHOLD_DB
from src.training.metrics import compute_batch_metrics
from src.evaluation.antenna_metrics import compute_antenna_metric_errors
from src.evaluation.visualization import (
    plot_pattern_comparison, plot_plane_cuts,
    plot_error_distribution, plot_scatter_pred_vs_true,
)

# ── Physics ─────────────────────────────────────────────────────────────────
C_LIGHT = 3e8
FREQ = 2.4e9
LAM = C_LIGHT / FREQ
K = 2 * np.pi / LAM
DX = 0.5 * LAM
DY = 0.5 * LAM

# ── Paths ───────────────────────────────────────────────────────────────────
HDF5_4TO8 = PROCESSED_DIR / "antenna_data_4to8.h5"
HDF5_16X16 = PROCESSED_DIR / "antenna_data_8to16_subarray.h5"
SPLITS = PROCESSED_DIR / "split_indices_4to8.npz"
ELEM_OUT = PROCESSED_DIR / "element_dB_hfss_8x8_recovered.npz"
RESULTS = RESULTS_DIR / "zero_ml_hfss_16x16"

# Recovery threshold: ignore points where AF^2 is too far below peak
AF_VALID_THRESHOLD_DB = -30.0

# ── Geometry grids (precomputed once) ───────────────────────────────────────
PHI_DEG = np.arange(-179.5, 180, 1.0)
THETA_DEG = np.arange(0, 181, 1.0)
PHI_GRID, THETA_GRID = np.meshgrid(PHI_DEG, THETA_DEG)
TH = np.deg2rad(THETA_GRID)
PH = np.deg2rad(PHI_GRID)
SIN_TH = np.sin(TH); COS_PH = np.cos(PH); SIN_PH = np.sin(PH)


def af2_dB(M: int, N: int, dphase_x: float, dphase_y: float) -> np.ndarray:
    """Analytic |AF|^2 in dB for an MxN centered array with progressive phase."""
    Beta = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            mx = (m + 1) - (M + 1) / 2
            ny = (n + 1) - (N + 1) / 2
            Beta[m, n] = mx * dphase_x + ny * dphase_y
    Beta = np.deg2rad(Beta)
    xm = (np.arange(M) - (M - 1) / 2) * DX
    yn = (np.arange(N) - (N - 1) / 2) * DY
    AF = np.zeros_like(TH, dtype=np.complex128)
    for m in range(M):
        for n in range(N):
            phase = K * (xm[m] * SIN_TH * COS_PH + yn[n] * SIN_TH * SIN_PH) + Beta[m, n]
            AF += np.exp(1j * phase)
    return 10 * np.log10(np.abs(AF) ** 2 + np.finfo(float).eps)


def recover_element_from_hfss_8x8(train_idx: np.ndarray) -> np.ndarray:
    """Recover element_dB(theta, phi) from training-split hfss_8x8.

    For each sample, recover element_per_sample = hfss_dB - AF^2_dB(beta).
    Mask points where AF^2_dB is too far below its peak (recovery is unreliable
    there). Align per-sample (subtract per-sample max so additive constants
    cancel) and take the median across samples.
    """
    n_train = len(train_idx)
    print(f"Recovering element_dB from {n_train} hfss_8x8 training samples", flush=True)

    with h5py.File(HDF5_4TO8, "r") as f:
        meta = f["metadata"][train_idx].astype(np.float64)
        # Read in chunks to keep memory in check (1600 * 181 * 360 * 4 = 416 MB ok)
        hfss_8x8 = f["hfss_8x8"][train_idx].astype(np.float32)

    elem_stack = np.full((n_train, N_THETA, N_PHI), np.nan, dtype=np.float32)
    t0 = time.time()
    for i in tqdm(range(n_train), desc="element recovery"):
        af2 = af2_dB(8, 8, meta[i, 0], meta[i, 1])
        af2_peak = af2.max()
        valid = af2 > (af2_peak + AF_VALID_THRESHOLD_DB)   # mask deep AF nulls
        elem = hfss_8x8[i] - af2.astype(np.float32)
        # Align per-sample so additive constants cancel: subtract the per-sample
        # max over the *valid* region (mainbeam-area-dominated max).
        elem_align = elem - float(elem[valid].max())
        elem_align[~valid] = np.nan
        elem_stack[i] = elem_align

    # Median across samples (NaN-aware) — robust to per-sample tail noise.
    print(f"Aggregating {n_train} per-sample recoveries (median)...", flush=True)
    element_dB = np.nanmedian(elem_stack, axis=0).astype(np.float32)
    # Replace any remaining NaN (in pixels with no valid coverage) by the
    # column-wise minimum, so the synthesis doesn't NaN out.
    if np.isnan(element_dB).any():
        bad = np.isnan(element_dB)
        print(f"  {bad.sum()} pixels never covered; filling with global min", flush=True)
        element_dB[bad] = float(np.nanmin(element_dB))

    # Re-anchor to peak=0 for human-readability.
    element_dB -= element_dB.max()
    n_valid_per_pixel = (~np.isnan(elem_stack)).sum(axis=0)

    print(f"Recovery done in {time.time()-t0:.1f}s. element_dB range "
          f"[{element_dB.min():.2f}, {element_dB.max():.2f}] dB. "
          f"Median sample-coverage per pixel: {int(np.median(n_valid_per_pixel))}/{n_train}",
          flush=True)
    return element_dB


def synthesize(element_dB: np.ndarray, betas: np.ndarray, M: int, N: int) -> np.ndarray:
    """Synthesize predicted patterns at scale (M, N) for given beta list.

    Returns (n_samples, N_THETA, N_PHI) float32, per-sample max-normalised to 0 dB.
    """
    n = len(betas)
    out = np.zeros((n, N_THETA, N_PHI), dtype=np.float32)
    for i in tqdm(range(n), desc=f"synth {M}x{N}"):
        af2 = af2_dB(M, N, betas[i, 0], betas[i, 1]).astype(np.float32)
        pat = element_dB + af2
        pat -= pat.max()
        out[i] = pat
    return out


def evaluate(name: str, preds: np.ndarray, truth: np.ndarray, mat_guide: np.ndarray | None):
    """Print metrics + return a flat dict for saving."""
    pm = compute_batch_metrics(preds, truth)
    print(f"\n=== {name}: pattern metrics ===")
    for k, v in pm.items():
        print(f"  {k}: {v:.6f}")

    ae = []
    for i in range(len(preds)):
        try:
            ae.append(compute_antenna_metric_errors(preds[i], truth[i]))
        except Exception:
            pass
    avg_ant = {}
    if ae:
        for key in ae[0]:
            vals = [e[key] for e in ae if not np.isnan(e[key])]
            avg_ant[key] = float(np.mean(vals)) if vals else float("nan")
        print(f"\n=== {name}: antenna metrics ===")
        for k, v in avg_ant.items():
            print(f"  {k}: {v:.6f}")

    # Null metrics relative to mat_guide (the analytical 16x16 we computed)
    null_rmses = []; nfa = nft = 0; depth_errs = []
    if mat_guide is not None:
        for i in range(len(preds)):
            mat = mat_guide[i]
            mask_null = mat < (mat.max() + NULL_THRESHOLD_DB)
            if mask_null.sum() > 0:
                null_rmses.append(float(np.sqrt(np.mean(
                    (preds[i][mask_null] - truth[i][mask_null]) ** 2))))
                pf = preds[i][mask_null] - mat[mask_null]
                tf = truth[i][mask_null] - mat[mask_null]
                nfa += int((np.abs(pf - tf) < 2.0).sum())
                nft += int(mask_null.sum())
            lm = minimum_filter(mat, size=5)
            ilm = (mat == lm) & mask_null
            if ilm.sum() > 0:
                mp = np.argwhere(ilm); mv = mat[ilm]
                di = np.argsort(mv)[:10]
                for k in di:
                    t, p = mp[k]
                    depth_errs.append(abs(preds[i][t, p] - truth[i][t, p]))
        nm = {
            "rmse_at_nulls_db": float(np.mean(null_rmses)) if null_rmses else float("nan"),
            "null_depth_error_db": float(np.mean(depth_errs)) if depth_errs else float("nan"),
            "null_fill_accuracy_pct": float(nfa / max(nft, 1) * 100),
        }
        print(f"\n=== {name}: null metrics (matlab_16x16 -20 dB threshold) ===")
        for k, v in nm.items():
            print(f"  {k}: {v:.6f}")
    else:
        nm = {}

    return {**{f"pattern_{k}": v for k, v in pm.items()},
            **{f"antenna_{k}": v for k, v in avg_ant.items()},
            **{f"null_{k}": v for k, v in nm.items()}}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    sp = np.load(SPLITS)
    train_idx = np.sort(sp["train"].astype(np.int64))
    test_idx = np.sort(sp["test"].astype(np.int64))
    val16_idx = test_idx[:100]
    test16_idx = test_idx[100:]
    print(f"Splits: {len(train_idx)} train (for element recovery) | "
          f"{len(val16_idx)} val_16x16 | {len(test16_idx)} test_16x16",
          flush=True)

    # ── 1. Recover element_dB from hfss_8x8 train split ─────────────────────
    if ELEM_OUT.exists():
        print(f"Loading cached element from {ELEM_OUT.name}", flush=True)
        element_dB = np.load(ELEM_OUT)["element_dB"].astype(np.float32)
    else:
        element_dB = recover_element_from_hfss_8x8(train_idx)
        np.savez(ELEM_OUT, element_dB=element_dB)
        print(f"Saved {ELEM_OUT.name}", flush=True)

    # ── 2. Sanity check: synthesize hfss_8x8 at training betas, compare ─────
    print("\n--- Sanity: synthesize hfss_8x8 at first 100 train betas ---", flush=True)
    with h5py.File(HDF5_4TO8, "r") as f:
        sanity_idx = train_idx[:100]
        sanity_meta = f["metadata"][sanity_idx].astype(np.float64)
        sanity_truth = f["hfss_8x8"][sanity_idx].astype(np.float32)
    sanity_pred = synthesize(element_dB, sanity_meta[:, :2], 8, 8)
    sanity_pm = compute_batch_metrics(sanity_pred, sanity_truth)
    print("=== Sanity (8x8 self-fit on train betas) ===")
    for k, v in sanity_pm.items():
        print(f"  {k}: {v:.6f}")

    # ── 3. Synthesize hfss_16x16 for val + test betas ───────────────────────
    print("\n--- Synthesize hfss_16x16 for val_16x16 + test_16x16 ---", flush=True)
    with h5py.File(HDF5_4TO8, "r") as f:
        all_test_meta = f["metadata"][test_idx].astype(np.float64)
    val_betas = all_test_meta[:100, :2]
    test_betas = all_test_meta[100:, :2]
    val_preds = synthesize(element_dB, val_betas, 16, 16)
    test_preds = synthesize(element_dB, test_betas, 16, 16)

    # Ground truth hfss_16x16
    with h5py.File(HDF5_16X16, "r") as f:
        val_truth = f["hfss_16x16"][val16_idx].astype(np.float32)
        test_truth = f["hfss_16x16"][test16_idx].astype(np.float32)

    # Analytical matlab_16x16 (from cached MATLAB element) for null masking + baseline.
    cached_matlab_elem = np.load(PROJECT_ROOT / "datasets_4x4_from_8x8_workspace_elem_recovered.npz")
    elem_matlab_dB = cached_matlab_elem["elem_dB"].astype(np.float32)
    print("Computing analytical matlab_16x16 for val + test betas (baseline)...", flush=True)
    val_mat16 = synthesize(elem_matlab_dB, val_betas, 16, 16)
    test_mat16 = synthesize(elem_matlab_dB, test_betas, 16, 16)

    # ── 4. Evaluate ─────────────────────────────────────────────────────────
    val_metrics = evaluate("VAL_16X16 (zero-ML)", val_preds, val_truth, val_mat16)
    test_metrics = evaluate("TEST_16X16 (zero-ML)", test_preds, test_truth, test_mat16)
    base_val = evaluate("VAL_16X16 (baseline matlab_16x16)", val_mat16, val_truth, val_mat16)
    base_test = evaluate("TEST_16X16 (baseline matlab_16x16)", test_mat16, test_truth, test_mat16)

    # ── 5. Save metrics + a few plots ────────────────────────────────────────
    with open(RESULTS / "metrics.txt", "w") as f:
        f.write("Zero-ML physics-decomposition baseline for hfss_16x16\n")
        f.write(f"Element recovered from {len(train_idx)} hfss_8x8 train samples\n")
        f.write(f"Sanity (8x8 self-fit): "
                f"rmse={sanity_pm['rmse_db']:.4f} dB, r={sanity_pm['pearson_r']:.4f}\n\n")
        for name, d in [("VAL_16X16", val_metrics), ("TEST_16X16", test_metrics),
                        ("BASELINE_VAL_16X16", base_val), ("BASELINE_TEST_16X16", base_test)]:
            f.write(f"--- {name} ---\n")
            for k, v in d.items():
                f.write(f"{k}: {v:.6f}\n")
            f.write("\n")
    print(f"\nMetrics saved to {RESULTS}/metrics.txt", flush=True)

    # Visualize element pattern + a few synthesis examples
    np.savez(RESULTS / "predictions.npz",
             val_preds=val_preds, val_truth=val_truth, val_idx=val16_idx,
             test_preds=test_preds, test_truth=test_truth, test_idx=test16_idx)

    vis_indices = np.linspace(0, len(test_preds) - 1, 10, dtype=int)
    for k, idx in enumerate(vis_indices):
        plot_pattern_comparison(
            test_mat16[idx], test_preds[idx], test_truth[idx],
            title=f"zero-ML hfss_16x16 test sample {idx} (4to8_test_idx={test16_idx[idx]})",
            save_path=str(RESULTS / f"comparison_{k:02d}.png"),
        )
        plot_plane_cuts(
            test_mat16[idx], test_preds[idx], test_truth[idx],
            title=f"zero-ML hfss_16x16 test sample {idx}",
            save_path=str(RESULTS / f"cuts_{k:02d}.png"),
        )

    errors = test_preds - test_truth
    plot_error_distribution(
        errors, title="zero-ML hfss_16x16 error distribution (test)",
        save_path=str(RESULTS / "error_distribution.png"),
    )
    plot_scatter_pred_vs_true(
        test_preds, test_truth,
        title="zero-ML hfss_16x16: predicted vs hfss_16x16 (test)",
        save_path=str(RESULTS / "scatter_pred_vs_true.png"),
    )

    import os
    print(f"Generated {len(os.listdir(RESULTS))} files in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
