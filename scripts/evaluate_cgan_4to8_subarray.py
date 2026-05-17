"""Evaluate the 4-to-8 sub-array synthesis cGAN on the held-out test set.

Mirrors scripts/evaluate_cgan_2to4_fusion_no_m4.py exactly, just one scale up.
The baseline for comparison is `matlab_4x4` (analytical) vs `hfss_8x8` target,
i.e. "no model correction" — same convention as the 4-to-8 fusion evaluator
on feature/cgan-4to8-fusion, which lets the metrics tables be diffed
1-to-1 against that branch.

Usage:
    python -m scripts.evaluate_cgan_4to8_subarray
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import minimum_filter
from scipy.signal import find_peaks

from src.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
    BATCH_SIZE, DEVICE, NULL_THRESHOLD_DB,
)
from src.training.metrics import compute_batch_metrics
from src.evaluation.antenna_metrics import compute_antenna_metric_errors
from src.evaluation.visualization import (
    plot_pattern_comparison, plot_plane_cuts,
    plot_error_distribution, plot_scatter_pred_vs_true,
)
from scripts.train_cgan_2to4_fusion_no_m4 import (
    EnhancedResUNetGenerator, GEN_BASE, ATTN_HEADS,
)
from scripts.train_cgan_4to8_subarray import Subarray4to8Dataset

HDF5 = PROCESSED_DIR / "antenna_data_4to8_subarray.h5"
NORM_4X4 = PROCESSED_DIR / "norm_stats_4x4_from_8x8.npz"
NORM_8X8 = PROCESSED_DIR / "norm_stats_8x8.npz"
SPLITS = PROCESSED_DIR / "split_indices_4to8.npz"
CKPT = CHECKPOINTS_DIR / "cgan_resunet_patchgan_4to8_subarray" / "best_generator.pt"
RESULTS = RESULTS_DIR / "cgan_resunet_patchgan_4to8_subarray"


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    sp = np.load(SPLITS)
    test_idx = np.sort(sp["test"].astype(np.int64))
    s4 = np.load(NORM_4X4); s8 = np.load(NORM_8X8)
    mean_4x4 = s4["mean"].astype(np.float32); std_4x4 = np.maximum(s4["std"].astype(np.float32), 1e-6)
    mean_8x8 = s8["mean"].astype(np.float32); std_8x8 = np.maximum(s8["std"].astype(np.float32), 1e-6)

    ds = Subarray4to8Dataset(HDF5, test_idx, mean_4x4, std_4x4, mean_8x8, std_8x8,
                             augment_noise=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    G = EnhancedResUNetGenerator(in_ch=5, out_ch=1, base=GEN_BASE,
                                 attn_heads=ATTN_HEADS).to(DEVICE)
    G.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    G.eval()

    preds_n, targets_n = [], []
    with torch.no_grad():
        for x, y, _null in tqdm(loader, desc="Inference"):
            preds_n.append(G(x.to(DEVICE)).cpu().numpy())
            targets_n.append(y.numpy())
    preds_db = np.concatenate(preds_n)[:, 0] * std_8x8 + mean_8x8
    targets_db = np.concatenate(targets_n)[:, 0] * std_8x8 + mean_8x8

    with h5py.File(HDF5, "r") as f:
        matlab_4x4_raw = f["matlab_4x4"][test_idx].astype(np.float32)
    n_test = len(preds_db)
    print(f"\nTest samples: {n_test}")

    # ── Pattern-level metrics ──
    pm = compute_batch_metrics(preds_db, targets_db)
    print("\n=== Pattern-level metrics (4-to-8 subarray cGAN) ===")
    for k, v in pm.items():
        print(f"  {k}: {v:.6f}")

    # ── Antenna metrics ──
    ae = []
    for i in range(n_test):
        try:
            ae.append(compute_antenna_metric_errors(preds_db[i], targets_db[i]))
        except Exception:
            pass
    avg_ant = {}
    for key in ae[0]:
        vals = [e[key] for e in ae if not np.isnan(e[key])]
        avg_ant[key] = float(np.mean(vals)) if vals else float("nan")
    print("\n=== Antenna-specific metrics ===")
    for k, v in avg_ant.items():
        print(f"  {k}: {v:.6f}")

    # ── Null metrics (against matlab_4x4 since no matlab_8x8 channel was used) ──
    null_rmses, non_null_rmses, depth_errs = [], [], []
    nfa, nft = 0, 0
    for i in range(n_test):
        pred, tgt, mat = preds_db[i], targets_db[i], matlab_4x4_raw[i]
        peak = mat.max()
        mask_null = mat < (peak + NULL_THRESHOLD_DB)
        if mask_null.sum() > 0:
            null_rmses.append(float(np.sqrt(np.mean((pred[mask_null] - tgt[mask_null]) ** 2))))
            pf = pred[mask_null] - mat[mask_null]
            tf2 = tgt[mask_null] - mat[mask_null]
            nfa += int((np.abs(pf - tf2) < 2.0).sum())
            nft += int(mask_null.sum())
        mask_nn = ~mask_null
        if mask_nn.sum() > 0:
            non_null_rmses.append(float(np.sqrt(np.mean((pred[mask_nn] - tgt[mask_nn]) ** 2))))
        lm = minimum_filter(mat, size=5)
        ilm = (mat == lm) & mask_null
        if ilm.sum() > 0:
            mp = np.argwhere(ilm)
            mv = mat[ilm]
            di = np.argsort(mv)[:10]
            for idx in di:
                t, p = mp[idx]
                depth_errs.append(abs(pred[t, p] - tgt[t, p]))
    null_metrics = {
        "rmse_at_nulls_db": float(np.mean(null_rmses)),
        "rmse_at_non_nulls_db": float(np.mean(non_null_rmses)),
        "null_depth_error_db": float(np.mean(depth_errs)),
        "null_fill_accuracy_pct": float(nfa / max(nft, 1) * 100),
    }
    print("\n=== Null-specific metrics (-20 dB threshold) ===")
    for k, v in null_metrics.items():
        print(f"  {k}: {v:.6f}")

    # ── Matched-position SLL ──
    fus_max_sll, fus_mean_sll = [], []
    for i in range(n_test):
        truth = targets_db[i]; pred = preds_db[i]
        pt, pp = np.unravel_index(np.argmax(truth), truth.shape)
        peak_val = truth[pt, pp]
        tcut = truth[:, pp]; pcut = pred[:, pp]
        peaks, _ = find_peaks(tcut, distance=3)
        if len(peaks) < 2: continue
        main_loc = np.argmin(np.abs(peaks - pt))
        sl = np.delete(peaks, main_loc)
        if len(sl) == 0: continue
        t_sl = tcut[sl] - peak_val
        p_sl = pcut[sl] - pred.max()
        errs = np.abs(p_sl - t_sl)
        fus_max_sll.append(float(errs[np.argmax(t_sl)]))
        fus_mean_sll.append(float(np.mean(errs)))
    sll_matched = float(np.mean(fus_max_sll))
    sll_mean = float(np.mean(fus_mean_sll))
    print(f"\n=== Matched-position SLL ===")
    print(f"  sll_matched_dominant_db: {sll_matched:.6f}")
    print(f"  sll_matched_mean_db:     {sll_mean:.6f}")

    # ── Baseline (4x4 MATLAB vs 8x8 HFSS, no correction) ──
    bm = compute_batch_metrics(matlab_4x4_raw, targets_db)
    print("\n=== Baseline (no correction, matlab_4x4 vs hfss_8x8) pattern metrics ===")
    for k, v in bm.items():
        print(f"  {k}: {v:.6f}")

    bae = []
    for i in range(n_test):
        try:
            bae.append(compute_antenna_metric_errors(matlab_4x4_raw[i], targets_db[i]))
        except Exception:
            pass
    b_ant = {}
    for key in bae[0]:
        vals = [e[key] for e in bae if not np.isnan(e[key])]
        b_ant[key] = float(np.mean(vals)) if vals else float("nan")
    print("\n=== Baseline antenna metrics ===")
    for k, v in b_ant.items():
        print(f"  {k}: {v:.6f}")

    b_null_rmses, b_depth = [], []
    bnfa, bnft = 0, 0
    for i in range(n_test):
        mat, tgt = matlab_4x4_raw[i], targets_db[i]
        peak = mat.max()
        mask = mat < (peak + NULL_THRESHOLD_DB)
        if mask.sum() > 0:
            b_null_rmses.append(float(np.sqrt(np.mean((mat[mask] - tgt[mask]) ** 2))))
            tf2 = tgt[mask] - mat[mask]
            bnfa += int((np.abs(tf2) < 2.0).sum())
            bnft += int(mask.sum())
        lm = minimum_filter(mat, size=5)
        ilm = (mat == lm) & mask
        if ilm.sum() > 0:
            mp = np.argwhere(ilm)
            mv = mat[ilm]
            di = np.argsort(mv)[:10]
            for idx in di:
                t, p = mp[idx]
                b_depth.append(abs(mat[t, p] - tgt[t, p]))
    print("\n=== Baseline null metrics ===")
    print(f"  rmse_at_nulls_db:       {np.mean(b_null_rmses):.6f}")
    print(f"  null_depth_error_db:    {np.mean(b_depth):.6f}")
    print(f"  null_fill_accuracy_pct: {bnfa / max(bnft, 1) * 100:.6f}")

    # ── Save metrics.txt ──
    metrics_out = {
        **{f"pattern_{k}": v for k, v in pm.items()},
        **{f"antenna_{k}": v for k, v in avg_ant.items()},
        "antenna_sll_error_matched_dominant_db": sll_matched,
        "antenna_sll_error_matched_mean_db": sll_mean,
        **{f"null_{k}": v for k, v in null_metrics.items()},
        "null_baseline_rmse_at_nulls_db": float(np.mean(b_null_rmses)),
        "null_baseline_null_depth_error_db": float(np.mean(b_depth)),
        "null_baseline_null_fill_accuracy_pct": float(bnfa / max(bnft, 1) * 100),
        **{f"baseline_{k}": v for k, v in bm.items()},
        **{f"baseline_antenna_{k}": v for k, v in b_ant.items()},
    }
    with open(RESULTS / "metrics.txt", "w") as f:
        f.write("Evaluation: cgan_resunet_patchgan_4to8_subarray\n")
        f.write(f"Test samples: {n_test}\n\n")
        for k, v in metrics_out.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"\nMetrics saved to {RESULTS}/metrics.txt")

    # ── Visualisations ──
    vis_indices = np.linspace(0, n_test - 1, 10, dtype=int)
    for idx in tqdm(vis_indices, desc="Saving plots"):
        plot_pattern_comparison(
            matlab_4x4_raw[idx], preds_db[idx], targets_db[idx],
            title=f"4-to-8 subarray cGAN test sample {idx}",
            save_path=str(RESULTS / f"comparison_{idx:04d}.png"),
        )
        plot_plane_cuts(
            matlab_4x4_raw[idx], preds_db[idx], targets_db[idx],
            title=f"4-to-8 subarray cGAN test sample {idx}",
            save_path=str(RESULTS / f"cuts_{idx:04d}.png"),
        )

    errors = preds_db - targets_db
    plot_error_distribution(
        errors, title="4-to-8 subarray cGAN error distribution",
        save_path=str(RESULTS / "error_distribution.png"),
    )
    plot_scatter_pred_vs_true(
        preds_db, targets_db, title="4-to-8 subarray cGAN: predicted vs true",
        save_path=str(RESULTS / "scatter_pred_vs_true.png"),
    )

    import os
    files = sorted(os.listdir(RESULTS))
    print(f"\nGenerated {len(files)} result files in {RESULTS}")


if __name__ == "__main__":
    main()
