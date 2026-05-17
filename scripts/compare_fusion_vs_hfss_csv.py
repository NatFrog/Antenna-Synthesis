"""
Direct CSV-vs-CSV comparison: fusion-cGAN predictions vs 4x4 HFSS ground truth.

For the held-out test samples:
  1. Run fusion cGAN inference (same pipeline as evaluate_cgan_2to4_fusion.py).
  2. Load each test sample's 4x4 HFSS truth from the original CSV (float32, no
     HDF5 float16 quantisation).
  3. Compute per-sample metrics: RMSE, MAE, max|error|, Pearson r, peak
     direction error.
  4. Report best/worst/median test samples, the global distribution summary,
     and save a per-sample CSV.

Outputs:
    results/cgan_resunet_patchgan_2to4/compare_vs_hfss_csv.csv
    results/cgan_resunet_patchgan_2to4/compare_worst5.png
    results/cgan_resunet_patchgan_2to4/compare_best5.png

Usage:
    python -m scripts.compare_fusion_vs_hfss_csv
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
import matplotlib.pyplot as plt

from src.config import PROCESSED_DIR, CHECKPOINTS_DIR, RESULTS_DIR, DEVICE, BATCH_SIZE, N_CONFIGS_PER_FILE
from src.data.loader import load_single_csv, get_file_path
from scripts.train_cgan_2to4_fusion import FusionDataset, ResUNetGenerator

HFSS_4X4_DIR = PROJECT_ROOT / "datasets_4x4_hfss" / "datasets_4x4_hfss"
HDF5 = PROCESSED_DIR / "antenna_data_2to4.h5"
NORM_2X2 = PROCESSED_DIR / "norm_stats_2x2.npz"
NORM_4X4 = PROCESSED_DIR / "norm_stats.npz"
SPLITS = PROCESSED_DIR / "split_indices_2to4.npz"
CKPT = CHECKPOINTS_DIR / "cgan_resunet_patchgan_2to4" / "best_generator.pt"
RESULTS = RESULTS_DIR / "cgan_resunet_patchgan_2to4"


def peak_direction_error(pred, truth, theta_grid, phi_grid):
    """Return angular error (deg) between pred peak and truth peak."""
    pt, pp = np.unravel_index(np.argmax(pred), pred.shape)
    tt, tp = np.unravel_index(np.argmax(truth), truth.shape)
    p_th, p_ph = theta_grid[pt], phi_grid[pp]
    t_th, t_ph = theta_grid[tt], phi_grid[tp]
    # great-circle angle between unit vectors
    p = np.deg2rad([p_th, p_ph]); t = np.deg2rad([t_th, t_ph])
    cos_sep = (np.sin(p[0]) * np.sin(t[0]) * np.cos(p[1] - t[1])
               + np.cos(p[0]) * np.cos(t[0]))
    return float(np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0))))


def run_inference(test_idx):
    """Run fusion generator on test indices; return (N, 181, 360) dB predictions."""
    s2 = np.load(NORM_2X2); s4 = np.load(NORM_4X4)
    m2m = s2["mean"].astype(np.float32); s2s = np.maximum(s2["std"].astype(np.float32), 1e-6)
    m4m = s4["mean"].astype(np.float32); s4s = np.maximum(s4["std"].astype(np.float32), 1e-6)

    ds = FusionDataset(HDF5, test_idx, m2m, s2s, m4m, s4s)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    G = ResUNetGenerator(in_ch=5, out_ch=1, base=32).to(DEVICE)
    G.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    G.eval()

    preds_n = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Inference"):
            preds_n.append(G(x.to(DEVICE)).cpu().numpy())
    preds_norm = np.concatenate(preds_n)[:, 0]
    return preds_norm * s4s + m4m          # denormalised dB


def load_csv_ground_truth(test_idx, theta_grid, phi_grid):
    """Load 4x4 HFSS truth patterns for the given global indices directly from
    the source CSVs (not from HDF5), returning float32 values."""
    # Group by file to avoid re-reading the same CSV
    by_file = {}
    for gi in test_idx:
        fi = gi // N_CONFIGS_PER_FILE + 1
        ci = gi % N_CONFIGS_PER_FILE
        by_file.setdefault(fi, []).append((gi, ci))

    N = len(test_idx)
    truth = np.empty((N, 181, 360), dtype=np.float32)
    idx_of = {int(gi): k for k, gi in enumerate(test_idx)}
    meta_dx = np.empty(N); meta_dy = np.empty(N)
    for fi in tqdm(sorted(by_file), desc="Reading HFSS CSVs"):
        d = load_single_csv(get_file_path(HFSS_4X4_DIR, fi), file_index=fi)
        for gi, ci in by_file[fi]:
            k = idx_of[int(gi)]
            truth[k] = d["patterns"][ci].astype(np.float32)
            col = d["config_cols"][ci]
            meta_dx[k] = d["metadata"][col]["dphase_x"]
            meta_dy[k] = d["metadata"][col]["dphase_y"]
    return truth, meta_dx, meta_dy


def plot_samples(indices, preds, truth, matlab, theta_grid, phi_grid, title, out_path):
    """5-sample grid: each row is (matlab, pred, truth, |pred-truth|)."""
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.3 * n), squeeze=False)
    for i, gi_local in enumerate(indices):
        M = matlab[gi_local]; P = preds[gi_local]; T = truth[gi_local]
        diff = np.abs(P - T)
        vmin, vmax = -40, 0
        cols = [
            ("4x4 MATLAB (baseline)", M, vmin, vmax, "viridis"),
            ("Fusion cGAN pred",      P, vmin, vmax, "viridis"),
            ("4x4 HFSS truth",        T, vmin, vmax, "viridis"),
            ("|pred - truth| (dB)",   diff, 0, 5,   "hot"),
        ]
        for j, (ttl, mat, lo, hi, cmap) in enumerate(cols):
            im = axes[i, j].imshow(mat, aspect="auto", origin="lower",
                                   extent=[phi_grid[0], phi_grid[-1], theta_grid[0], theta_grid[-1]],
                                   vmin=lo, vmax=hi, cmap=cmap)
            axes[i, j].set_title(f"{ttl}" + (f"\nsample #{gi_local}" if j == 0 else ""))
            axes[i, j].set_xlabel("phi (deg)"); axes[i, j].set_ylabel("theta (deg)")
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    sp = np.load(SPLITS)
    test_idx = np.sort(sp["test"].astype(np.int64))

    # Pull theta/phi grids from HDF5 (canonical)
    with h5py.File(HDF5, "r") as f:
        theta_grid = f["theta_grid"][:].astype(np.float32)
        phi_grid = f["phi_grid"][:].astype(np.float32)
        matlab_4x4_raw = f["matlab_4x4"][test_idx].astype(np.float32)   # for plotting

    # 1) Inference
    preds = run_inference(test_idx)
    print(f"preds shape: {preds.shape}  range: [{preds.min():.2f}, {preds.max():.2f}] dB")

    # 2) HFSS truth straight from CSV
    truth, dphase_x, dphase_y = load_csv_ground_truth(test_idx, theta_grid, phi_grid)
    print(f"truth shape: {truth.shape}  range: [{truth.min():.2f}, {truth.max():.2f}] dB")

    # 3) Per-sample metrics (all pixel-wise in dB, no normalisation)
    N = len(test_idx)
    rmse = np.empty(N); mae = np.empty(N); maxerr = np.empty(N)
    pearson = np.empty(N); peakerr = np.empty(N)
    for i in range(N):
        P = preds[i]; T = truth[i]
        d = P - T
        rmse[i] = float(np.sqrt(np.mean(d * d)))
        mae[i] = float(np.mean(np.abs(d)))
        maxerr[i] = float(np.max(np.abs(d)))
        pearson[i] = float(np.corrcoef(P.ravel(), T.ravel())[0, 1])
        peakerr[i] = peak_direction_error(P, T, theta_grid, phi_grid)

    # 4) Summary
    def qs(x, name, unit):
        q = np.percentile(x, [0, 25, 50, 75, 100])
        print(f"  {name:18s}  min={q[0]:7.3f}  p25={q[1]:7.3f}  median={q[2]:7.3f}  "
              f"p75={q[3]:7.3f}  max={q[4]:7.3f}  mean={x.mean():7.3f}  {unit}")
    print(f"\nPer-sample stats over {N} held-out test samples:")
    qs(rmse,    "RMSE",             "dB")
    qs(mae,     "MAE",              "dB")
    qs(maxerr,  "max|error|",       "dB")
    qs(pearson, "Pearson r",        "(unitless)")
    qs(peakerr, "peak dir error",   "deg")

    # 5) Save per-sample table
    import pandas as pd
    out_csv = RESULTS / "compare_vs_hfss_csv.csv"
    df = pd.DataFrame({
        "global_idx": test_idx,
        "file_idx": test_idx // N_CONFIGS_PER_FILE + 1,
        "config_idx": test_idx % N_CONFIGS_PER_FILE,
        "dphase_x_deg": dphase_x,
        "dphase_y_deg": dphase_y,
        "rmse_db": rmse,
        "mae_db": mae,
        "max_abs_err_db": maxerr,
        "pearson_r": pearson,
        "peak_dir_err_deg": peakerr,
    })
    df.to_csv(out_csv, index=False)
    print(f"\nPer-sample comparison table -> {out_csv}")

    # 6) Worst and best cases
    worst = np.argsort(-rmse)[:5]
    best = np.argsort(rmse)[:5]
    print("\nWorst-5 by RMSE:")
    for i in worst:
        print(f"  sample gi={test_idx[i]:4d}  rmse={rmse[i]:.3f} dB  max|e|={maxerr[i]:.2f} dB  "
              f"peak_err={peakerr[i]:.2f} deg")
    print("\nBest-5 by RMSE:")
    for i in best:
        print(f"  sample gi={test_idx[i]:4d}  rmse={rmse[i]:.3f} dB  max|e|={maxerr[i]:.2f} dB  "
              f"peak_err={peakerr[i]:.2f} deg")

    plot_samples(worst, preds, truth, matlab_4x4_raw, theta_grid, phi_grid,
                 "Fusion cGAN vs 4x4 HFSS truth — worst-5 by RMSE",
                 RESULTS / "compare_worst5.png")
    plot_samples(best, preds, truth, matlab_4x4_raw, theta_grid, phi_grid,
                 "Fusion cGAN vs 4x4 HFSS truth — best-5 by RMSE",
                 RESULTS / "compare_best5.png")
    print(f"Saved visualisations -> {RESULTS}/compare_worst5.png and compare_best5.png")


if __name__ == "__main__":
    main()
