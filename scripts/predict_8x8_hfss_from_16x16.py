"""
Run the trained 4-to-8 sub-array synthesis cGAN over the 2000 (matlab_4x4,
hfss_pred_4x4) pairs already packed into processed/antenna_data_4to8.h5,
producing predicted HFSS 8x8 patterns ("hfss_pred_8x8") that become the ch4
ANCHOR at the N=16 inference stage of the 5-channel residual model.

This is the natural N=8 -> 8x8 step of the cascade: at the N=16 inference
stage of train_residual_multiscale.py we want a "real-coupling reference"
pattern at one scale step below the target. The 4-to-8 sub-array cGAN is the
only model in this branch that produces 8x8 patterns from sub-array data, so
it's the right anchor producer.

Inputs to the 4-to-8 sub-array cGAN (matches scripts/train_cgan_4to8_subarray.py):
    ch0 : matlab_4x4_n        (normalised by norm_stats_4x4_from_8x8.npz)
    ch1 : hfss_pred_4x4_n     (same norm stats; cascade input from the
                               4x4 cGAN -- already in antenna_data_4to8.h5)
    ch2 : residual_n          (matlab_4x4_n - hfss_pred_4x4_n)
    ch3 : dphase_x / 180      (broadcast scalar)
    ch4 : dphase_y / 180      (broadcast scalar)
Output:
    hfss_pred_8x8_dB          (denormalised by norm_stats_8x8.npz; the model's
                               training targets were per-sample max-normed to
                               0 dB so the output is also max-normed)

Source artefact: processed/antenna_data_4to8.h5  (matlab_4x4, hfss_pred_4x4,
                                                  metadata, theta/phi grids)

Output: datasets_8x8_hfss_pred_from_16x16/patterns_global_####.csv
        Same canonical CSV layout the rest of the project uses (5 metadata
        rows + grid block). 40 files * 50 samples = 2000 patterns, written
        in the same global ordering as antenna_data_4to8.h5 so the downstream
        scripts/preprocess_data_8to16_subarray.py packs them back into
        processed/antenna_data_8to16_subarray.h5 row-aligned with the 4to8
        metadata.

Usage:
    python -m scripts.predict_8x8_hfss_from_16x16 --smoke   # one CSV (50 samples)
    python -m scripts.predict_8x8_hfss_from_16x16           # all 40 files / 2000 samples
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
import pandas as pd
import torch

from src.config import PROCESSED_DIR, CHECKPOINTS_DIR, DEVICE, N_THETA, N_PHI
from scripts.train_cgan_2to4_fusion_no_m4 import (
    EnhancedResUNetGenerator, GEN_BASE, ATTN_HEADS,
)

# ── Paths ───────────────────────────────────────────────────────────────────
SOURCE_H5 = PROCESSED_DIR / "antenna_data_4to8.h5"
OUT_DIR = PROJECT_ROOT / "datasets_8x8_hfss_pred_from_16x16"
NORM_4X4 = PROCESSED_DIR / "norm_stats_4x4_from_8x8.npz"
NORM_8X8 = PROCESSED_DIR / "norm_stats_8x8.npz"
CKPT = CHECKPOINTS_DIR / "cgan_resunet_patchgan_4to8_subarray" / "best_generator.pt"

# ── Project-wide sample partitioning (must match the 4-to-8 fusion file) ───
N_FILES = 40
N_CONFIGS_PER_FILE = 50
N_TOTAL = N_FILES * N_CONFIGS_PER_FILE   # 2000

BATCH = 16


def load_generator() -> torch.nn.Module:
    if not CKPT.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {CKPT}. Train it via "
            "scripts/train_cgan_4to8_subarray.py first.")
    g = EnhancedResUNetGenerator(in_ch=5, out_ch=1, base=GEN_BASE,
                                 attn_heads=ATTN_HEADS).to(DEVICE)
    g.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    g.eval()
    return g


def load_norm_stats():
    if not NORM_4X4.exists():
        raise FileNotFoundError(f"Norm stats missing: {NORM_4X4}")
    if not NORM_8X8.exists():
        raise FileNotFoundError(f"Norm stats missing: {NORM_8X8}")
    s4 = np.load(NORM_4X4); s8 = np.load(NORM_8X8)
    mean_4x4 = s4["mean"].astype(np.float32)
    std_4x4 = np.maximum(s4["std"].astype(np.float32), 1e-6)
    mean_8x8 = s8["mean"].astype(np.float32)
    std_8x8 = np.maximum(s8["std"].astype(np.float32), 1e-6)
    return mean_4x4, std_4x4, mean_8x8, std_8x8


@torch.no_grad()
def predict_chunk(generator: torch.nn.Module,
                  matlab_4x4: np.ndarray,    # (n, NTH, NPH) float32, dB
                  hfss_pred_4x4: np.ndarray, # (n, NTH, NPH) float32, dB
                  dphase_x: np.ndarray,      # (n,) float32, deg
                  dphase_y: np.ndarray,      # (n,) float32, deg
                  norm_stats) -> np.ndarray:
    """Return predicted hfss_pred_8x8 in dB, shape (n, NTH, NPH)."""
    mean_4x4, std_4x4, mean_8x8, std_8x8 = norm_stats
    n = matlab_4x4.shape[0]
    out_db = np.zeros((n, N_THETA, N_PHI), dtype=np.float32)

    for i in range(0, n, BATCH):
        j = min(i + BATCH, n)
        m4_b = matlab_4x4[i:j]
        hp4_b = hfss_pred_4x4[i:j]
        bx = dphase_x[i:j].astype(np.float32)
        by = dphase_y[i:j].astype(np.float32)

        m4_n = (m4_b - mean_4x4[None]) / std_4x4[None]
        hp4_n = (hp4_b - mean_4x4[None]) / std_4x4[None]
        residual_n = m4_n - hp4_n

        bs = j - i
        x = np.empty((bs, 5, N_THETA, N_PHI), dtype=np.float32)
        x[:, 0] = m4_n
        x[:, 1] = hp4_n
        x[:, 2] = residual_n
        x[:, 3] = (bx[:, None, None] / 180.0).astype(np.float32)
        x[:, 4] = (by[:, None, None] / 180.0).astype(np.float32)

        x_t = torch.from_numpy(x).to(DEVICE, non_blocking=True)
        y_n = generator(x_t).cpu().numpy()[:, 0]      # (bs, NTH, NPH) z-scored
        y_db = y_n * std_8x8[None] + mean_8x8[None]   # back to dB
        out_db[i:j] = y_db.astype(np.float32)

    return out_db


def write_csv(out_path: Path, file_idx: int,
              dphase_x: np.ndarray, dphase_y: np.ndarray,
              preds_db: np.ndarray, theta_grid: np.ndarray,
              phi_grid: np.ndarray) -> dict:
    """Write the project's canonical CSV layout for one file (50 samples).

    Layout:
        row 0 : headers theta_deg, phi_deg, s00001..s02000 (globally indexed)
        row 1 : dphase_x_deg
        row 2 : dphase_y_deg
        row 3 : phi_peak_deg     (recomputed from preds_db)
        row 4 : theta_peak_deg   (recomputed from preds_db)
        rows 5+: theta_deg, phi_deg, sample_1, sample_2, ...
    """
    n = preds_db.shape[0]

    # Per-sample peak metadata (in degrees)
    theta_peak = np.zeros(n, dtype=np.float64)
    phi_peak = np.zeros(n, dtype=np.float64)
    for s in range(n):
        ti, pi = np.unravel_index(np.argmax(preds_db[s]), preds_db[s].shape)
        theta_peak[s] = theta_grid[ti]
        phi_peak[s] = phi_grid[pi]

    # Canonical CSV layout: phi outer, theta inner (matches read_pattern_csv
    # in scripts/preprocess_data_8to16_subarray.py and preprocess_data_4to8_subarray.py
    # which call `flat.reshape(N_PHI, N_THETA, num_samples)`).
    PHI_GRID, THETA_GRID = np.meshgrid(phi_grid, theta_grid, indexing="ij")
    theta_flat = THETA_GRID.reshape(-1)    # length N_PHI*N_THETA
    phi_flat = PHI_GRID.reshape(-1)
    # Each sample's pattern goes from (NTH, NPH) -> (NPH, NTH) -> flat.
    # The `.T.reshape(-1)` makes the resulting row order: row r = (phi=r//NTH,
    # theta=r%NTH) which is what the reader's reshape(NPH, NTH, n) expects.
    flat_samples = np.empty((N_PHI * N_THETA, n), dtype=np.float32)
    for s in range(n):
        flat_samples[:, s] = preds_db[s].T.reshape(-1)

    start_idx = (file_idx - 1) * N_CONFIGS_PER_FILE + 1
    headers = ["theta_deg", "phi_deg"] + [f"s{start_idx + i:05d}" for i in range(n)]

    out_rows = [headers]
    for label, values in (("dphase_x_deg", dphase_x),
                          ("dphase_y_deg", dphase_y),
                          ("phi_peak_deg", phi_peak),
                          ("theta_peak_deg", theta_peak)):
        out_rows.append([label, ""] + [f"{v:.6f}" for v in values])

    data_block = np.column_stack(
        [theta_flat.reshape(-1, 1), phi_flat.reshape(-1, 1), flat_samples]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        for r in out_rows:
            f.write(",".join(map(str, r)) + "\n")
        pd.DataFrame(data_block).to_csv(f, header=False, index=False,
                                        float_format="%.6f")

    return {
        "num_samples": n,
        "G_min": float(preds_db.min()),
        "G_max": float(preds_db.max()),
        "theta_peak_mean": float(theta_peak.mean()),
        "phi_peak_mean": float(phi_peak.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="process only the first 50 samples (file 0001)")
    args = ap.parse_args()

    if not SOURCE_H5.exists():
        raise FileNotFoundError(
            f"{SOURCE_H5}: build it first via "
            "scripts/preprocess_data_4to8_fusion.py")

    print(f"Device: {DEVICE}", flush=True)
    G = load_generator()
    norm_stats = load_norm_stats()
    print(f"Loaded 4-to-8 sub-array cGAN ({sum(p.numel() for p in G.parameters())/1e6:.2f}M params)",
          flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing predictions -> {OUT_DIR.relative_to(PROJECT_ROOT)}", flush=True)

    n_files = 1 if args.smoke else N_FILES
    t0 = time.time()
    with h5py.File(SOURCE_H5, "r") as f:
        if "matlab_4x4" not in f or "hfss_pred_4x4" not in f:
            raise RuntimeError(
                f"{SOURCE_H5} is missing matlab_4x4 / hfss_pred_4x4 datasets. "
                "Rebuild via scripts/preprocess_data_4to8_fusion.py.")
        theta_grid = f["theta_grid"][:].astype(np.float64)
        phi_grid = f["phi_grid"][:].astype(np.float64)

        for file_idx in range(1, n_files + 1):
            start = (file_idx - 1) * N_CONFIGS_PER_FILE
            stop = start + N_CONFIGS_PER_FILE

            m4 = f["matlab_4x4"][start:stop].astype(np.float32)
            hp4 = f["hfss_pred_4x4"][start:stop].astype(np.float32)
            meta = f["metadata"][start:stop].astype(np.float64)
            dpx = meta[:, 0].astype(np.float32)
            dpy = meta[:, 1].astype(np.float32)

            preds_db = predict_chunk(G, m4, hp4, dpx, dpy, norm_stats)

            out_path = OUT_DIR / f"patterns_global_{file_idx:04d}.csv"
            info = write_csv(out_path, file_idx, dpx, dpy, preds_db,
                             theta_grid, phi_grid)
            elapsed = time.time() - t0
            print(f"[{file_idx:3d}/{n_files}] {out_path.name}: "
                  f"{info['num_samples']} samples, "
                  f"G in [{info['G_min']:.1f}, {info['G_max']:.1f}] dB, "
                  f"peak mean (th={info['theta_peak_mean']:.1f}, "
                  f"ph={info['phi_peak_mean']:.1f}) deg | "
                  f"elapsed {elapsed:.1f}s", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
