"""
Run the trained 4x4 cGAN generator over the 2000 ideal MATLAB 4x4 patterns
derived from the 8x8 dataset, producing predicted HFSS-equivalent patterns.

This is the 4x4 analogue of predict_2x2_hfss_from_4x4.py and produces ch1
of the 4-to-8 fusion model's input stack.

Pipeline per sample (identical to scripts/evaluate_cgan_2x2.py / the 4x4 cGAN):
    matlab_dB -> (matlab - mean_map) / std_map  -> 3-ch input
    [x_norm, dphase_x/180, dphase_y/180]  -> generator  -> y_norm
    predicted_hfss_dB = y_norm * std_map + mean_map

Input : datasets_4x4_from_8x8/patterns_global_####.csv   (40 files x 50 samples)
Output: datasets_4x4_hfss_pred_from_8x8/patterns_global_####.csv
        Same CSV layout as the input (and the source 8x8 MATLAB dataset):
            row 1 : headers theta_deg, phi_deg, s00001..s02000 (globally sequential)
            row 2 : dphase_x_deg (carried over from input)
            row 3 : dphase_y_deg (carried over from input)
            row 4 : phi_peak_deg   (recomputed from predicted pattern)
            row 5 : theta_peak_deg (recomputed from predicted pattern)
            rows 6+: theta_deg, phi_deg, predicted HFSS pattern in dB (not re-normalized)

Usage:
    python scripts/predict_4x4_hfss_from_8x8.py --smoke   # one CSV (50 samples)
    python scripts/predict_4x4_hfss_from_8x8.py           # all 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import PROCESSED_DIR, CHECKPOINTS_DIR, DEVICE, N_THETA, N_PHI
from scripts.train_cgan import UNetGenerator

IN_DIR = PROJECT_ROOT / "datasets_4x4_from_8x8"
OUT_DIR = PROJECT_ROOT / "datasets_4x4_hfss_pred_from_8x8"
# 4x4 cGAN training stats (per-pixel mean/std on the existing 4x4 MATLAB train split)
NORM_STATS = PROCESSED_DIR / "norm_stats.npz"
CKPT = CHECKPOINTS_DIR / "cgan_unet_patchgan" / "best_generator.pt"

# Inference batch size (GPU memory permitting). 16 matches training; (3, 181, 360) fits easily.
BATCH = 16


def load_generator() -> torch.nn.Module:
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint missing: {CKPT}")
    g = UNetGenerator(in_channels=3, out_channels=1).to(DEVICE)
    g.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    g.eval()
    return g


def load_norm_stats():
    if not NORM_STATS.exists():
        raise FileNotFoundError(f"Norm stats missing: {NORM_STATS}")
    s = np.load(NORM_STATS)
    mean_map = s["mean"].astype(np.float32)
    std_map = np.maximum(s["std"].astype(np.float32), 1e-6)
    return mean_map, std_map


def load_csv_samples(path: Path) -> dict:
    """Parse a derived 4x4 CSV into arrays shaped for inference."""
    df = pd.read_csv(path, header=None, low_memory=False)
    headers = df.iloc[0, :].astype(str).tolist()
    dphase_x = df.iloc[1, 2:].astype(float).values
    dphase_y = df.iloc[2, 2:].astype(float).values
    theta_flat = df.iloc[5:, 0].astype(float).values
    phi_flat = df.iloc[5:, 1].astype(float).values
    pat = df.iloc[5:, 2:].astype(float).values  # (65160, 50)

    n = pat.shape[1]
    patterns = np.empty((n, N_THETA, N_PHI), dtype=np.float32)
    for s in range(n):
        # Storage is column-major (theta fast, phi slow): reshape to (N_PHI, N_THETA), then transpose.
        patterns[s] = pat[:, s].reshape(N_PHI, N_THETA).T.astype(np.float32)

    return {
        "headers": headers,
        "dphase_x": dphase_x,
        "dphase_y": dphase_y,
        "theta_flat": theta_flat,
        "phi_flat": phi_flat,
        "patterns": patterns,
    }


@torch.no_grad()
def predict_file(samples: dict, generator, mean_map, std_map) -> np.ndarray:
    """Run the generator over all samples in a file. Returns (N, 181, 360) predicted dB."""
    patterns = samples["patterns"]
    dphase_x = samples["dphase_x"].astype(np.float32) / 180.0
    dphase_y = samples["dphase_y"].astype(np.float32) / 180.0
    n = patterns.shape[0]

    preds_db = np.empty_like(patterns, dtype=np.float32)
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        x0 = (patterns[start:end] - mean_map) / std_map
        dx_chan = np.broadcast_to(
            dphase_x[start:end, None, None], x0.shape
        ).astype(np.float32)
        dy_chan = np.broadcast_to(
            dphase_y[start:end, None, None], x0.shape
        ).astype(np.float32)
        inp = np.stack([x0, dx_chan, dy_chan], axis=1)
        inp_t = torch.from_numpy(inp).to(DEVICE)
        out = generator(inp_t).cpu().numpy()[:, 0]
        preds_db[start:end] = out * std_map + mean_map

    return preds_db


def write_csv(out_path: Path, samples: dict, preds_db: np.ndarray):
    """Write predicted patterns in the same CSV layout as the input."""
    headers = samples["headers"]
    dphase_x = samples["dphase_x"]
    dphase_y = samples["dphase_y"]
    theta_flat = samples["theta_flat"]
    phi_flat = samples["phi_flat"]

    n = preds_db.shape[0]
    theta_peak = np.empty(n); phi_peak = np.empty(n)
    theta_grid = np.arange(0, N_THETA, 1.0)
    phi_grid = np.arange(-179.5, 180, 1.0)
    for s in range(n):
        idx = np.unravel_index(np.argmax(preds_db[s]), preds_db[s].shape)
        theta_peak[s] = theta_grid[idx[0]]
        phi_peak[s] = phi_grid[idx[1]]

    pred_flat = np.empty((N_THETA * N_PHI, n), dtype=np.float32)
    for s in range(n):
        pred_flat[:, s] = preds_db[s].T.reshape(-1, order="C")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        f.write(",".join(headers) + "\n")
        def _meta(label, values):
            f.write(",".join([label, ""] + [f"{v:.6f}" for v in values]) + "\n")
        _meta("dphase_x_deg", dphase_x)
        _meta("dphase_y_deg", dphase_y)
        _meta("phi_peak_deg", phi_peak)
        _meta("theta_peak_deg", theta_peak)
        block = np.column_stack([theta_flat.reshape(-1, 1),
                                 phi_flat.reshape(-1, 1),
                                 pred_flat])
        pd.DataFrame(block).to_csv(f, header=False, index=False, float_format="%.6f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="process only the first CSV (50 samples)")
    args = ap.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    generator = load_generator()
    mean_map, std_map = load_norm_stats()
    print(f"mean_map: [{mean_map.min():.2f}, {mean_map.max():.2f}] dB   "
          f"std_map: [{std_map.min():.2f}, {std_map.max():.2f}] dB", flush=True)

    paths = sorted(IN_DIR.glob("patterns_global_*.csv"))
    if args.smoke:
        paths = paths[:1]

    print(f"Processing {len(paths)} CSV(s) -> {OUT_DIR}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, p in enumerate(paths, 1):
        samples = load_csv_samples(p)
        preds_db = predict_file(samples, generator, mean_map, std_map)
        out_path = OUT_DIR / p.name
        write_csv(out_path, samples, preds_db)
        g_min, g_max = float(preds_db.min()), float(preds_db.max())
        print(f"[{i:3d}/{len(paths)}] {p.name}: 50 samples, "
              f"pred G in [{g_min:.1f}, {g_max:.1f}] dB", flush=True)
    print(f"Total: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())