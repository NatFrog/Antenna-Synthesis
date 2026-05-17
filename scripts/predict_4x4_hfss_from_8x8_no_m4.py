"""
Run the trained no-m4 sub-array synthesis cGAN over the 2000 (matlab_2x2,
hfss_2x2) pairs at the 8x8 dataset's beta values, producing predicted HFSS
4x4 patterns ("hfss_pred_4x4") that will become ch1 of the 4-to-8 sub-array
synthesis model.

Inputs to the no-m4 model (matches scripts/train_cgan_2to4_fusion_no_m4.py):
    ch0 : matlab_2x2_n        (normalised by norm_stats_2x2.npz)
    ch1 : hfss_2x2_n          (normalised by norm_stats_2x2.npz)
                              -- here using REAL HFSS 2x2 data the user
                              supplied at the 8x8 betas, in place of
                              hfss_pred_2x2 from the 2x2 cGAN.
    ch2 : residual_n          (matlab_2x2_n - hfss_2x2_n)
    ch3 : dphase_x / 180      (broadcast scalar)
    ch4 : dphase_y / 180      (broadcast scalar)
Output:
    hfss_pred_4x4_dB          (denormalised by norm_stats.npz)

Source files
------------
    matlab_2x2 : datasets_2x2_from_8x8/patterns_global_####.csv
                 (analytical, derived from cached 8x8-MATLAB elem pattern)
    hfss_2x2   : datasets_2x2from8x8_hfss/datasets_2x2from8x8_hfss/
                                          patterns_global_####.csv
                 (real HFSS 2x2 sims at the 8x8 betas)

Output: datasets_4x4_hfss_pred_from_8x8_no_m4/patterns_global_####.csv
        Same CSV layout as the input (5 metadata + grid block).

Usage:
    python scripts/predict_4x4_hfss_from_8x8_no_m4.py --smoke   # one CSV
    python scripts/predict_4x4_hfss_from_8x8_no_m4.py           # all 40
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
from scripts.train_cgan_2to4_fusion_no_m4 import (
    EnhancedResUNetGenerator, GEN_BASE, ATTN_HEADS,
)

MATLAB_DIR = PROJECT_ROOT / "datasets_2x2_from_8x8"
HFSS_DIR = PROJECT_ROOT / "datasets_2x2from8x8_hfss" / "datasets_2x2from8x8_hfss"
OUT_DIR = PROJECT_ROOT / "datasets_4x4_hfss_pred_from_8x8_no_m4"
NORM_2X2 = PROCESSED_DIR / "norm_stats_2x2.npz"
NORM_4X4 = PROCESSED_DIR / "norm_stats.npz"
CKPT = CHECKPOINTS_DIR / "cgan_resunet_patchgan_2to4_no_m4" / "best_generator.pt"

BATCH = 16


def load_generator() -> torch.nn.Module:
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint missing: {CKPT}")
    g = EnhancedResUNetGenerator(in_ch=5, out_ch=1, base=GEN_BASE,
                                 attn_heads=ATTN_HEADS).to(DEVICE)
    g.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    g.eval()
    return g


def load_norm_stats():
    s2 = np.load(NORM_2X2); s4 = np.load(NORM_4X4)
    mean_2x2 = s2["mean"].astype(np.float32)
    std_2x2 = np.maximum(s2["std"].astype(np.float32), 1e-6)
    mean_4x4 = s4["mean"].astype(np.float32)
    std_4x4 = np.maximum(s4["std"].astype(np.float32), 1e-6)
    return mean_2x2, std_2x2, mean_4x4, std_4x4


def read_csv_block(path: Path):
    """Return (theta_flat, phi_flat, dphase_x, dphase_y, patterns) where
    patterns has shape (N_THETA, N_PHI, num_samples) in dB.
    """
    df = pd.read_csv(path, header=None, low_memory=False)
    num_samples = df.shape[1] - 2
    dphase_x = df.iloc[1, 2:].astype(float).to_numpy()
    dphase_y = df.iloc[2, 2:].astype(float).to_numpy()
    theta_flat = df.iloc[5:, 0].astype(float).to_numpy()
    phi_flat = df.iloc[5:, 1].astype(float).to_numpy()
    flat = df.iloc[5:, 2:].astype(np.float32).to_numpy()  # (N_THETA*N_PHI, num_samples)
    # Each column is reshape(NPH, NTH).T = (NTH, NPH) per the project's CSV convention.
    pat = np.transpose(flat.reshape(N_PHI, N_THETA, num_samples), axes=(1, 0, 2))
    return theta_flat, phi_flat, dphase_x, dphase_y, pat, num_samples


@torch.no_grad()
def run_file(matlab_path: Path, hfss_path: Path, out_path: Path,
             generator: torch.nn.Module, norm_stats) -> dict:
    mean_2x2, std_2x2, mean_4x4, std_4x4 = norm_stats

    theta_flat, phi_flat, dpx_m, dpy_m, mat_db, num_samples = read_csv_block(matlab_path)
    _, _, dpx_h, dpy_h, hfss_db, num_h = read_csv_block(hfss_path)
    assert num_samples == num_h, f"{matlab_path.name}: sample count mismatch"
    assert np.allclose(dpx_m, dpx_h, atol=1e-3) and np.allclose(dpy_m, dpy_h, atol=1e-3), \
        f"{matlab_path.name}: dphase mismatch between matlab and hfss CSVs"

    preds_db = np.zeros((N_THETA, N_PHI, num_samples), dtype=np.float32)

    for i in range(0, num_samples, BATCH):
        j = min(i + BATCH, num_samples)
        bs = j - i
        m_b = mat_db[..., i:j]                 # (NTH, NPH, bs)
        h_b = hfss_db[..., i:j]
        bx = dpx_m[i:j].astype(np.float32)
        by = dpy_m[i:j].astype(np.float32)

        m_n = (m_b - mean_2x2[..., None]) / std_2x2[..., None]
        h_n = (h_b - mean_2x2[..., None]) / std_2x2[..., None]
        residual_n = m_n - h_n

        # (bs, 5, NTH, NPH)
        x = np.empty((bs, 5, N_THETA, N_PHI), dtype=np.float32)
        x[:, 0] = np.transpose(m_n, (2, 0, 1))
        x[:, 1] = np.transpose(h_n, (2, 0, 1))
        x[:, 2] = np.transpose(residual_n, (2, 0, 1))
        x[:, 3] = (bx[:, None, None] / 180.0).astype(np.float32)
        x[:, 4] = (by[:, None, None] / 180.0).astype(np.float32)

        x_t = torch.from_numpy(x).to(DEVICE, non_blocking=True)
        y_n = generator(x_t).cpu().numpy()[:, 0]  # (bs, NTH, NPH) normalised
        y_db = y_n * std_4x4 + mean_4x4
        preds_db[..., i:j] = np.transpose(y_db, (1, 2, 0))

    # peak metadata for the output CSV
    theta_peak = np.zeros(num_samples)
    phi_peak = np.zeros(num_samples)
    THETA_axis = np.linspace(0, 180, N_THETA)
    PHI_axis = np.linspace(-179.5, 179.5, N_PHI)
    for s in range(num_samples):
        idx = np.unravel_index(np.argmax(preds_db[..., s]), preds_db[..., s].shape)
        theta_peak[s] = THETA_axis[idx[0]]
        phi_peak[s] = PHI_axis[idx[1]]

    # Write CSV in the project's canonical layout
    file_idx_match = matlab_path.stem.split("_")[-1]
    file_idx = int(file_idx_match)
    start_idx = (file_idx - 1) * num_samples + 1

    headers = ["theta_deg", "phi_deg"] + [f"s{start_idx + i:05d}" for i in range(num_samples)]
    meta = {
        "dphase_x_deg": dpx_m,
        "dphase_y_deg": dpy_m,
        "phi_peak_deg": phi_peak,
        "theta_peak_deg": theta_peak,
    }
    out_rows = [headers]
    for label, values in meta.items():
        out_rows.append([label, ""] + [f"{v:.6f}" for v in values])

    flat_out = np.transpose(preds_db, (1, 0, 2)).reshape(N_PHI * N_THETA, num_samples)
    data_block = np.column_stack(
        [theta_flat.reshape(-1, 1), phi_flat.reshape(-1, 1), flat_out]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        for r in out_rows:
            f.write(",".join(map(str, r)) + "\n")
        pd.DataFrame(data_block).to_csv(f, header=False, index=False, float_format="%.6f")

    return {
        "num_samples": num_samples,
        "G_min": float(preds_db.min()),
        "G_max": float(preds_db.max()),
        "theta_peak_mean": float(theta_peak.mean()),
        "phi_peak_mean": float(phi_peak.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="process only the first CSV (50 samples)")
    args = ap.parse_args()

    if not MATLAB_DIR.exists():
        raise FileNotFoundError(MATLAB_DIR)
    if not HFSS_DIR.exists():
        raise FileNotFoundError(HFSS_DIR)

    matlab_paths = sorted(MATLAB_DIR.glob("patterns_global_*.csv"))
    hfss_paths = sorted(HFSS_DIR.glob("patterns_global_*.csv"))
    if len(matlab_paths) != len(hfss_paths):
        raise RuntimeError(
            f"matlab CSVs ({len(matlab_paths)}) != hfss CSVs ({len(hfss_paths)})")

    if args.smoke:
        matlab_paths = matlab_paths[:1]
        hfss_paths = hfss_paths[:1]

    print(f"Found {len(matlab_paths)} CSV pair(s). Loading model + norm stats...",
          flush=True)
    G = load_generator()
    norm_stats = load_norm_stats()
    print(f"Generator on {DEVICE}, batch={BATCH}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing predictions -> {OUT_DIR}", flush=True)

    t0 = time.time()
    for i, (mp, hp) in enumerate(zip(matlab_paths, hfss_paths), 1):
        if mp.name != hp.name:
            raise RuntimeError(f"mismatched pair: {mp.name} vs {hp.name}")
        out_path = OUT_DIR / mp.name
        info = run_file(mp, hp, out_path, G, norm_stats)
        elapsed = time.time() - t0
        print(f"[{i:3d}/{len(matlab_paths)}] {mp.name}: {info['num_samples']} samples, "
              f"G in [{info['G_min']:.1f}, {info['G_max']:.1f}] dB, "
              f"peak mean (th={info['theta_peak_mean']:.1f}, "
              f"ph={info['phi_peak_mean']:.1f}) deg | "
              f"elapsed {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
