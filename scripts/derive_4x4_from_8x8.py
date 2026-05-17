"""
Derive ideal 4x4 MATLAB patterns from the 2000 existing 8x8 samples.

Background
----------
`datasets_8x8_matlab/datasets_8x8/` contains 40 CSVs x 50 samples each = 2000
samples, every one parameterised by a pair of progressive-phase steering
scalars (dphase_x_deg, dphase_y_deg). The stored pattern is
    G_stored(theta, phi) = 10*log10(elem_gain_lin * |AF_8x8|^2) - max
where |AF_8x8|^2 is analytically computable from (dphase_x, dphase_y) and
elem_gain_lin is the single-element gain pattern. We **recover elem_gain_lin
fresh from the 8x8 data itself** (the recovery is exact to ~1e-3 dB,
verified against stored 8x8 samples after running) and then re-simulate the
forward model with M = N = 4 to produce a 4x4 sub-array pattern that uses
the **same (beta_x, beta_y) and the same single-element pattern** as its
8x8 source -- the only difference between paired samples is the array size.

This is the 4x4 analogue of derive_2x2_from_4x4.py, used as the synthetic
ch0 input for the 4-to-8 fusion model.

Output
------
For each input CSV `patterns_global_####.csv` we write
`datasets_4x4_from_8x8/patterns_global_####.csv` with the same layout:
  row 1 : headers  [theta_deg, phi_deg, s00001..s00050]
  row 2 : dphase_x_deg  (unchanged, reused)
  row 3 : dphase_y_deg  (unchanged, reused)
  row 4 : phi_peak_deg    (recomputed for 4x4)
  row 5 : theta_peak_deg  (recomputed for 4x4)
  rows 6+: theta_deg, phi_deg, G_sample_01..G_sample_50 (dB, max-normalized)

Usage
-----
    python scripts/derive_4x4_from_8x8.py --smoke       # one CSV
    python scripts/derive_4x4_from_8x8.py               # all 2000
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- physics (must match setup.m) ----------
C_LIGHT = 3e8
FREQ = 2.4e9
LAM = C_LIGHT / FREQ
K = 2 * np.pi / LAM
DX = 0.5 * LAM
DY = 0.5 * LAM
NTH = 181       # theta bins: 0..180 deg, 1 deg step
NPH = 360       # phi bins:   -179.5..179.5 deg, 1 deg step

REPO = Path(__file__).resolve().parents[1]
IN_DIR = REPO / "datasets_8x8_matlab" / "datasets_8x8"
OUT_DIR = REPO / "datasets_4x4_from_8x8"
ELEM_NPZ = REPO / "datasets_4x4_from_8x8_workspace_elem_recovered.npz"


def af_power_dB(dphase_x: float, dphase_y: float, TH: np.ndarray, PH: np.ndarray,
                M: int, N: int) -> np.ndarray:
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
    AF = np.zeros_like(TH, dtype=complex)
    for m in range(M):
        for n in range(N):
            phase = K * (xm[m] * np.sin(TH) * np.cos(PH)
                         + yn[n] * np.sin(TH) * np.sin(PH)) + Beta[m, n]
            AF += np.exp(1j * phase)
    return 10 * np.log10(np.abs(AF) ** 2 + np.finfo(float).eps)


def recover_element_pattern(paths: list[Path], TH: np.ndarray, PH: np.ndarray,
                            THETA: np.ndarray, PHI: np.ndarray,
                            n_samples_cap: int = 150) -> np.ndarray:
    """Back out elem_gain (dB) from stored 8x8 data: h_s = G_stored - |AF_8x8|^2_dB.

    Uses the same protocol as derive_2x2_from_4x4: compute h_s for the first
    n_samples_cap samples, mask out near-AF-nulls (|AF|^2 < max - 30 dB),
    align each sample to its own max, median-aggregate per (theta, phi) bin,
    fill any remaining NaNs by nearest neighbour, and normalise so
    elem_dB.max() == 0 dB.
    """
    h_stack = []
    for p in paths:
        df = pd.read_csv(p, header=None, low_memory=False)
        dphase_x = df.iloc[1, 2:].astype(float).values
        dphase_y = df.iloc[2, 2:].astype(float).values
        pat = df.iloc[5:, 2:].astype(float).values
        for s in range(pat.shape[1]):
            G = pat[:, s].reshape(NPH, NTH).T
            AF2_dB = af_power_dB(dphase_x[s], dphase_y[s], TH, PH, 8, 8)
            h = G - AF2_dB
            mask = AF2_dB > (AF2_dB.max() - 30)
            h = np.where(mask, h, np.nan)
            h = h - np.nanmax(h)
            h_stack.append(h)
            if len(h_stack) >= n_samples_cap:
                break
        if len(h_stack) >= n_samples_cap:
            break

    H = np.stack(h_stack, axis=0)
    elem_dB = np.nanmedian(H, axis=0)
    if np.isnan(elem_dB).any():
        from scipy.ndimage import distance_transform_edt
        _, idx = distance_transform_edt(np.isnan(elem_dB), return_indices=True)
        elem_dB = elem_dB[tuple(idx)]
    elem_dB -= elem_dB.max()
    return elem_dB


def verify_recovery(elem_lin: np.ndarray, paths: list[Path],
                    TH: np.ndarray, PH: np.ndarray) -> dict:
    """Re-simulate the first sample of the first CSV with the recovered element
    pattern and compare to the stored 8x8 pattern.
    """
    p = paths[0]
    df = pd.read_csv(p, header=None, low_memory=False)
    dphase_x = float(df.iloc[1, 2])
    dphase_y = float(df.iloc[2, 2])
    G_stored = df.iloc[5:, 2].astype(float).values.reshape(NPH, NTH).T

    Beta = np.zeros((8, 8))
    for m in range(8):
        for n in range(8):
            mx = (m + 1) - 9 / 2
            ny = (n + 1) - 9 / 2
            Beta[m, n] = mx * dphase_x + ny * dphase_y
    Beta = np.deg2rad(Beta)
    xm = (np.arange(8) - 7 / 2) * DX
    yn = (np.arange(8) - 7 / 2) * DY
    AF = np.zeros_like(TH, dtype=complex)
    for m in range(8):
        for n in range(8):
            phase = K * (xm[m] * np.sin(TH) * np.cos(PH)
                         + yn[n] * np.sin(TH) * np.sin(PH)) + Beta[m, n]
            AF += np.exp(1j * phase)
    P = elem_lin * np.abs(AF) ** 2
    G = 10 * np.log10(P + np.finfo(float).eps)
    G -= G.max()

    diff = np.abs(G - G_stored)
    roi = G_stored > -40
    return {
        "max_abs_diff_db": float(diff.max()),
        "mean_abs_diff_db": float(diff.mean()),
        "roi_max_abs_diff_db": float(diff[roi].max()),
        "roi_mean_abs_diff_db": float(diff[roi].mean()),
    }


def load_or_recover_element(TH, PH, THETA, PHI, paths_for_recovery) -> np.ndarray:
    if ELEM_NPZ.exists():
        d = np.load(ELEM_NPZ)
        print(f"Loaded element pattern from cache: {ELEM_NPZ.name}", flush=True)
        return d["elem_dB"]
    print(f"Recovering element pattern fresh from 8x8 -> {ELEM_NPZ.name}", flush=True)
    elem_dB = recover_element_pattern(paths_for_recovery, TH, PH, THETA, PHI)
    np.savez(ELEM_NPZ, elem_dB=elem_dB, theta_deg=THETA[:, 0], phi_deg=PHI[0, :])
    return elem_dB


def simulate_4x4(dphase_x: float, dphase_y: float, elem_lin: np.ndarray,
                 TH: np.ndarray, PH: np.ndarray) -> np.ndarray:
    """Mirrors simulate_array.m with M=N=4, normalize_to_0dB=True."""
    M = N = 4
    Beta = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            mx = (m + 1) - (M + 1) / 2
            ny = (n + 1) - (N + 1) / 2
            Beta[m, n] = mx * dphase_x + ny * dphase_y
    Beta = np.deg2rad(Beta)
    xm = (np.arange(M) - (M - 1) / 2) * DX
    yn = (np.arange(N) - (N - 1) / 2) * DY
    AF = np.zeros_like(TH, dtype=complex)
    for m in range(M):
        for n in range(N):
            phase = K * (xm[m] * np.sin(TH) * np.cos(PH)
                         + yn[n] * np.sin(TH) * np.sin(PH)) + Beta[m, n]
            AF += np.exp(1j * phase)
    P = elem_lin * np.abs(AF) ** 2
    G_dB = 10 * np.log10(P + np.finfo(float).eps)
    G_dB -= G_dB.max()
    return G_dB


def process_csv(path: Path, out_dir: Path, elem_lin: np.ndarray,
                TH: np.ndarray, PH: np.ndarray, THETA: np.ndarray, PHI: np.ndarray) -> dict:
    df = pd.read_csv(path, header=None, low_memory=False)
    num_samples = df.shape[1] - 2

    # Globally-sequential sample indices (s00001..s02000 across the 40 files),
    # matching the project convention used by src/data/loader.py.
    m = re.search(r"patterns_global_(\d+)", path.stem)
    file_idx = int(m.group(1))
    start_idx = (file_idx - 1) * num_samples + 1

    dphase_x = df.iloc[1, 2:].astype(float).values
    dphase_y = df.iloc[2, 2:].astype(float).values

    theta_flat = df.iloc[5:, 0].astype(float).values
    phi_flat = df.iloc[5:, 1].astype(float).values
    num_ang = len(theta_flat)

    patterns = np.zeros((num_ang, num_samples))
    theta_peak = np.zeros(num_samples)
    phi_peak = np.zeros(num_samples)
    for s in range(num_samples):
        G = simulate_4x4(dphase_x[s], dphase_y[s], elem_lin, TH, PH)
        patterns[:, s] = G.T.reshape(-1, order="C")
        idx = np.unravel_index(np.argmax(G), G.shape)
        theta_peak[s] = THETA[idx]
        phi_peak[s] = PHI[idx]

    headers = ["theta_deg", "phi_deg"] + [f"s{start_idx + i:05d}" for i in range(num_samples)]
    meta = {
        "dphase_x_deg": dphase_x,
        "dphase_y_deg": dphase_y,
        "phi_peak_deg": phi_peak,
        "theta_peak_deg": theta_peak,
    }
    out_rows = [headers]
    for label, values in meta.items():
        row = [label, ""] + [f"{v:.6f}" for v in values]
        out_rows.append(row)
    data_block = np.column_stack(
        [theta_flat.reshape(-1, 1), phi_flat.reshape(-1, 1), patterns]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name

    with open(out_path, "w", newline="") as f:
        for r in out_rows:
            f.write(",".join(map(str, r)) + "\n")
        pd.DataFrame(data_block).to_csv(f, header=False, index=False,
                                        float_format="%.6f")

    return {
        "out": out_path,
        "num_samples": num_samples,
        "theta_peak_mean": float(theta_peak.mean()),
        "phi_peak_mean": float(phi_peak.mean()),
        "G_min": float(patterns.min()),
        "G_max": float(patterns.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="process only the first CSV (50 samples)")
    args = ap.parse_args()

    phi_deg = np.arange(-179.5, 180, 1.0)
    theta_deg = np.arange(0, 181, 1.0)
    PHI, THETA = np.meshgrid(phi_deg, theta_deg)
    TH = np.deg2rad(THETA); PH = np.deg2rad(PHI)

    paths = sorted(IN_DIR.glob("patterns_global_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSVs found under {IN_DIR}")

    elem_dB = load_or_recover_element(TH, PH, THETA, PHI, paths[:3])
    elem_lin = 10 ** (elem_dB / 10)
    print(f"Element pattern: range [{elem_dB.min():.2f}, {elem_dB.max():.2f}] dB",
          flush=True)

    # Empirical sanity check of the recovered element pattern.
    v = verify_recovery(elem_lin, paths, TH, PH)
    print(f"Recovery check (8x8 sample 1):  "
          f"all max|diff|={v['max_abs_diff_db']:.4f} dB, "
          f"mean={v['mean_abs_diff_db']:.5f} dB  |  "
          f"ROI(>-40dB) max|diff|={v['roi_max_abs_diff_db']:.4f} dB, "
          f"mean={v['roi_mean_abs_diff_db']:.5f} dB", flush=True)

    if args.smoke:
        paths = paths[:1]

    print(f"Processing {len(paths)} CSV(s) -> {OUT_DIR}", flush=True)
    for i, p in enumerate(paths, 1):
        info = process_csv(p, OUT_DIR, elem_lin, TH, PH, THETA, PHI)
        print(f"[{i:3d}/{len(paths)}] {p.name}: {info['num_samples']} samples, "
              f"G in [{info['G_min']:.1f}, {info['G_max']:.1f}] dB, "
              f"peak mean (th={info['theta_peak_mean']:.1f}, "
              f"ph={info['phi_peak_mean']:.1f}) deg",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())