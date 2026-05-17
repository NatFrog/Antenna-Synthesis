"""
Derive analytical (no-coupling) MATLAB 2x2 patterns at the 8x8 dataset's beta
values, reusing the cached 8x8-MATLAB-derived element pattern.

Background
----------
The 4-to-8 sub-array synthesis pipeline needs an analytical 2x2 channel
(matlab_2x2) at the same (beta_x, beta_y) values as the 2000 8x8 samples.
We use the per-element gain pattern that was already recovered exactly from
the analytical MATLAB 8x8 dataset (cached on the cgan-4to8-fusion branch as
`datasets_4x4_from_8x8_workspace_elem_recovered.npz`). Because that source
is coupling-free, the recovered element pattern is exact to ~1e-3 dB; it is
geometry-independent and reusable for any sub-array size.

For each (beta_x, beta_y) sample we then compute:
    matlab_2x2(theta, phi)
        = 10 * log10( elem_lin(theta, phi) * |AF_2x2(theta, phi; beta)|^2 )
        - max
and write the result with the same CSV layout as the other project datasets.

Note: the user's HFSS 2x2 dataset (`datasets_2x2from8x8_hfss/`) is *not*
used here. It feeds ch1 of the no-m4 model later in the pipeline (real-HFSS
2x2 with mutual coupling). Using it for element recovery would contaminate
the analytical reference channel with coupling residue.

Output
------
For each input CSV `patterns_global_####.csv` (sourced from the new HFSS 2x2
folder solely to read the (beta_x, beta_y) metadata in the project's
canonical CSV format) we write `datasets_2x2_from_8x8/patterns_global_####.csv`
with the same layout:
  row 1 : headers  [theta_deg, phi_deg, s00001..s00050]
  row 2 : dphase_x_deg     (unchanged, reused)
  row 3 : dphase_y_deg     (unchanged, reused)
  row 4 : phi_peak_deg     (recomputed for analytical 2x2)
  row 5 : theta_peak_deg   (recomputed for analytical 2x2)
  rows 6+: theta_deg, phi_deg, G_sample_01..G_sample_50 (dB, max-normalized)

Usage
-----
    python scripts/derive_2x2_from_8x8.py --smoke       # one CSV
    python scripts/derive_2x2_from_8x8.py               # all 40
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- physics (must match setup.m / derive_4x4_from_8x8.py) ----------
C_LIGHT = 3e8
FREQ = 2.4e9
LAM = C_LIGHT / FREQ
K = 2 * np.pi / LAM
DX = 0.5 * LAM
DY = 0.5 * LAM
NTH = 181       # theta bins: 0..180 deg, 1 deg step
NPH = 360       # phi bins:   -179.5..179.5 deg, 1 deg step

REPO = Path(__file__).resolve().parents[1]
IN_DIR = REPO / "datasets_2x2from8x8_hfss" / "datasets_2x2from8x8_hfss"
OUT_DIR = REPO / "datasets_2x2_from_8x8"
# Cached element pattern recovered from MATLAB 8x8 (analytical, coupling-free)
# on the cgan-4to8-fusion branch. Element pattern is geometry-independent,
# so it is valid for the 2x2 derivation here.
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


def load_cached_element() -> np.ndarray:
    """Load the cached 8x8-MATLAB-derived element pattern.

    The cache was produced once on the cgan-4to8-fusion branch by running
    elem = G_MATLAB_8x8 / |AF_8x8|^2 (per-sample, then masked + median),
    where G_MATLAB_8x8 is coupling-free analytical data, so the recovery is
    exact. Element patterns are geometry-independent -> reuse for 2x2.
    """
    if not ELEM_NPZ.exists():
        raise FileNotFoundError(
            f"Element pattern cache missing at {ELEM_NPZ.name}. "
            "Restore it from feature/cgan-4to8-fusion or re-derive via "
            "scripts/derive_4x4_from_8x8.py on that branch.")
    d = np.load(ELEM_NPZ)
    print(f"Loaded cached element pattern from {ELEM_NPZ.name}", flush=True)
    return d["elem_dB"]


def simulate_2x2(dphase_x: float, dphase_y: float, elem_lin: np.ndarray,
                 TH: np.ndarray, PH: np.ndarray) -> np.ndarray:
    """Mirrors simulate_array.m with M=N=2, normalize_to_0dB=True."""
    M = N = 2
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
                TH: np.ndarray, PH: np.ndarray, THETA: np.ndarray,
                PHI: np.ndarray) -> dict:
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
        G = simulate_2x2(dphase_x[s], dphase_y[s], elem_lin, TH, PH)
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

    elem_dB = load_cached_element()
    elem_lin = 10 ** (elem_dB / 10)
    print(f"Element pattern: range [{elem_dB.min():.2f}, {elem_dB.max():.2f}] dB",
          flush=True)

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
