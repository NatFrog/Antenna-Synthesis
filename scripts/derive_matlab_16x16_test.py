"""
Generate analytical (no-coupling) MATLAB 16x16 patterns for the 4-to-8 test
split betas only. Used at evaluation time to substitute matlab_16x16 in for
matlab_8x8 in ch5 of the with-m8 cGAN.

Method: same analytical pipeline as derive_2x2_from_8x8.py - the cached
8x8-MATLAB-recovered element pattern is geometry-independent, so the same
elem_lin reconstructs any MxN array via:
    matlab_MxN(theta, phi)
        = 10*log10(elem_lin * |AF_MxN(theta, phi; beta)|^2) - max
This matches the per-sample max-normalisation used by the project's other
matlab_* channels (verified vs antenna_data_4to8.h5: peaks all == 0 dB).

Output:
    processed/matlab_16x16_test.npz
        arr      : (200, 181, 360) float16   max-normalised dB
        test_idx : (200,) int64              indices into 4to8 dataset
        beta     : (200, 2)  float64         (dpx, dpy) for sanity-check

Usage:
    python -m scripts.derive_matlab_16x16_test
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

from src.config import N_THETA, N_PHI, PROCESSED_DIR

# Physics constants (must match setup.m / derive_2x2_from_8x8.py)
C_LIGHT = 3e8
FREQ = 2.4e9
LAM = C_LIGHT / FREQ
K = 2 * np.pi / LAM
DX = 0.5 * LAM
DY = 0.5 * LAM

ELEM_NPZ = PROJECT_ROOT / "datasets_4x4_from_8x8_workspace_elem_recovered.npz"
SOURCE_H5 = PROCESSED_DIR / "antenna_data_4to8.h5"
SPLITS = PROCESSED_DIR / "split_indices_4to8.npz"
OUT_NPZ = PROCESSED_DIR / "matlab_16x16_test.npz"

ARRAY_M = 16
ARRAY_N = 16


def simulate_array(M: int, N: int, dphase_x: float, dphase_y: float,
                   elem_lin: np.ndarray, TH: np.ndarray, PH: np.ndarray) -> np.ndarray:
    """Analytic MxN broadside array, progressive phase (dphase in degrees).

    Mirrors simulate_2x2 in derive_2x2_from_8x8.py with M, N parameterised.
    Returns G_dB max-normalised to 0 (matches the project's matlab_* convention).
    """
    Beta = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            mx = (m + 1) - (M + 1) / 2
            ny = (n + 1) - (N + 1) / 2
            Beta[m, n] = mx * dphase_x + ny * dphase_y
    Beta = np.deg2rad(Beta)
    xm = (np.arange(M) - (M - 1) / 2) * DX
    yn = (np.arange(N) - (N - 1) / 2) * DY

    sin_th = np.sin(TH)
    cos_ph = np.cos(PH)
    sin_ph = np.sin(PH)
    AF = np.zeros_like(TH, dtype=np.complex128)
    for m in range(M):
        for n in range(N):
            phase = K * (xm[m] * sin_th * cos_ph + yn[n] * sin_th * sin_ph) + Beta[m, n]
            AF += np.exp(1j * phase)

    P = elem_lin * np.abs(AF) ** 2
    G_dB = 10 * np.log10(P + np.finfo(float).eps)
    G_dB -= G_dB.max()
    return G_dB


def main():
    if not ELEM_NPZ.exists():
        raise FileNotFoundError(f"Element pattern cache missing: {ELEM_NPZ.name}")
    if not SOURCE_H5.exists():
        raise FileNotFoundError(f"Source HDF5 missing: {SOURCE_H5}")
    if not SPLITS.exists():
        raise FileNotFoundError(f"Split file missing: {SPLITS}")

    sp = np.load(SPLITS)
    test_idx = np.sort(sp["test"].astype(np.int64))
    print(f"Generating analytical matlab_{ARRAY_M}x{ARRAY_N} for "
          f"{len(test_idx)} test betas", flush=True)

    with h5py.File(SOURCE_H5, "r") as f:
        meta = f["metadata"][test_idx].astype(np.float64)
    beta = meta[:, :2]   # (n_test, 2): dpx, dpy

    elem = np.load(ELEM_NPZ)
    elem_dB = elem["elem_dB"].astype(np.float64)
    elem_lin = 10 ** (elem_dB / 10)
    print(f"Element pattern: shape={elem_dB.shape}, range=[{elem_dB.min():.2f}, "
          f"{elem_dB.max():.2f}] dB", flush=True)

    phi_deg = np.arange(-179.5, 180, 1.0)
    theta_deg = np.arange(0, 181, 1.0)
    PHI, THETA = np.meshgrid(phi_deg, theta_deg)
    TH = np.deg2rad(THETA)
    PH = np.deg2rad(PHI)

    arr = np.zeros((len(test_idx), N_THETA, N_PHI), dtype=np.float16)
    t0 = time.time()
    for i in tqdm(range(len(test_idx)), desc=f"matlab_{ARRAY_M}x{ARRAY_N}"):
        G = simulate_array(ARRAY_M, ARRAY_N, beta[i, 0], beta[i, 1],
                           elem_lin, TH, PH)
        arr[i] = G.astype(np.float16)
    print(f"Simulated {len(test_idx)} patterns in {time.time()-t0:.1f}s",
          flush=True)
    print(f"  G range: [{float(arr.min()):.2f}, {float(arr.max()):.2f}] dB "
          f"(per-sample max should be 0 -> max across all = {float(arr.reshape(len(test_idx), -1).max(axis=1).mean()):.4f})",
          flush=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, arr=arr, test_idx=test_idx, beta=beta)
    size_mb = OUT_NPZ.stat().st_size / 1e6
    print(f"Saved {OUT_NPZ.name} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
