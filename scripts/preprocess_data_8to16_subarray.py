"""
Build the 8-to-16 sub-array synthesis HDF5 dataset.

Consumes:
  - processed/antenna_data_4to8.h5 (built by scripts/preprocess_data_4to8_fusion.py).
    We copy `metadata`, `theta_grid`, `phi_grid` from it -- the per-row
    (dphase_x, dphase_y) ordering of this output file is locked 1-to-1 to the
    4-to-8 fusion file so downstream code can reuse split_indices_4to8.npz to
    address rows in this file too.
  - datasets_16x16_hfss/datasets_16x16_hfss/ (40 CSVs * 50 samples = 2000)
    Real HFSS sims at the 16x16 scale -- the eventual ground truth used for
    val_16x16 and test_16x16 in scripts/train_residual_multiscale.py.
  - datasets_8x8_hfss_pred_from_16x16/ (40 CSVs * 50 samples = 2000)
    Cascade-predicted HFSS 8x8 patterns at the 16x16 betas. Produced by
    scripts/predict_8x8_hfss_from_16x16.py running the trained 8x8 cGAN over
    analytical matlab_8x8 patterns derived at the 16x16 betas. Used as the
    ch4 ANCHOR at N=16 inference time in the 5-channel residual model.

Produces:
  - processed/antenna_data_8to16_subarray.h5
       /hfss_16x16     (2000, 181, 360) float16 -- real HFSS at the 16x16 scale
       /hfss_pred_8x8  (2000, 181, 360) float16 -- cGAN-cascade prediction
                                                     at the next-smaller scale
                                                     (ch4 anchor at N=16)
       /metadata       (2000, 6) float64
       /theta_grid     (181,) float32
       /phi_grid       (360,) float32

Alignment is verified per-CSV by comparing the (dphase_x, dphase_y) headers in
each source CSV against the matching slice of the 4to8 metadata. Any drift
raises an error rather than silently writing misaligned data.

Usage:
    python -m scripts.preprocess_data_8to16_subarray
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from src.config import N_THETA, N_PHI, PROCESSED_DIR

# ── Source folders / files ─────────────────────────────────────────────────
H16X16_DIR = PROJECT_ROOT / "datasets_16x16_hfss" / "datasets_16x16_hfss"
HP8X8_DIR = PROJECT_ROOT / "datasets_8x8_hfss_pred_from_16x16"
SOURCE_H5 = PROCESSED_DIR / "antenna_data_4to8.h5"

# ── Output ────────────────────────────────────────────────────────────────
OUT_H5 = PROCESSED_DIR / "antenna_data_8to16_subarray.h5"

# ── Pipeline-specific constants (must match the 4-to-8 fusion file) ───────
N_FILES = 40
N_CONFIGS_PER_FILE = 50
N_TOTAL = N_FILES * N_CONFIGS_PER_FILE   # 2000


def read_pattern_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (patterns, dphase_x, dphase_y) where
    patterns has shape (num_samples, N_THETA, N_PHI) in dB.

    Mirrors the project's canonical CSV layout:
        row 0 : headers
        row 1 : dphase_x_deg
        row 2 : dphase_y_deg
        row 3 : phi_peak_deg     (ignored here)
        row 4 : theta_peak_deg   (ignored here)
        rows 5+: theta_deg, phi_deg, sample_1, sample_2, ...
    """
    df = pd.read_csv(path, header=None, low_memory=False)
    num_samples = df.shape[1] - 2
    dpx = df.iloc[1, 2:].astype(np.float64).to_numpy()
    dpy = df.iloc[2, 2:].astype(np.float64).to_numpy()
    flat = df.iloc[5:, 2:].astype(np.float32).to_numpy()  # (NTH*NPH, num)
    pat = np.transpose(flat.reshape(N_PHI, N_THETA, num_samples), axes=(2, 1, 0))
    return pat, dpx, dpy


def _check_sources():
    missing = []
    if not SOURCE_H5.exists():
        missing.append(
            f"{SOURCE_H5}  (build via scripts/preprocess_data_4to8_fusion.py)"
        )
    if not H16X16_DIR.is_dir():
        missing.append(
            f"{H16X16_DIR}  (raw HFSS 16x16 simulation CSVs)"
        )
    if not HP8X8_DIR.is_dir():
        missing.append(
            f"{HP8X8_DIR}  (run scripts/predict_8x8_hfss_from_16x16.py first)"
        )
    if missing:
        raise FileNotFoundError(
            "Source artefact(s) missing:\n  - " + "\n  - ".join(missing)
        )

    h16_paths = sorted(H16X16_DIR.glob("patterns_global_*.csv"))
    hp8_paths = sorted(HP8X8_DIR.glob("patterns_global_*.csv"))
    if len(h16_paths) != N_FILES:
        raise RuntimeError(f"Expected {N_FILES} 16x16 HFSS CSVs in "
                           f"{H16X16_DIR}, got {len(h16_paths)}")
    if len(hp8_paths) != N_FILES:
        raise RuntimeError(f"Expected {N_FILES} hfss_pred_8x8 CSVs in "
                           f"{HP8X8_DIR}, got {len(hp8_paths)}")
    return h16_paths, hp8_paths


def main():
    h16_paths, hp8_paths = _check_sources()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building {OUT_H5}", flush=True)
    print(f"  Reusing metadata, theta_grid, phi_grid from {SOURCE_H5.name}",
          flush=True)
    print(f"  hfss_16x16    from {H16X16_DIR.relative_to(PROJECT_ROOT)}",
          flush=True)
    print(f"  hfss_pred_8x8 from {HP8X8_DIR.relative_to(PROJECT_ROOT)}",
          flush=True)

    t0 = time.time()
    with h5py.File(SOURCE_H5, "r") as src, h5py.File(OUT_H5, "w") as dst:
        # ── Copy unchanged channels from the 4to8 fusion file ───────────────
        for key in ("metadata", "theta_grid", "phi_grid"):
            if key not in src:
                raise RuntimeError(f"Source HDF5 missing dataset '{key}'")
            print(f"  copy {key} (shape={src[key].shape})...", flush=True)
            dst.create_dataset(key, data=src[key][:], dtype=src[key].dtype)

        # ── Allocate the two new pattern datasets ───────────────────────────
        h16 = dst.create_dataset(
            "hfss_16x16",
            shape=(N_TOTAL, N_THETA, N_PHI),
            dtype=np.float16,
            chunks=(1, N_THETA, N_PHI),
            compression="gzip", compression_opts=4,
        )
        hp8 = dst.create_dataset(
            "hfss_pred_8x8",
            shape=(N_TOTAL, N_THETA, N_PHI),
            dtype=np.float16,
            chunks=(1, N_THETA, N_PHI),
            compression="gzip", compression_opts=4,
        )

        # Pull (dphase_x, dphase_y) for the global ordering; both new datasets
        # must agree with these row-by-row.
        meta_dpx = src["metadata"][:, 0]
        meta_dpy = src["metadata"][:, 1]

        # ── Fill hfss_16x16 ────────────────────────────────────────────────
        for i, p in enumerate(tqdm(h16_paths, desc="hfss_16x16")):
            pat, dpx, dpy = read_pattern_csv(p)
            n = pat.shape[0]
            start = i * N_CONFIGS_PER_FILE
            stop = start + n
            if not (np.allclose(dpx, meta_dpx[start:stop], atol=1e-3)
                    and np.allclose(dpy, meta_dpy[start:stop], atol=1e-3)):
                raise RuntimeError(
                    f"{p.name}: 16x16 HFSS beta mismatch vs 4to8 metadata "
                    f"at rows {start}:{stop}"
                )
            h16[start:stop] = pat.astype(np.float16)

        # ── Fill hfss_pred_8x8 ─────────────────────────────────────────────
        for i, p in enumerate(tqdm(hp8_paths, desc="hfss_pred_8x8")):
            pat, dpx, dpy = read_pattern_csv(p)
            n = pat.shape[0]
            start = i * N_CONFIGS_PER_FILE
            stop = start + n
            if not (np.allclose(dpx, meta_dpx[start:stop], atol=1e-3)
                    and np.allclose(dpy, meta_dpy[start:stop], atol=1e-3)):
                raise RuntimeError(
                    f"{p.name}: hfss_pred_8x8 beta mismatch vs 4to8 metadata "
                    f"at rows {start}:{stop}"
                )
            hp8[start:stop] = pat.astype(np.float16)

    elapsed = time.time() - t0

    # ── Quick verification ────────────────────────────────────────────────
    with h5py.File(OUT_H5, "r") as f:
        for k in ("hfss_16x16", "hfss_pred_8x8"):
            d = f[k]
            sample = d[0]
            print(f"  {k}: shape={d.shape}, dtype={d.dtype}, "
                  f"sample[0] range=[{sample.min():.2f}, {sample.max():.2f}] dB",
                  flush=True)
        print(f"  metadata: shape={f['metadata'].shape}", flush=True)

    size_mb = OUT_H5.stat().st_size / 1e6
    print(f"\nDone in {elapsed:.1f}s. {OUT_H5.name} = {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
