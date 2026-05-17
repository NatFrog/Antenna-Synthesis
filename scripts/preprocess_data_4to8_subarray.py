"""
Build the 4-to-8 sub-array synthesis HDF5 dataset.

Consumes:
  - processed/antenna_data_4to8.h5 (must exist locally; built by
    scripts/preprocess_data_4to8_fusion.py on feature/cgan-4to8-fusion).
    We copy `matlab_4x4`, `hfss_8x8`, `metadata`, `theta_grid`, `phi_grid`
    from it -- those channels are unchanged in this pipeline.
  - datasets_4x4_hfss_pred_from_8x8_no_m4/   (40 CSVs, 50 samples each = 2000)
    The new-no-m4-cGAN predictions of HFSS 4x4 (this is the only channel that
    differs from the upstream 4-to-8 fusion h5).

Produces:
  - processed/antenna_data_4to8_subarray.h5
       /matlab_4x4   (2000, 181, 360) float16 -- analytical 4x4 from 8x8
       /hfss_pred_4x4 (2000, 181, 360) float16 -- no-m4 cGAN prediction (NEW)
       /hfss_8x8     (2000, 181, 360) float16 -- ground-truth 8x8 (target)
       /metadata     (2000, 6) float64
       /theta_grid   (181,) float32
       /phi_grid     (360,) float32

The split / norm-stats artefacts (split_indices_4to8.npz,
norm_stats_4x4_from_8x8.npz, norm_stats_8x8.npz) are reused as-is from the
upstream fusion pipeline (already brought into processed/ for this branch),
so the train/val/test split and per-pixel statistics match the upstream model
1-to-1 -- enabling a clean head-to-head comparison.

Usage:
    python -m scripts.preprocess_data_4to8_subarray
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
HP4X4_DIR = PROJECT_ROOT / "datasets_4x4_hfss_pred_from_8x8_no_m4"
SOURCE_H5 = PROCESSED_DIR / "antenna_data_4to8.h5"

# ── Output ────────────────────────────────────────────────────────────────
OUT_H5 = PROCESSED_DIR / "antenna_data_4to8_subarray.h5"

# ── Pipeline-specific constants ────────────────────────────────────────────
N_FILES = 40
N_CONFIGS_PER_FILE = 50
N_TOTAL = N_FILES * N_CONFIGS_PER_FILE   # 2000


def read_pattern_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (patterns, dphase_x, dphase_y) where
    patterns has shape (num_samples, N_THETA, N_PHI) in dB.
    """
    df = pd.read_csv(path, header=None, low_memory=False)
    num_samples = df.shape[1] - 2
    dpx = df.iloc[1, 2:].astype(np.float64).to_numpy()
    dpy = df.iloc[2, 2:].astype(np.float64).to_numpy()
    flat = df.iloc[5:, 2:].astype(np.float32).to_numpy()  # (NTH*NPH, num)
    pat = np.transpose(flat.reshape(N_PHI, N_THETA, num_samples), axes=(2, 1, 0))
    return pat, dpx, dpy


def main():
    if not SOURCE_H5.exists():
        raise FileNotFoundError(
            f"Source HDF5 missing: {SOURCE_H5}. Build it via "
            "scripts/preprocess_data_4to8_fusion.py on feature/cgan-4to8-fusion.")
    if not HP4X4_DIR.is_dir():
        raise FileNotFoundError(
            f"hfss_pred_4x4 (no-m4) folder missing: {HP4X4_DIR}. Run "
            "scripts/predict_4x4_hfss_from_8x8_no_m4.py first.")
    paths = sorted(HP4X4_DIR.glob("patterns_global_*.csv"))
    if len(paths) != N_FILES:
        raise RuntimeError(f"Expected {N_FILES} CSVs, got {len(paths)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building {OUT_H5}", flush=True)
    print(f"  Reusing matlab_4x4, hfss_8x8, metadata from {SOURCE_H5.name}", flush=True)
    print(f"  Adding new hfss_pred_4x4 from {HP4X4_DIR.name}", flush=True)

    t0 = time.time()
    with h5py.File(SOURCE_H5, "r") as src, h5py.File(OUT_H5, "w") as dst:
        # Copy the channels that are unchanged.
        for key in ("matlab_4x4", "hfss_8x8", "metadata", "theta_grid", "phi_grid"):
            if key not in src:
                raise RuntimeError(f"Source HDF5 missing dataset '{key}'")
            print(f"  copy {key} (shape={src[key].shape})...", flush=True)
            dst.create_dataset(key, data=src[key][:], dtype=src[key].dtype)

        # Build the new hfss_pred_4x4 dataset from the no-m4 prediction CSVs.
        hp = dst.create_dataset(
            "hfss_pred_4x4",
            shape=(N_TOTAL, N_THETA, N_PHI),
            dtype=np.float16,
            chunks=(1, N_THETA, N_PHI),
            compression="gzip", compression_opts=4,
        )
        # Pull out the metadata's (dphase_x, dphase_y) for sanity-check alignment.
        meta_dpx = src["metadata"][:, 0]
        meta_dpy = src["metadata"][:, 1]

        for i, p in enumerate(tqdm(paths, desc="hfss_pred_4x4 (no-m4)")):
            pat, dpx, dpy = read_pattern_csv(p)
            n = pat.shape[0]
            start = i * N_CONFIGS_PER_FILE
            stop = start + n
            # Verify alignment with the metadata's beta values
            if not (np.allclose(dpx, meta_dpx[start:stop], atol=1e-3)
                    and np.allclose(dpy, meta_dpy[start:stop], atol=1e-3)):
                raise RuntimeError(f"{p.name}: beta mismatch vs source metadata")
            hp[start:stop] = pat.astype(np.float16)

    elapsed = time.time() - t0

    # Quick verification
    with h5py.File(OUT_H5, "r") as f:
        for k in ("matlab_4x4", "hfss_pred_4x4", "hfss_8x8"):
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
