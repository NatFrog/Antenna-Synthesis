"""
Preprocess the actual 2-to-4 fusion dataset (UW sub-array sources) into a
single HDF5 file.

Variant of preprocess_data_2to4_fusion.py that uses the UW sub-array datasets
in place of the Python-derived 2x2 patterns and the cGAN-predicted HFSS 2x2:

    UW_datasets_2x2_to_4x4_subArray_matlab/datasets_2x2_subArray_matlab/  -> matlab_2x2
    UW_datasets_2x2_to_4x4_subArray_hfss/datasets_2x2_subArray_hfss/      -> hfss_2x2  (real, not predicted)
    datasets_4x4_matlab/datasets_4x4/                                     -> matlab_4x4
    datasets_4x4_hfss/datasets_4x4_hfss/                                  -> hfss_4x4 (training target)

The UW sources share (file_idx, sample_col) and the same (dphase_x, dphase_y)
sequence as datasets_4x4_matlab/, so the same split (split_indices_2to4.npz)
applies and head-to-head comparison with the synthetic fusion model is clean.

Output:
    processed/antenna_data_2to4_uw.h5     (4 pattern datasets + metadata + grids)
    processed/norm_stats_2x2_uw.npz       (per-pixel mean/std from matlab_2x2 train split)
    processed/split_indices_2to4_uw.npz   (copy of split_indices_2to4.npz for symmetry)

The 4x4 channel + target reuse the existing processed/norm_stats.npz
(unchanged from the 4x4 cGAN), so only the 2x2-channel norm is recomputed.

Usage:
    python -m scripts.preprocess_data_2to4_fusion_uw
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
from tqdm import tqdm

from src.config import (
    PROCESSED_DIR,
    N_FILES, N_CONFIGS_PER_FILE, N_TOTAL_CONFIGS,
    N_THETA, N_PHI,
)
from src.data.loader import load_single_csv, get_file_path

# ── Source folders ─────────────────────────────────────────────────────────
M2X2_DIR = PROJECT_ROOT / "UW_datasets_2x2_to_4x4_subArray_matlab" / "datasets_2x2_subArray_matlab"
H2X2_DIR = PROJECT_ROOT / "UW_datasets_2x2_to_4x4_subArray_hfss" / "datasets_2x2_subArray_hfss"
M4X4_DIR = PROJECT_ROOT / "datasets_4x4_matlab" / "datasets_4x4"
H4X4_DIR = PROJECT_ROOT / "datasets_4x4_hfss" / "datasets_4x4_hfss"

# ── Output artefacts ───────────────────────────────────────────────────────
HDF5_PATH = PROCESSED_DIR / "antenna_data_2to4_uw.h5"
NORM_2X2_OUT = PROCESSED_DIR / "norm_stats_2x2_uw.npz"
SPLIT_OUT = PROCESSED_DIR / "split_indices_2to4_uw.npz"

# Existing split that we reuse (4000 train / ~495 val / 505 test); this is the
# same split the original 2-to-4 fusion model used.
SPLIT_SRC = PROCESSED_DIR / "split_indices_2to4.npz"


def _check_sources():
    missing = []
    for p in (M2X2_DIR, H2X2_DIR, M4X4_DIR, H4X4_DIR):
        if not p.is_dir():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Source folder(s) missing:\n  - " + "\n  - ".join(missing))
    if not SPLIT_SRC.exists():
        raise FileNotFoundError(f"Required split file missing: {SPLIT_SRC}")


def preprocess_all():
    _check_sources()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Creating actual fusion HDF5 -> {HDF5_PATH}")
    print(f"  Expected samples: {N_TOTAL_CONFIGS}")
    print(f"  Pattern shape:    ({N_THETA}, {N_PHI})")

    with h5py.File(str(HDF5_PATH), "w") as f:
        def mkpat(name):
            return f.create_dataset(
                name, shape=(N_TOTAL_CONFIGS, N_THETA, N_PHI), dtype=np.float16,
                chunks=(1, N_THETA, N_PHI), compression="gzip", compression_opts=4,
            )

        m2 = mkpat("matlab_2x2")
        h2 = mkpat("hfss_2x2")
        m4 = mkpat("matlab_4x4")
        h4 = mkpat("hfss_4x4")

        meta = f.create_dataset("metadata", shape=(N_TOTAL_CONFIGS, 6), dtype=np.float64)
        theta_grid = f.create_dataset("theta_grid", shape=(N_THETA,), dtype=np.float32)
        phi_grid = f.create_dataset("phi_grid", shape=(N_PHI,), dtype=np.float32)

        grids_written = False
        for file_idx in tqdm(range(1, N_FILES + 1), desc="Files"):
            p_m2 = get_file_path(M2X2_DIR, file_idx)
            p_h2 = get_file_path(H2X2_DIR, file_idx)
            p_m4 = get_file_path(M4X4_DIR, file_idx)
            p_h4 = get_file_path(H4X4_DIR, file_idx)

            missing = [p for p in (p_m2, p_h2, p_m4, p_h4) if not p.exists()]
            if missing:
                print(f"  WARNING: missing at file_idx={file_idx}: {missing}")
                continue

            d_m2 = load_single_csv(p_m2, file_index=file_idx)
            d_h2 = load_single_csv(p_h2, file_index=file_idx)
            d_m4 = load_single_csv(p_m4, file_index=file_idx)
            d_h4 = load_single_csv(p_h4, file_index=file_idx)

            if not grids_written:
                theta_grid[:] = d_m4["theta_grid"]
                phi_grid[:] = d_m4["phi_grid"]
                grids_written = True

            cols = d_m4["config_cols"]
            for ci, col in enumerate(cols):
                gi = (file_idx - 1) * N_CONFIGS_PER_FILE + ci
                m2[gi] = d_m2["patterns"][ci].astype(np.float16)
                h2[gi] = d_h2["patterns"][ci].astype(np.float16)
                m4[gi] = d_m4["patterns"][ci].astype(np.float16)
                h4[gi] = d_h4["patterns"][ci].astype(np.float16)

                mm4 = d_m4["metadata"][col]
                hh4 = d_h4["metadata"][col]
                meta[gi] = [
                    mm4["dphase_x"], mm4["dphase_y"],
                    mm4["phi_peak"], mm4["theta_peak"],
                    hh4["phi_peak"], hh4["theta_peak"],
                ]

    size_gb = HDF5_PATH.stat().st_size / (1024 ** 3)
    print(f"\nWritten: {HDF5_PATH}   ({size_gb:.2f} GB)")


def compute_norm_stats():
    """Compute per-pixel (mean, std) of matlab_2x2 over the training split."""
    print(f"\nComputing actual-variant 2x2 norm stats from training split -> {NORM_2X2_OUT.name}")
    s = np.load(SPLIT_SRC)
    train_idx = np.sort(s["train"].astype(np.int64))
    print(f"  {len(train_idx)} training samples")

    with h5py.File(str(HDF5_PATH), "r") as f:
        sum_ = np.zeros((N_THETA, N_PHI), dtype=np.float64)
        sumsq = np.zeros((N_THETA, N_PHI), dtype=np.float64)
        for gi in tqdm(train_idx, desc="Accumulating"):
            x = f["matlab_2x2"][int(gi)].astype(np.float64)
            sum_ += x
            sumsq += x * x

    n = len(train_idx)
    mean = (sum_ / n).astype(np.float32)
    var = (sumsq / n) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)

    np.savez(NORM_2X2_OUT, mean=mean, std=std)
    print(f"  mean range: [{mean.min():.3f}, {mean.max():.3f}] dB")
    print(f"  std  range: [{std.min():.3f}, {std.max():.3f}] dB")
    print(f"  saved to {NORM_2X2_OUT}")


def link_split():
    s = np.load(SPLIT_SRC)
    np.savez(SPLIT_OUT, train=s["train"], val=s["val"], test=s["test"])
    print(f"Reused split -> {SPLIT_OUT.name} "
          f"({len(s['train'])} train / {len(s['val'])} val / {len(s['test'])} test)")


def verify():
    print("\n-- Verification --")
    rng = np.random.default_rng(42)
    check = rng.choice(N_TOTAL_CONFIGS, size=5, replace=False)

    with h5py.File(str(HDF5_PATH), "r") as f:
        for gi in check:
            file_idx = gi // N_CONFIGS_PER_FILE + 1
            ci = gi % N_CONFIGS_PER_FILE
            srcs = {
                "matlab_2x2":  get_file_path(M2X2_DIR, file_idx),
                "hfss_2x2":    get_file_path(H2X2_DIR, file_idx),
                "matlab_4x4":  get_file_path(M4X4_DIR, file_idx),
                "hfss_4x4":    get_file_path(H4X4_DIR, file_idx),
            }
            ti, pi = rng.integers(0, N_THETA), rng.integers(0, N_PHI)
            msgs = []
            for name, p in srcs.items():
                d = load_single_csv(p, file_index=file_idx)
                csv_val = float(d["patterns"][ci, ti, pi])
                h5_val = float(f[name][gi, ti, pi])
                msgs.append(f"{name}: |diff|={abs(csv_val - h5_val):.4f}")
            print(f"  gi={gi} (file {file_idx}, col idx {ci}) point ({ti},{pi})  " +
                  "  ".join(msgs))


if __name__ == "__main__":
    preprocess_all()
    verify()
    compute_norm_stats()
    link_split()
    print("\nDone.")
