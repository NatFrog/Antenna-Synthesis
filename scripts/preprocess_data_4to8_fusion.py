"""
Preprocess the 4-to-8 fusion dataset into a single HDF5 file.

Inputs (all four perfectly index-paired at file+column level):
    datasets_4x4_from_8x8/             -> matlab_4x4
    datasets_4x4_hfss_pred_from_8x8/   -> hfss_pred_4x4
    datasets_8x8_matlab/datasets_8x8/  -> matlab_8x8
    datasets_8x8_hfss/datasets_8x8_hfss/ -> hfss_8x8  (training target)

Output:
    processed/antenna_data_4to8.h5       (4 pattern datasets + metadata + grids)
    processed/norm_stats_4x4_from_8x8.npz (per-pixel mean/std from matlab_4x4 train)
    processed/norm_stats_8x8.npz         (per-pixel mean/std from matlab_8x8 train)
    processed/split_indices_4to8.npz     (1600 train / 200 val / 200 test, seed=42)

The 4-to-8 dataset has 40 source CSVs * 50 samples = 2000 samples (vs 5000 for
the 2-to-4 case). This is a fresh per-pipeline split (no existing 8x8 split to
reuse).

Usage:
    python -m scripts.preprocess_data_4to8_fusion
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
from tqdm import tqdm

from src.config import N_THETA, N_PHI, PROCESSED_DIR
from src.data.loader import load_single_csv, get_file_path

# ── Pipeline-specific constants ────────────────────────────────────────────
N_FILES_4TO8 = 40
N_CONFIGS_PER_FILE = 50
N_TOTAL = N_FILES_4TO8 * N_CONFIGS_PER_FILE      # 2000

# Split sizes (80/10/10 of 2000)
N_TRAIN = 1600
N_VAL = 200
N_TEST = 200
SPLIT_SEED = 42

# ── Source folders ─────────────────────────────────────────────────────────
M4X4_DIR = PROJECT_ROOT / "datasets_4x4_from_8x8"
HP4X4_DIR = PROJECT_ROOT / "datasets_4x4_hfss_pred_from_8x8"
M8X8_DIR = PROJECT_ROOT / "datasets_8x8_matlab" / "datasets_8x8"
H8X8_DIR = PROJECT_ROOT / "datasets_8x8_hfss" / "datasets_8x8_hfss"

# ── Output artefacts ───────────────────────────────────────────────────────
HDF5_PATH = PROCESSED_DIR / "antenna_data_4to8.h5"
NORM_4X4_OUT = PROCESSED_DIR / "norm_stats_4x4_from_8x8.npz"
NORM_8X8_OUT = PROCESSED_DIR / "norm_stats_8x8.npz"
SPLIT_OUT = PROCESSED_DIR / "split_indices_4to8.npz"


def _check_sources():
    missing = []
    for p in (M4X4_DIR, HP4X4_DIR, M8X8_DIR, H8X8_DIR):
        if not p.is_dir():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Source folder(s) missing:\n  - " + "\n  - ".join(missing))


def preprocess_all():
    _check_sources()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Creating 4-to-8 fusion HDF5 -> {HDF5_PATH}")
    print(f"  Expected samples: {N_TOTAL}  ({N_FILES_4TO8} files x {N_CONFIGS_PER_FILE})")
    print(f"  Pattern shape:    ({N_THETA}, {N_PHI})")

    with h5py.File(str(HDF5_PATH), "w") as f:
        def mkpat(name):
            return f.create_dataset(
                name, shape=(N_TOTAL, N_THETA, N_PHI), dtype=np.float16,
                chunks=(1, N_THETA, N_PHI), compression="gzip", compression_opts=4,
            )

        m4 = mkpat("matlab_4x4")
        hp4 = mkpat("hfss_pred_4x4")
        m8 = mkpat("matlab_8x8")
        h8 = mkpat("hfss_8x8")

        # 6-column metadata mirrors the 2-to-4 fusion HDF5:
        # [dphase_x, dphase_y, matlab8_phi_peak, matlab8_theta_peak,
        #  hfss8_phi_peak, hfss8_theta_peak] -- taken from the 8x8 CSVs.
        meta = f.create_dataset("metadata", shape=(N_TOTAL, 6), dtype=np.float64)
        theta_grid = f.create_dataset("theta_grid", shape=(N_THETA,), dtype=np.float32)
        phi_grid = f.create_dataset("phi_grid", shape=(N_PHI,), dtype=np.float32)

        grids_written = False
        for file_idx in tqdm(range(1, N_FILES_4TO8 + 1), desc="Files"):
            p_m4 = get_file_path(M4X4_DIR, file_idx)
            p_hp4 = get_file_path(HP4X4_DIR, file_idx)
            p_m8 = get_file_path(M8X8_DIR, file_idx)
            p_h8 = get_file_path(H8X8_DIR, file_idx)

            missing = [p for p in (p_m4, p_hp4, p_m8, p_h8) if not p.exists()]
            if missing:
                print(f"  WARNING: missing at file_idx={file_idx}: {missing}")
                continue

            d_m4 = load_single_csv(p_m4, file_index=file_idx)
            d_hp4 = load_single_csv(p_hp4, file_index=file_idx)
            d_m8 = load_single_csv(p_m8, file_index=file_idx)
            d_h8 = load_single_csv(p_h8, file_index=file_idx)

            if not grids_written:
                theta_grid[:] = d_m8["theta_grid"]
                phi_grid[:] = d_m8["phi_grid"]
                grids_written = True

            cols = d_m8["config_cols"]
            for ci, col in enumerate(cols):
                gi = (file_idx - 1) * N_CONFIGS_PER_FILE + ci
                m4[gi] = d_m4["patterns"][ci].astype(np.float16)
                hp4[gi] = d_hp4["patterns"][ci].astype(np.float16)
                m8[gi] = d_m8["patterns"][ci].astype(np.float16)
                h8[gi] = d_h8["patterns"][ci].astype(np.float16)

                mm8 = d_m8["metadata"][col]
                hh8 = d_h8["metadata"][col]
                meta[gi] = [
                    mm8["dphase_x"], mm8["dphase_y"],
                    mm8["phi_peak"], mm8["theta_peak"],
                    hh8["phi_peak"], hh8["theta_peak"],
                ]

    size_gb = HDF5_PATH.stat().st_size / (1024 ** 3)
    print(f"\nWritten: {HDF5_PATH}   ({size_gb:.2f} GB)")


def make_split():
    """Random 80/10/10 split over 2000 indices, seed=42."""
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(N_TOTAL)
    train = np.sort(perm[:N_TRAIN])
    val = np.sort(perm[N_TRAIN:N_TRAIN + N_VAL])
    test = np.sort(perm[N_TRAIN + N_VAL:])
    np.savez(SPLIT_OUT, train=train, val=val, test=test)
    print(f"Wrote split -> {SPLIT_OUT.name} "
          f"({len(train)} train / {len(val)} val / {len(test)} test, seed={SPLIT_SEED})")
    return train, val, test


def compute_norm_stats(train_idx: np.ndarray, h5_dataset_name: str, out_path: Path,
                       label: str):
    """Per-pixel (mean, std) over the training split for one HDF5 channel."""
    print(f"\nComputing {label} norm stats from training split -> {out_path.name}")
    with h5py.File(str(HDF5_PATH), "r") as f:
        sum_ = np.zeros((N_THETA, N_PHI), dtype=np.float64)
        sumsq = np.zeros((N_THETA, N_PHI), dtype=np.float64)
        for gi in tqdm(train_idx, desc="Accumulating"):
            x = f[h5_dataset_name][int(gi)].astype(np.float64)
            sum_ += x
            sumsq += x * x

    n = len(train_idx)
    mean = (sum_ / n).astype(np.float32)
    var = (sumsq / n) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    np.savez(out_path, mean=mean, std=std)
    print(f"  mean range: [{mean.min():.3f}, {mean.max():.3f}] dB")
    print(f"  std  range: [{std.min():.3f}, {std.max():.3f}] dB")


def verify():
    """Spot-check a few samples against their source CSVs."""
    print("\n-- Verification --")
    rng = np.random.default_rng(42)
    check = rng.choice(N_TOTAL, size=5, replace=False)

    with h5py.File(str(HDF5_PATH), "r") as f:
        for gi in check:
            file_idx = gi // N_CONFIGS_PER_FILE + 1
            ci = gi % N_CONFIGS_PER_FILE
            srcs = {
                "matlab_4x4":     get_file_path(M4X4_DIR, file_idx),
                "hfss_pred_4x4":  get_file_path(HP4X4_DIR, file_idx),
                "matlab_8x8":     get_file_path(M8X8_DIR, file_idx),
                "hfss_8x8":       get_file_path(H8X8_DIR, file_idx),
            }
            ti, pi = rng.integers(0, N_THETA), rng.integers(0, N_PHI)
            msgs = []
            for name, p in srcs.items():
                d = load_single_csv(p, file_index=file_idx)
                csv_val = float(d["patterns"][ci, ti, pi])
                h5_val = float(f[name][gi, ti, pi])
                msgs.append(f"{name}: |diff|={abs(csv_val - h5_val):.4f}")
            print(f"  gi={gi} (file {file_idx}, col {ci}) point ({ti},{pi})  " +
                  "  ".join(msgs))


if __name__ == "__main__":
    preprocess_all()
    verify()
    train, val, test = make_split()
    compute_norm_stats(train, "matlab_4x4", NORM_4X4_OUT, "4x4 (from 8x8)")
    compute_norm_stats(train, "matlab_8x8", NORM_8X8_OUT, "8x8")
    print("\nDone.")