"""
Preprocess the 2-to-4 fusion dataset into a single HDF5 file.

Inputs (all four perfectly index-paired at file+column level):
    datasets_2x2_from_4x4/             -> matlab_2x2
    datasets_2x2_hfss_pred_from_4x4/   -> hfss_pred_2x2
    datasets_4x4_matlab/datasets_4x4/  -> matlab_4x4
    datasets_4x4_hfss/datasets_4x4_hfss/ -> hfss_4x4  (training target)

Output:
    processed/antenna_data_2to4.h5        (4 pattern datasets + metadata + grids)

Normalisation:
    No new stats computed here. At train/eval time:
      2x2 channels use processed/norm_stats_2x2.npz
      4x4 channel + target use processed/norm_stats.npz

Split:
    Reuses processed/split_indices.npz (the 4x4 split) verbatim. Indices 0-4999
    map 1-to-1 across all four sources because every sample shares
    (file_idx, config_idx).

Usage:
    python -m scripts.preprocess_data_2to4_fusion
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
M2X2_DIR = PROJECT_ROOT / "datasets_2x2_from_4x4"
H2X2_PRED_DIR = PROJECT_ROOT / "datasets_2x2_hfss_pred_from_4x4"
M4X4_DIR = PROJECT_ROOT / "datasets_4x4_matlab" / "datasets_4x4"
H4X4_DIR = PROJECT_ROOT / "datasets_4x4_hfss" / "datasets_4x4_hfss"

# ── Output artefacts ───────────────────────────────────────────────────────
HDF5_PATH = PROCESSED_DIR / "antenna_data_2to4.h5"

# Existing split that we reuse for head-to-head comparability with the 4x4 cGAN.
SPLIT_4X4 = PROCESSED_DIR / "split_indices.npz"

# Required norm stats (must exist; we don't compute new ones)
NORM_STATS_2X2 = PROCESSED_DIR / "norm_stats_2x2.npz"
NORM_STATS_4X4 = PROCESSED_DIR / "norm_stats.npz"


def _check_sources():
    missing = []
    for p in (M2X2_DIR, H2X2_PRED_DIR, M4X4_DIR, H4X4_DIR):
        if not p.is_dir():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Source folder(s) missing:\n  - " + "\n  - ".join(missing))
    for p in (SPLIT_4X4, NORM_STATS_2X2, NORM_STATS_4X4):
        if not p.exists():
            raise FileNotFoundError(f"Required artefact missing: {p}")


def preprocess_all():
    _check_sources()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Creating fusion HDF5 -> {HDF5_PATH}")
    print(f"  Expected samples: {N_TOTAL_CONFIGS}")
    print(f"  Pattern shape:    ({N_THETA}, {N_PHI})")

    with h5py.File(str(HDF5_PATH), "w") as f:
        def mkpat(name):
            return f.create_dataset(
                name, shape=(N_TOTAL_CONFIGS, N_THETA, N_PHI), dtype=np.float16,
                chunks=(1, N_THETA, N_PHI), compression="gzip", compression_opts=4,
            )

        m2 = mkpat("matlab_2x2")
        hp2 = mkpat("hfss_pred_2x2")
        m4 = mkpat("matlab_4x4")
        h4 = mkpat("hfss_4x4")

        # 6-column metadata mirrors scripts/preprocess_data.py for drop-in consumption:
        # [dphase_x, dphase_y, matlab_phi_peak, matlab_theta_peak,
        #  hfss_phi_peak,  hfss_theta_peak] — taken from the 4x4 CSVs.
        meta = f.create_dataset("metadata", shape=(N_TOTAL_CONFIGS, 6), dtype=np.float64)
        theta_grid = f.create_dataset("theta_grid", shape=(N_THETA,), dtype=np.float32)
        phi_grid = f.create_dataset("phi_grid", shape=(N_PHI,), dtype=np.float32)

        grids_written = False
        for file_idx in tqdm(range(1, N_FILES + 1), desc="Files"):
            p_m2 = get_file_path(M2X2_DIR, file_idx)
            p_hp2 = get_file_path(H2X2_PRED_DIR, file_idx)
            p_m4 = get_file_path(M4X4_DIR, file_idx)
            p_h4 = get_file_path(H4X4_DIR, file_idx)

            missing = [p for p in (p_m2, p_hp2, p_m4, p_h4) if not p.exists()]
            if missing:
                print(f"  WARNING: missing at file_idx={file_idx}: {missing}")
                continue

            d_m2 = load_single_csv(p_m2, file_index=file_idx)
            d_hp2 = load_single_csv(p_hp2, file_index=file_idx)
            d_m4 = load_single_csv(p_m4, file_index=file_idx)
            d_h4 = load_single_csv(p_h4, file_index=file_idx)

            if not grids_written:
                theta_grid[:] = d_m4["theta_grid"]
                phi_grid[:] = d_m4["phi_grid"]
                grids_written = True

            cols = d_m4["config_cols"]                     # use 4x4 convention as canonical
            for ci, col in enumerate(cols):
                gi = (file_idx - 1) * N_CONFIGS_PER_FILE + ci
                m2[gi] = d_m2["patterns"][ci].astype(np.float16)
                hp2[gi] = d_hp2["patterns"][ci].astype(np.float16)
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


def verify():
    """Spot-check a few samples against their source CSVs."""
    print("\n-- Verification --")
    rng = np.random.default_rng(42)
    check = rng.choice(N_TOTAL_CONFIGS, size=5, replace=False)

    with h5py.File(str(HDF5_PATH), "r") as f:
        for gi in check:
            file_idx = gi // N_CONFIGS_PER_FILE + 1
            ci = gi % N_CONFIGS_PER_FILE

            srcs = {
                "matlab_2x2":     get_file_path(M2X2_DIR, file_idx),
                "hfss_pred_2x2":  get_file_path(H2X2_PRED_DIR, file_idx),
                "matlab_4x4":     get_file_path(M4X4_DIR, file_idx),
                "hfss_4x4":       get_file_path(H4X4_DIR, file_idx),
            }
            ti, pi = rng.integers(0, N_THETA), rng.integers(0, N_PHI)
            msgs = []
            for name, p in srcs.items():
                d = load_single_csv(p, file_index=file_idx)
                csv_val = float(d["patterns"][ci, ti, pi])
                h5_val = float(f[name][gi, ti, pi])
                diff = abs(csv_val - h5_val)
                msgs.append(f"{name}: |diff|={diff:.4f}")
            print(f"  gi={gi} (file {file_idx}, col idx {ci}) point ({ti},{pi})  " +
                  "  ".join(msgs))


def link_split():
    """Re-emit the 4x4 split alongside the fusion HDF5 with a matching filename,
    so downstream scripts don't have to special-case paths."""
    s = np.load(SPLIT_4X4)
    out = PROCESSED_DIR / "split_indices_2to4.npz"
    np.savez(out, train=s["train"], val=s["val"], test=s["test"])
    print(f"Reused 4x4 splits -> {out.name} "
          f"({len(s['train'])} train / {len(s['val'])} val / {len(s['test'])} test)")


if __name__ == "__main__":
    preprocess_all()
    verify()
    link_split()
    print("\nDone.")
