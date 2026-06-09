"""
Load 16×16 zero-shot evaluation bundles (HFSS truth + compositions + splits).

Used by ``eval_multihead_*_16x16`` scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.composition.phase_aware import (
    compose_scale_magnitude_only,
    load_hfss_csv,
    load_sub6_csvs,
    matlab_array_db,
    steering_key,
)
from src.config import N_PHI, N_THETA, PROCESSED_DIR, RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HF16_DIR = PROJECT_ROOT / "datasets_16x16_hfss" / "datasets_16x16_hfss"
HF16_FALLBACK = PROJECT_ROOT / "dataset_16x16"
SB6_DIR = PROJECT_ROOT / "datasets_6x6sub-block_hfss"
NPZ_4X4 = PROCESSED_DIR / "subblock_4x4_compose.npz"
EXTRAS_OLD = PROCESSED_DIR / "stage1_extras.npz"


@dataclass
class ZeroshotBundle:
    hfss: np.ndarray
    matlab: np.ndarray
    sub_block: np.ndarray
    dpx: np.ndarray
    dpy: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    matlab_mean: np.ndarray
    matlab_std: np.ndarray
    sub_mean: np.ndarray
    sub_std: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    sub6_blocks: np.ndarray
    sub6_pair_idx: np.ndarray
    fingerprint: np.ndarray | None = None
    sample_ids: np.ndarray | None = None


def _theta_phi() -> tuple[np.ndarray, np.ndarray]:
    if NPZ_4X4.exists():
        old = np.load(NPZ_4X4)
        return old["theta"].astype(np.float32), old["phi"].astype(np.float32)
    return (
        np.linspace(0, 180, N_THETA, dtype=np.float32),
        np.linspace(-179.5, 179.5, N_PHI, dtype=np.float32),
    )


def _load_hfss16() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import glob

    hf16_dir = HF16_DIR if HF16_DIR.is_dir() else HF16_FALLBACK
    files = sorted(glob.glob(str(hf16_dir / "patterns_global_*.csv")))
    if not files:
        raise FileNotFoundError(f"No 16×16 HFSS CSVs in {hf16_dir}")
    ids_parts, dpx_parts, dpy_parts, hf_parts = [], [], [], []
    for f in files:
        ids, dpx, dpy, pat = load_hfss_csv(f, 1)
        ids_parts.append(ids)
        dpx_parts.append(dpx)
        dpy_parts.append(dpy)
        hf_parts.append(pat[:, 0])
    return (
        np.concatenate(ids_parts),
        np.concatenate(dpx_parts),
        np.concatenate(dpy_parts),
        np.concatenate(hf_parts, axis=0).astype(np.float32),
    )


def load_16x16_bundle() -> ZeroshotBundle:
    """Build matched 16×16 HFSS + position-aware sub-block composition bundle."""
    theta_deg, phi_deg = _theta_phi()
    ids16, dpx16, dpy16, hf16 = _load_hfss16()
    _, dpx6, dpy6, sub6 = load_sub6_csvs(SB6_DIR)

    key6 = {steering_key(dpx6[i], dpy6[i]): i for i in range(len(dpx6))}
    matched_16, matched_6 = [], []
    for i in range(len(dpx16)):
        k = steering_key(dpx16[i], dpy16[i])
        if k in key6:
            matched_16.append(i)
            matched_6.append(key6[k])
    if not matched_16:
        raise RuntimeError("No steering overlap between 16×16 HFSS and 6×6 sub-blocks.")

    matched_16 = np.array(matched_16, dtype=np.int64)
    matched_6 = np.array(matched_6, dtype=np.int64)
    n = len(matched_16)

    hfss = hf16[matched_16]
    dpx = dpx16[matched_16].astype(np.float32)
    dpy = dpy16[matched_16].astype(np.float32)
    sample_ids = ids16[matched_16]
    sub6_blocks = sub6[matched_6]

    matlab = np.empty((n, N_THETA, N_PHI), dtype=np.float32)
    sub_block = np.empty((n, N_THETA, N_PHI), dtype=np.float32)
    for i in range(n):
        matlab[i] = matlab_array_db(16, float(dpx[i]), float(dpy[i]), theta_deg, phi_deg)
        sub_block[i] = compose_scale_magnitude_only(
            sub6_blocks[i], float(dpx[i]), float(dpy[i]),
            scale=16, theta_deg=theta_deg, phi_deg=phi_deg,
        )

    fingerprint: np.ndarray | None = None
    if EXTRAS_OLD.exists() and "matlab_2x2" in np.load(EXTRAS_OLD):
        extras = np.load(EXTRAS_OLD)
        key4 = {
            steering_key(extras["dpx"][j], extras["dpy"][j]): j
            for j in range(len(extras["dpx"]))
        }
        fp = np.empty((n, N_THETA, N_PHI), dtype=np.float32)
        for i in range(n):
            j4 = key4.get(steering_key(dpx[i], dpy[i]))
            if j4 is not None:
                fp[i] = (
                    extras["matlab_2x2"][j4].astype(np.float32)
                    - extras["hfss_2x2_mean"][j4].astype(np.float32)
                )
            else:
                from scripts.stage1_6x6_fingerprint import fingerprint_channel
                fp[i] = fingerprint_channel(matlab[i], sub6_blocks[i])
        fingerprint = fp

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n)
    n_tr = int(0.80 * n)
    n_va = int(0.10 * n)
    train_idx = np.sort(perm[:n_tr])
    val_idx = np.sort(perm[n_tr : n_tr + n_va])
    test_idx = np.sort(perm[n_tr + n_va :])

    def _stats(arr: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = arr[idx].astype(np.float32)
        return s.mean(0), np.maximum(s.std(0), 1e-6).astype(np.float32)

    matlab_mean, matlab_std = _stats(matlab, train_idx)
    sub_mean, sub_std = _stats(sub_block, train_idx)

    sub6_pair_idx = matched_6.astype(np.int64)

    return ZeroshotBundle(
        hfss=hfss,
        matlab=matlab,
        sub_block=sub_block,
        dpx=dpx,
        dpy=dpy,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        matlab_mean=matlab_mean,
        matlab_std=matlab_std,
        sub_mean=sub_mean,
        sub_std=sub_std,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        sub6_blocks=sub6_blocks,
        sub6_pair_idx=sub6_pair_idx,
        fingerprint=fingerprint,
        sample_ids=sample_ids,
    )
