"""
Prep training cache for phase-aware 6×6 multi-head stage-1.

Reads ``phase_aware_6x6_compose.npz`` (or builds it via prep_phase_aware_compose)
and writes:

  stage1_6x6_phase_extras.npz
  norm_stats_stage1_6x6_phase.npz
  split_indices_stage1_6x6_phase.npz

Compared to ``prep_stage1_6x6.py``:
  - 1750 samples (excludes HFSS batch ids 251–500)
  - sub_block_6x6 = phase-aware composed (≈ HFSS truth, MAE ~10⁻⁵ dB)
  - residual target ≈ 0 (model learns near-identity at 6×6)

Usage:
    python -m scripts.prep_stage1_6x6_phase
    python -m scripts.prep_stage1_6x6_phase --rebuild-compose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.composition.phase_aware import (
    matlab_array_db,
    steering_key,
)
from src.config import PROCESSED_DIR, RANDOM_SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO
from scripts.stage1_6x6_fingerprint import fingerprint_channel

COMPOSE_6 = PROCESSED_DIR / "phase_aware_6x6_compose.npz"
EXTRAS_OLD = PROCESSED_DIR / "stage1_extras.npz"
NPZ_4X4 = PROCESSED_DIR / "subblock_4x4_compose.npz"  # optional: theta/phi axes

OUT_EXTRAS = PROCESSED_DIR / "stage1_6x6_phase_extras.npz"
OUT_NORM = PROCESSED_DIR / "norm_stats_stage1_6x6_phase.npz"
OUT_SPLIT = PROCESSED_DIR / "split_indices_stage1_6x6_phase.npz"


def _make_split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    tr = perm[:n_train]
    va = perm[n_train : n_train + n_val]
    te = perm[n_train + n_val :]
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def _load_sub6_for_fingerprint(ids: np.ndarray) -> np.ndarray:
    from src.composition.phase_aware import load_sub6_csvs

    sb_dir = PROJECT_ROOT / "datasets_6x6sub-block_hfss"
    ids6, _, _, sub6 = load_sub6_csvs(sb_dir)
    key6 = {int(ids6[i]): i for i in range(len(ids6))}
    out = np.empty((len(ids), 9, 181, 360), dtype=np.float32)
    missing = 0
    for i, sid in enumerate(ids):
        j = key6.get(int(sid))
        if j is None:
            missing += 1
            continue
        out[i] = sub6[j]
    if missing:
        raise KeyError(
            f"{missing}/{len(ids)} sample ids have no 6×6 sub-block CSV match"
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Prep phase-aware 6×6 stage-1 cache")
    parser.add_argument(
        "--rebuild-compose",
        action="store_true",
        help="Run prep_phase_aware_compose for 6×6 first",
    )
    args = parser.parse_args()

    if args.rebuild_compose or not COMPOSE_6.exists():
        from scripts.prep_phase_aware_compose import build_6x6, _theta_phi

        theta_deg, phi_deg = _theta_phi()
        build_6x6(theta_deg, phi_deg, smoke=False)

    pa = np.load(COMPOSE_6)
    n = len(pa["ids"])
    dpx = pa["dpx"].astype(np.float32)
    dpy = pa["dpy"].astype(np.float32)
    sub_block_6x6 = pa["composed_dB"].astype(np.float32)
    hfss_6x6 = pa["truth_dB"].astype(np.float32)

    if NPZ_4X4.exists():
        theta_deg = np.load(NPZ_4X4)["theta"].astype(np.float32)
        phi_deg = np.load(NPZ_4X4)["phi"].astype(np.float32)
    else:
        from src.config import N_PHI, N_THETA
        theta_deg = np.linspace(0, 180, N_THETA, dtype=np.float32)
        phi_deg = np.linspace(-179.5, 179.5, N_PHI, dtype=np.float32)

    print(f"Generating matlab_6x6 for {n} samples ...", flush=True)
    matlab_6x6 = np.empty((n, 181, 360), dtype=np.float32)
    for i in range(n):
        matlab_6x6[i] = matlab_array_db(6, float(dpx[i]), float(dpy[i]), theta_deg, phi_deg)
        if (i + 1) % 350 == 0:
            print(f"  matlab_6x6: {i + 1}/{n}", flush=True)

    print("Building fingerprint channel ...", flush=True)
    sub6_all = _load_sub6_for_fingerprint(pa["ids"])
    fingerprint = np.empty((n, 181, 360), dtype=np.float32)
    fp_from_old = 0
    if EXTRAS_OLD.exists() and "dpx" in np.load(EXTRAS_OLD):
        extras_old = np.load(EXTRAS_OLD)
        key4 = {
            steering_key(extras_old["dpx"][i], extras_old["dpy"][i]): i
            for i in range(len(extras_old["dpx"]))
        }
        for i in range(n):
            k4 = steering_key(dpx[i], dpy[i])
            if k4 in key4:
                j = key4[k4]
                fingerprint[i] = (
                    extras_old["matlab_2x2"][j].astype(np.float32)
                    - extras_old["hfss_2x2_mean"][j].astype(np.float32)
                )
                fp_from_old += 1
            else:
                fingerprint[i] = fingerprint_channel(matlab_6x6[i], sub6_all[i])
        print(f"  fingerprint: {fp_from_old}/{n} from stage1_extras, rest from sub-block mean")
    else:
        for i in range(n):
            fingerprint[i] = fingerprint_channel(matlab_6x6[i], sub6_all[i])
        print(f"  fingerprint: {n}/{n} from matlab_6x6 - mean(9 HFSS sub-blocks)")

    residual = (hfss_6x6 - sub_block_6x6).astype(np.float32)
    print(
        f"Residual (hfss - sub_block) train stats preview: "
        f"mean={residual.mean():+.6f}  std={residual.std():.6f}  "
        f"max|.|={np.abs(residual).max():.6f}",
        flush=True,
    )

    np.savez_compressed(
        OUT_EXTRAS,
        matlab_6x6=matlab_6x6.astype(np.float16),
        sub_block_6x6=sub_block_6x6.astype(np.float16),
        hfss_6x6=hfss_6x6.astype(np.float16),
        fingerprint=fingerprint.astype(np.float16),
        dpx=dpx,
        dpy=dpy,
        ids=pa["ids"].astype(np.int64),
        theta=theta_deg,
        phi=phi_deg,
    )

    tr, va, te = _make_split(n)
    np.savez_compressed(OUT_SPLIT, train=tr, val=va, test=te)
    print(f"Split: {len(tr)} train / {len(va)} val / {len(te)} test")

    def stats(arr, indices):
        a = arr[indices].astype(np.float32)
        return a.mean(axis=0), np.maximum(a.std(axis=0), 1e-6)

    m6_mean, m6_std = stats(matlab_6x6, tr)
    sb_mean, sb_std = stats(sub_block_6x6, tr)
    fp_mean, fp_std = stats(fingerprint, tr)
    res_mean, res_std = stats(residual, tr)

    np.savez_compressed(
        OUT_NORM,
        matlab_6x6_mean=m6_mean.astype(np.float32),
        matlab_6x6_std=m6_std.astype(np.float32),
        sub_block_6x6_mean=sb_mean.astype(np.float32),
        sub_block_6x6_std=sb_std.astype(np.float32),
        fingerprint_mean=fp_mean.astype(np.float32),
        fingerprint_std=fp_std.astype(np.float32),
        residual_mean=res_mean.astype(np.float32),
        residual_std=res_std.astype(np.float32),
    )
    print(f"Saved {OUT_EXTRAS.name}, {OUT_NORM.name}, {OUT_SPLIT.name}")


if __name__ == "__main__":
    main()
