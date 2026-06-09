"""
Prep cache for matlab-relative 6×6 multi-head training (standalone pipeline).

Uses **only** native 6×6 sources — no ``subblock_4x4_compose.npz``, no
``stage1_extras.npz``, no reused 4×4 splits:

  - HFSS 6×6 sub-blocks  → ``datasets_6x6sub-block_*_fixed`` (else flat HFSS CSVs)
  - HFSS whole 6×6       → ``datasets_6x6_hfss``
  - MATLAB 6×6           → analytical 36-element AF from steering in CSVs

Samples are aligned by ``sample_id`` between sub-block and whole-array CSVs.
Fingerprint is derived on the fly: ``matlab_6x6 − mean(9 HFSS sub-blocks)``.

Target residual (norm stats): ``hfss_6x6 − matlab_6x6``.
Coupling gap is **not** stored; trainers derive ``sub_block − matlab``.

Outputs (``processed/``):

  stage1_6x6_mr_extras.npz
  norm_stats_stage1_6x6_mr.npz
  split_indices_stage1_6x6_mr.npz

Usage:
    python -m scripts.prep_stage1_6x6_matlab_rel
    python -m scripts.prep_stage1_6x6_matlab_rel --sb-dir path/to/subblock_csvs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.composition.phase_aware import (
    compose_scale_magnitude_only,
    load_sub6_csvs,
    load_whole6_csvs,
    matlab_array_db,
)
from src.config import N_PHI, N_THETA, PROCESSED_DIR, RANDOM_SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO
from scripts.stage1_6x6_fingerprint import fingerprint_channel

SB6_FIXED_CANDIDATES = (
    PROJECT_ROOT / "datasets_6x6sub-block_hfss_fixed",
    PROJECT_ROOT / "datasets_6x6sub-block_rE_hfss_fixed",
)
SB6_FALLBACK = PROJECT_ROOT / "datasets_6x6sub-block_hfss"
WH6_CANDIDATES = (
    PROJECT_ROOT / "datasets_6x6_hfss",
    PROJECT_ROOT / "datasets_6x6_hfss" / "datasets_6x6_hfss",
)

OUT_EXTRAS = PROCESSED_DIR / "stage1_6x6_mr_extras.npz"
OUT_NORM = PROCESSED_DIR / "norm_stats_stage1_6x6_mr.npz"
OUT_SPLIT = PROCESSED_DIR / "split_indices_stage1_6x6_mr.npz"


def _resolve_wh6_dir() -> Path:
    import glob

    best: Path | None = None
    best_n = -1
    for d in WH6_CANDIDATES:
        if not d.is_dir():
            continue
        n = len(glob.glob(str(d / "patterns_global_*.csv")))
        if n > best_n:
            best_n = n
            best = d
    if best is None:
        raise FileNotFoundError(
            "No whole 6×6 HFSS CSV directory found. Tried:\n  "
            + "\n  ".join(str(p) for p in WH6_CANDIDATES)
        )
    return best


def _try_load_sb6_dir(d: Path) -> int | None:
    """Return sample count if CSVs parse cleanly, else None."""
    try:
        ids, _, _, _ = load_sub6_csvs(d)
        return len(ids)
    except Exception as exc:
        print(f"  skip {d.name}: {exc}", flush=True)
        return None


def _resolve_sb6_dir(explicit: Path | None) -> Path:
    import glob

    if explicit is not None:
        if not explicit.is_dir():
            raise FileNotFoundError(f"--sb-dir not found: {explicit}")
        if not glob.glob(str(explicit / "patterns_global_*.csv")):
            raise FileNotFoundError(f"No patterns_global_*.csv in {explicit}")
        n = _try_load_sb6_dir(explicit)
        if n is None:
            raise RuntimeError(f"Could not parse sub-block CSVs in {explicit}")
        print(f"  using --sb-dir ({n} samples)", flush=True)
        return explicit

    candidates = (*SB6_FIXED_CANDIDATES, SB6_FALLBACK)
    for d in candidates:
        if not d.is_dir() or not glob.glob(str(d / "patterns_global_*.csv")):
            continue
        n = _try_load_sb6_dir(d)
        if n is not None:
            if d in SB6_FIXED_CANDIDATES:
                print(f"  using fixed sub-block dir ({n} samples)", flush=True)
            else:
                print(f"  WARNING: fixed dirs failed parse; using {d} ({n} samples)", flush=True)
            return d

    raise FileNotFoundError(
        "No parseable 6×6 sub-block CSV directory found. Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def _make_split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    tr = np.sort(perm[:n_train])
    va = np.sort(perm[n_train : n_train + n_val])
    te = np.sort(perm[n_train + n_val :])
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def _pair_by_id(
    ids_sub: np.ndarray,
    dpx_sub: np.ndarray,
    dpy_sub: np.ndarray,
    sub6: np.ndarray,
    ids_wh: np.ndarray,
    dpx_wh: np.ndarray,
    dpy_wh: np.ndarray,
    hfss: np.ndarray,
) -> tuple[np.ndarray, ...]:
    key_wh = {int(ids_wh[i]): i for i in range(len(ids_wh))}
    rows_sub, rows_wh = [], []
    steer_mismatch = 0
    for i in range(len(ids_sub)):
        sid = int(ids_sub[i])
        j = key_wh.get(sid)
        if j is None:
            continue
        if (
            abs(float(dpx_sub[i]) - float(dpx_wh[j])) > 1e-3
            or abs(float(dpy_sub[i]) - float(dpy_wh[j])) > 1e-3
        ):
            steer_mismatch += 1
            continue
        rows_sub.append(i)
        rows_wh.append(j)

    if not rows_sub:
        raise RuntimeError("No sample_id overlap between sub-block and whole 6×6 HFSS CSVs.")

    rows_sub = np.array(rows_sub, dtype=np.int64)
    rows_wh = np.array(rows_wh, dtype=np.int64)
    print(
        f"  paired {len(rows_sub)} samples by id "
        f"(sub={len(ids_sub)}  whole={len(ids_wh)}  steer_skip={steer_mismatch})",
        flush=True,
    )
    return (
        ids_sub[rows_sub],
        dpx_sub[rows_sub].astype(np.float32),
        dpy_sub[rows_sub].astype(np.float32),
        sub6[rows_sub],
        hfss[rows_wh].astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep matlab-relative 6×6 stage-1 cache")
    parser.add_argument(
        "--sb-dir",
        type=Path,
        default=None,
        help="Override 6×6 sub-block CSV directory",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sb6_dir = _resolve_sb6_dir(args.sb_dir)
    wh6_dir = _resolve_wh6_dir()
    print(f"Sub-block CSV dir: {sb6_dir}", flush=True)
    print(f"Whole 6×6 HFSS dir: {wh6_dir}", flush=True)

    theta_deg = np.linspace(0, 180, N_THETA, dtype=np.float32)
    phi_deg = np.linspace(-179.5, 179.5, N_PHI, dtype=np.float32)

    print("Loading 6×6 sub-block CSVs ...", flush=True)
    ids_sub, dpx_sub, dpy_sub, sub6 = load_sub6_csvs(sb6_dir)
    print(f"  {len(ids_sub)} sub-block rows", flush=True)

    print("Loading whole 6×6 HFSS CSVs ...", flush=True)
    ids_wh, dpx_wh, dpy_wh, hfss_all = load_whole6_csvs(wh6_dir)
    print(f"  {len(ids_wh)} whole-array rows", flush=True)

    sample_ids, dpx, dpy, sub6_blocks, hfss_6x6 = _pair_by_id(
        ids_sub, dpx_sub, dpy_sub, sub6,
        ids_wh, dpx_wh, dpy_wh, hfss_all,
    )
    n = len(sample_ids)

    print(f"Generating matlab_6x6 for {n} samples ...", flush=True)
    matlab_6x6 = np.empty((n, N_THETA, N_PHI), dtype=np.float32)
    for i in range(n):
        matlab_6x6[i] = matlab_array_db(6, float(dpx[i]), float(dpy[i]), theta_deg, phi_deg)
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  matlab_6x6: {i + 1}/{n}", flush=True)

    print("Composing sub_block_6x6 from nine HFSS sub-blocks ...", flush=True)
    sub_block_6x6 = np.empty((n, N_THETA, N_PHI), dtype=np.float32)
    for i in range(n):
        sub_block_6x6[i] = compose_scale_magnitude_only(
            sub6_blocks[i], float(dpx[i]), float(dpy[i]),
            scale=6, theta_deg=theta_deg, phi_deg=phi_deg,
        )
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  sub_block_6x6: {i + 1}/{n}", flush=True)

    print("Building fingerprint (matlab − mean(9 sub-blocks)) ...", flush=True)
    fingerprint = np.empty((n, N_THETA, N_PHI), dtype=np.float32)
    for i in range(n):
        fingerprint[i] = fingerprint_channel(matlab_6x6[i], sub6_blocks[i])

    residual = (hfss_6x6 - matlab_6x6).astype(np.float32)
    gap = (sub_block_6x6 - matlab_6x6).astype(np.float32)
    print(
        f"Residual hfss−matlab: mean={residual.mean():+.3f}  std={residual.std():.3f}  "
        f"|coupling_gap| mean={np.abs(gap).mean():.3f}",
        flush=True,
    )

    tr, va, te = _make_split(n)
    print(f"Split (seed {RANDOM_SEED}): {len(tr)} train / {len(va)} val / {len(te)} test", flush=True)

    def stats(arr: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = arr[idx].astype(np.float32)
        return s.mean(0), np.maximum(s.std(0), 1e-6).astype(np.float32)

    m6_mean, m6_std = stats(matlab_6x6, tr)
    sb_mean, sb_std = stats(sub_block_6x6, tr)
    fp_mean, fp_std = stats(fingerprint, tr)
    res_mean, res_std = stats(residual, tr)

    print(f"\nSaving {OUT_EXTRAS.name} ...", flush=True)
    np.savez_compressed(
        OUT_EXTRAS,
        matlab_6x6=matlab_6x6.astype(np.float16),
        sub_block_6x6=sub_block_6x6.astype(np.float16),
        hfss_6x6=hfss_6x6.astype(np.float16),
        fingerprint=fingerprint.astype(np.float16),
        dpx=dpx,
        dpy=dpy,
        sample_ids=sample_ids.astype(np.int64),
        theta=theta_deg,
        phi=phi_deg,
    )
    np.savez_compressed(OUT_SPLIT, train=tr, val=va, test=te)
    np.savez_compressed(
        OUT_NORM,
        matlab_6x6_mean=m6_mean,
        matlab_6x6_std=m6_std,
        sub_block_6x6_mean=sb_mean,
        sub_block_6x6_std=sb_std,
        fingerprint_mean=fp_mean,
        fingerprint_std=fp_std,
        residual_mean=res_mean,
        residual_std=res_std,
    )

    print("\nTrain-set channel stats (peak-norm dB):")
    for name, arr in [
        ("matlab_6x6", matlab_6x6[tr]),
        ("sub_block_6x6", sub_block_6x6[tr]),
        ("fingerprint", fingerprint[tr]),
        ("hfss_6x6", hfss_6x6[tr]),
        ("residual TGT", residual[tr]),
    ]:
        print(
            f"  {name:14s}  min={arr.min():+8.2f}  max={arr.max():+6.2f}  "
            f"mean={arr.mean():+7.3f}  std={arr.std():6.3f}",
            flush=True,
        )
    print(f"\nDone. Outputs in {PROCESSED_DIR}", flush=True)


if __name__ == "__main__":
    main()
