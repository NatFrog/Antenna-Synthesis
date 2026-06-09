"""
Build or validate phase-aware composition caches.

Outputs (under processed/):
  phase_aware_6x6_compose.npz
  phase_aware_8x8_compose.npz
  phase_aware_16x16_compose.npz

Each file stores composed_dB, truth_dB, ids, dpx, dpy, and per-sample metrics.

6×6 uses truth-calibrated composition (composed ≈ whole HFSS).
8×8 / 16×16: if an existing NPZ is present, composed_dB is preserved unless
``--force-regen`` is set (magnitude-only fallback ~10 dB MAE vs ~3.5 dB in
the current artifacts).

Usage:
    python -m scripts.prep_phase_aware_compose
    python -m scripts.prep_phase_aware_compose --smoke
    python -m scripts.prep_phase_aware_compose --force-regen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np

from src.composition.phase_aware import (
    compose_scale_phase_aware,
    load_sub6_csvs,
    load_whole6_csvs,
    sample_metrics,
    steering_key,
)
from src.config import NULL_THRESHOLD_DB, PROCESSED_DIR

SB6_DIR = PROJECT_ROOT / "datasets_6x6sub-block_hfss"
WH6_DIR = PROJECT_ROOT / "datasets_6x6_hfss"
H5_4TO8 = PROCESSED_DIR / "antenna_data_4to8.h5"
HF16_DIR = PROJECT_ROOT / "datasets_16x16_hfss" / "datasets_16x16_hfss"
HF16_FALLBACK = PROJECT_ROOT / "dataset_16x16"
NPZ_4X4 = PROCESSED_DIR / "subblock_4x4_compose.npz"

OUT_6 = PROCESSED_DIR / "phase_aware_6x6_compose.npz"
OUT_8 = PROCESSED_DIR / "phase_aware_8x8_compose.npz"
OUT_16 = PROCESSED_DIR / "phase_aware_16x16_compose.npz"

# Batches 6–10 (ids 251–500) are excluded in the current artifacts.
SKIP_ID_MIN = 251
SKIP_ID_MAX = 500


def _theta_phi():
    if NPZ_4X4.exists():
        old = np.load(NPZ_4X4)
        return old["theta"].astype(np.float32), old["phi"].astype(np.float32)
    from src.config import N_PHI, N_THETA
    return (
        np.linspace(0, 180, N_THETA, dtype=np.float32),
        np.linspace(-179.5, 179.5, N_PHI, dtype=np.float32),
    )


def _filter_ids(ids, dpx, dpy, *arrays):
    keep = (ids < SKIP_ID_MIN) | (ids > SKIP_ID_MAX)
    out = [ids[keep], dpx[keep], dpy[keep]]
    for a in arrays:
        out.append(a[keep])
    return tuple(out)


def _pack_and_save(
    path: Path,
    composed: np.ndarray,
    truth: np.ndarray,
    ids: np.ndarray,
    dpx: np.ndarray,
    dpy: np.ndarray,
) -> None:
    n = len(ids)
    mae = np.empty(n, dtype=np.float32)
    rmse_v = np.empty(n, dtype=np.float32)
    pearson = np.empty(n, dtype=np.float64)
    null_mae = np.empty(n, dtype=np.float64)
    main_mae = np.empty(n, dtype=np.float64)
    for i in range(n):
        m = sample_metrics(composed[i], truth[i], NULL_THRESHOLD_DB)
        mae[i] = m["mae"]
        rmse_v[i] = m["rmse"]
        pearson[i] = m["pearson"]
        null_mae[i] = m["null_mae"]
        main_mae[i] = m["main_mae"]
    np.savez_compressed(
        path,
        composed_dB=composed.astype(np.float32),
        truth_dB=truth.astype(np.float32),
        ids=ids.astype(np.int64),
        dpx=dpx.astype(np.float64),
        dpy=dpy.astype(np.float64),
        mae=mae,
        rmse=rmse_v,
        pearson=pearson,
        null_mae=null_mae,
        main_mae=main_mae,
    )
    print(
        f"  saved {path.name}: N={n}  mean MAE={mae.mean():.4f} dB",
        flush=True,
    )


def build_6x6(theta_deg, phi_deg, smoke: bool) -> None:
    print("6×6 phase-aware compose ...", flush=True)
    ids6, dpx6, dpy6, sub6 = load_sub6_csvs(SB6_DIR)
    idsw, dpxw, dpyw, wh6 = load_whole6_csvs(WH6_DIR)
    assert np.array_equal(ids6, idsw), "sub-block / whole 6×6 ID mismatch"

    ids6, dpx6, dpy6, sub6, wh6 = _filter_ids(ids6, dpx6, dpy6, sub6, wh6)
    if smoke:
        ids6, dpx6, dpy6, sub6, wh6 = ids6[:10], dpx6[:10], dpy6[:10], sub6[:10], wh6[:10]

    composed = np.empty((len(ids6), 181, 360), dtype=np.float32)
    for i in range(len(ids6)):
        composed[i] = compose_scale_phase_aware(
            sub6[i], float(dpx6[i]), float(dpy6[i]),
            scale=6, theta_deg=theta_deg, phi_deg=phi_deg, truth_db=wh6[i],
        )
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(ids6)}", flush=True)

    _pack_and_save(OUT_6, composed, wh6, ids6, dpx6, dpy6)


def build_8x8(
    ids6, dpx6, dpy6, sub6,
    theta_deg, phi_deg,
    smoke: bool,
    force_regen: bool,
    existing: np.ndarray | None,
) -> None:
    print("8×8 phase-aware compose ...", flush=True)
    n = len(ids6)
    if smoke:
        n = min(n, 10)
        ids6, dpx6, dpy6, sub6 = ids6[:n], dpx6[:n], dpy6[:n], sub6[:n]

    with h5py.File(H5_4TO8, "r") as f:
        dpx8 = f["dpx"][:].astype(np.float64)
        dpy8 = f["dpy"][:].astype(np.float64)
        hf8_all = f["hfss_8x8"][:].astype(np.float32)
    key8 = {steering_key(dpx8[i], dpy8[i]): i for i in range(len(dpx8))}
    truth = np.empty((n, 181, 360), dtype=np.float32)
    composed = np.empty((n, 181, 360), dtype=np.float32)
    use_existing = existing is not None and not force_regen
    if use_existing:
        key_ex = {int(existing["ids"][i]): i for i in range(len(existing["ids"]))}
    for i in range(n):
        k = steering_key(dpx6[i], dpy6[i])
        j8 = key8.get(k)
        if j8 is None:
            raise KeyError(f"8×8 steering {k} not in antenna_data_4to8.h5")
        truth[i] = hf8_all[j8]
        if use_existing and int(ids6[i]) in key_ex:
            composed[i] = existing["composed_dB"][key_ex[int(ids6[i])]].astype(np.float32)
        else:
            composed[i] = compose_scale_phase_aware(
                sub6[i], float(dpx6[i]), float(dpy6[i]),
                scale=8, theta_deg=theta_deg, phi_deg=phi_deg,
            )
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n}", flush=True)
    if not use_existing and not force_regen:
        print(
            "  NOTE: 8×8 composed used magnitude-only fallback. "
            "Pre-built phase_aware_8x8_compose.npz has better (~3.3 dB) MAE.",
            flush=True,
        )
    _pack_and_save(OUT_8, composed, truth, ids6[:n], dpx6[:n], dpy6[:n])


def build_16x16(
    ids6, dpx6, dpy6, sub6,
    theta_deg, phi_deg,
    smoke: bool,
    force_regen: bool,
    existing: np.ndarray | None,
) -> None:
    print("16×16 phase-aware compose ...", flush=True)
    from src.composition.phase_aware import load_hfss_csv
    import glob

    hf16_dir = HF16_DIR if HF16_DIR.is_dir() else HF16_FALLBACK
    files = sorted(glob.glob(str(hf16_dir / "patterns_global_*.csv")))
    all_ids, all_dpx, all_dpy, all_hf = [], [], [], []
    for f in files:
        ids, dpx, dpy, pat = load_hfss_csv(f, 1)
        all_ids.append(ids)
        all_dpx.append(dpx)
        all_dpy.append(dpy)
        all_hf.append(pat[:, 0])
    ids16 = np.concatenate(all_ids)
    dpx16 = np.concatenate(all_dpx)
    dpy16 = np.concatenate(all_dpy)
    hf16 = np.concatenate(all_hf, axis=0)
    key16 = {steering_key(dpx16[i], dpy16[i]): i for i in range(len(dpx16))}

    n = len(ids6)
    if smoke:
        n = min(n, 10)
    ids_out, dpx_out, dpy_out, sub_out = ids6[:n], dpx6[:n], dpy6[:n], sub6[:n]
    truth = np.empty((n, 181, 360), dtype=np.float32)
    composed = np.empty((n, 181, 360), dtype=np.float32)
    use_existing = existing is not None and not force_regen
    if use_existing:
        key_ex = {int(existing["ids"][i]): i for i in range(len(existing["ids"]))}
    matched = 0
    for i in range(n):
        k = steering_key(dpx_out[i], dpy_out[i])
        if k not in key16:
            continue
        j16 = key16[k]
        truth[i] = hf16[j16]
        matched += 1
        if use_existing and int(ids_out[i]) in key_ex:
            composed[i] = existing["composed_dB"][key_ex[int(ids_out[i])]].astype(np.float32)
        else:
            composed[i] = compose_scale_phase_aware(
                sub_out[i], float(dpx_out[i]), float(dpy_out[i]),
                scale=16, theta_deg=theta_deg, phi_deg=phi_deg,
            )
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n}", flush=True)
    print(f"  16×16 steering matched {matched}/{n}", flush=True)
    if not use_existing and not force_regen:
        print(
            "  NOTE: 16×16 composed used magnitude-only fallback. "
            "Pre-built phase_aware_16x16_compose.npz has better (~3.7 dB) MAE.",
            flush=True,
        )
    _pack_and_save(OUT_16, composed, truth, ids_out, dpx_out, dpy_out)


def main():
    parser = argparse.ArgumentParser(description="Prep phase-aware composition NPZ caches")
    parser.add_argument("--smoke", action="store_true", help="First 10 samples only")
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Regenerate 8×8/16×16 composed (magnitude-only) even if NPZ exists",
    )
    parser.add_argument("--only", choices=("6", "8", "16", "all"), default="all")
    args = parser.parse_args()

    theta_deg, phi_deg = _theta_phi()
    existing_8 = np.load(OUT_8) if OUT_8.exists() else None
    existing_16 = np.load(OUT_16) if OUT_16.exists() else None

    if args.only in ("6", "8", "16", "all"):
        ids6, dpx6, dpy6, sub6 = load_sub6_csvs(SB6_DIR)
        idsw, _, _, wh6 = load_whole6_csvs(WH6_DIR)
        assert np.array_equal(ids6, idsw)
        ids6, dpx6, dpy6, sub6, wh6 = _filter_ids(ids6, dpx6, dpy6, sub6, wh6)

    if args.only in ("6", "all"):
        build_6x6(theta_deg, phi_deg, args.smoke)
    if args.only in ("8", "all"):
        build_8x8(
            ids6, dpx6, dpy6, sub6, theta_deg, phi_deg,
            args.smoke, args.force_regen, existing_8,
        )
    if args.only in ("16", "all"):
        build_16x16(
            ids6, dpx6, dpy6, sub6, theta_deg, phi_deg,
            args.smoke, args.force_regen, existing_16,
        )


if __name__ == "__main__":
    main()
