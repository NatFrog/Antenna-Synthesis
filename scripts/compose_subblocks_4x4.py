"""
Position-wise 4x4 composition from HFSS 2x2 sub-block measurements (b0..b3).

Each column ``s#####_b{k}`` in ``datasets_2x2Sub-blocks_hfss`` is the peak-normalised
far field when only the 2x2 tile at a fixed quadrant of the 4x4 array is excited.
This script places each block on the correct lattice corner using the progressive-phase
array factor (same physics as ``simulate_array.m`` / ``infer_resunet_4x4_sliding_window_8x8``):

  1. For block k at window origin (r0, c0), recover element gain:
         elem_k ~ (G_bk_lin) / |AF_2x2(r0,c0)|^2
     with null masking where |AF_2x2| is weak.

  2. Combine element patterns (median over k by default).

  3. Synthesise the full 4x4 pattern:
         G_4x4_lin = elem_combined * |AF_4x4|^2
         G_4x4_dB  = peak_norm(10*log10(G_4x4_lin))

Block → quadrant (4x4 parent, WIN=2), row-major window origins:

  b0 → (0, 0)  top-left     (corner tile in AEP.py labelling)
  b1 → (0, 1)  top-right    (edge)
  b2 → (1, 0)  bottom-left  (edge)
  b3 → (1, 1)  bottom-right (interior-class tile)

Steering (dphase_x, dphase_y) is read from the sub-block CSV metadata for each sample.

Usage:
    python -m scripts.compose_subblocks_4x4 --smoke
    python -m scripts.compose_subblocks_4x4 --verify-npz
    python -m scripts.compose_subblocks_4x4 --write-npz processed/subblock_compositions.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import N_CONFIGS_PER_FILE, N_PHI, N_THETA, PROCESSED_DIR, RESULTS_DIR
from src.data.loader import get_config_columns, get_file_path, load_single_csv
from src.training.metrics import rmse
from scripts.preprocess_data_4x4_subblock_coupling import (
    SUBBLOCK_DIR,
    _list_subblock_files,
    _load_subblock_blocks,
    _peak_norm_db,
    _subblock_mean_from_blocks,
)

# Physics (2.4 GHz, lambda/2 spacing) — match simulate_array.m / derive_* scripts
C_LIGHT = 3e8
FREQ = 2.4e9
LAM = C_LIGHT / FREQ
K = 2 * np.pi / LAM
DX = 0.5 * LAM
DY = 0.5 * LAM

PARENT = 4
WIN = 2
AF_NULL_DB = 30.0

# b0..b3 → (row, col) origin of the active 2x2 window on the 4x4 lattice
BLOCK_ORIGINS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
)

H4X4_DIR = PROJECT_ROOT / "datasets_4x4consistent_hfss"
DEFAULT_OUT_NPZ = PROCESSED_DIR / "subblock_compositions.npz"


def build_grids() -> tuple[np.ndarray, np.ndarray]:
    theta = np.arange(0, N_THETA, dtype=np.float64)
    phi = np.arange(-179.5, 180.0, 1.0, dtype=np.float64)
    th, ph = np.meshgrid(np.deg2rad(theta), np.deg2rad(phi), indexing="ij")
    return th.astype(np.float32), ph.astype(np.float32)


def af_complex_window(
    dphase_x: float,
    dphase_y: float,
    th: np.ndarray,
    ph: np.ndarray,
    r0: int,
    c0: int,
    parent: int = PARENT,
    win: int = WIN,
) -> np.ndarray:
    """Complex AF for one WIN×WIN patch at (r0,c0) on a parent×parent array."""
    af = np.zeros_like(th, dtype=np.complex64)
    for li in range(win):
        for lj in range(win):
            gi, gj = r0 + li, c0 + lj
            mx = (gi + 1) - (parent + 1) / 2.0
            ny = (gj + 1) - (parent + 1) / 2.0
            beta = np.deg2rad(mx * dphase_x + ny * dphase_y)
            xm = (gi - (parent - 1) / 2.0) * DX
            yn = (gj - (parent - 1) / 2.0) * DY
            phase = K * (xm * np.sin(th) * np.cos(ph) + yn * np.sin(th) * np.sin(ph)) + beta
            af += np.exp(1j * phase.astype(np.complex64))
    return af


def af_complex_full(
    dphase_x: float,
    dphase_y: float,
    th: np.ndarray,
    ph: np.ndarray,
    parent: int = PARENT,
) -> np.ndarray:
    """Complex AF for the full parent×parent array."""
    af = np.zeros_like(th, dtype=np.complex64)
    for gi in range(parent):
        for gj in range(parent):
            mx = (gi + 1) - (parent + 1) / 2.0
            ny = (gj + 1) - (parent + 1) / 2.0
            beta = np.deg2rad(mx * dphase_x + ny * dphase_y)
            xm = (gi - (parent - 1) / 2.0) * DX
            yn = (gj - (parent - 1) / 2.0) * DY
            phase = K * (xm * np.sin(th) * np.cos(ph) + yn * np.sin(th) * np.sin(ph)) + beta
            af += np.exp(1j * phase.astype(np.complex64))
    return af


def recover_elem_lin_from_block(
    g_db: np.ndarray,
    dphase_x: float,
    dphase_y: float,
    th: np.ndarray,
    ph: np.ndarray,
    r0: int,
    c0: int,
) -> np.ndarray:
    """Element gain (linear) from one peak-norm sub-block pattern and its 2x2 AF."""
    af = af_complex_window(dphase_x, dphase_y, th, ph, r0, c0)
    af2_db = 10.0 * np.log10(np.abs(af) ** 2 + np.finfo(float).eps)
    h_db = g_db.astype(np.float64) - af2_db
    mask = af2_db > (float(af2_db.max()) - AF_NULL_DB)
    h_db = np.where(mask, h_db, np.nan)
    if np.isnan(h_db).any():
        from scipy.ndimage import distance_transform_edt

        _, idx = distance_transform_edt(np.isnan(h_db), return_indices=True)
        h_db = h_db[tuple(idx)]
    h_db = h_db - float(np.nanmax(h_db))
    return np.power(10.0, h_db / 10.0).astype(np.float32)


def compose_4x4_from_blocks(
    blocks_db: np.ndarray,
    dphase_x: float,
    dphase_y: float,
    th: np.ndarray,
    ph: np.ndarray,
    elem_combine: str = "median",
) -> np.ndarray:
    """
    Position-wise composition: recover elem per quadrant, combine, apply full 4x4 AF.

    Parameters
    ----------
    blocks_db : (4, H, W) peak-normalised dB patterns b0..b3
    elem_combine : ``median`` | ``mean``
    """
    if blocks_db.shape[0] != 4:
        raise ValueError(f"expected 4 blocks, got shape {blocks_db.shape}")

    elems = []
    for k, (r0, c0) in enumerate(BLOCK_ORIGINS):
        elems.append(recover_elem_lin_from_block(blocks_db[k], dphase_x, dphase_y, th, ph, r0, c0))
    stack = np.stack(elems, axis=0)
    if elem_combine == "median":
        elem_lin = np.nanmedian(stack, axis=0).astype(np.float32)
    elif elem_combine == "mean":
        elem_lin = np.nanmean(stack, axis=0).astype(np.float32)
    else:
        raise ValueError(f"unknown elem_combine={elem_combine!r}")

    af4 = af_complex_full(dphase_x, dphase_y, th, ph)
    p_lin = elem_lin * (np.abs(af4) ** 2)
    return _peak_norm_db(10.0 * np.log10(p_lin + np.finfo(float).eps))


def load_dphase_from_subblock_csv(file_idx: int, col: str, sub_df: pd.DataFrame) -> tuple[float, float]:
    meta_col = f"{col}_b0"
    if meta_col not in sub_df.columns:
        raise KeyError(f"{meta_col} missing for file_idx={file_idx}")
    return float(sub_df[meta_col].iloc[0]), float(sub_df[meta_col].iloc[1])


def iter_samples(
    sub_paths: list[Path],
    max_files: int | None = None,
    max_per_file: int | None = None,
):
    """Yield (global_idx_0, file_idx, col, blocks, dpx, dpy)."""
    paths = sub_paths[:max_files] if max_files else sub_paths
    gi = 0
    for sub_path in paths:
        file_idx = int(sub_path.stem.split("_")[-1])
        sub_df = pd.read_csv(sub_path, header=0, low_memory=False)
        cols = get_config_columns(file_idx)
        n_cols = len(cols) if max_per_file is None else min(max_per_file, len(cols))
        for col in cols[:n_cols]:
            blocks = _load_subblock_blocks(file_idx, col, sub_df)
            dpx, dpy = load_dphase_from_subblock_csv(file_idx, col, sub_df)
            yield gi, file_idx, col, blocks, dpx, dpy
            gi += 1


def verify_against_npz(
    comp_paths: list[Path],
    npz_path: Path,
    n_check: int,
    elem_combine: str,
) -> None:
    if not npz_path.exists():
        print(f"No npz at {npz_path} — skip verify", flush=True)
        return
    ref = np.load(npz_path)
    th, ph = build_grids()
    errs, errs_mean = [], []
    n = 0
    for gi, _fi, _col, blocks, dpx, dpy in iter_samples(comp_paths, max_files=None):
        if gi >= len(ref["sub_block_4x4"]) or n >= n_check:
            break
        pred = compose_4x4_from_blocks(blocks, dpx, dpy, th, ph, elem_combine=elem_combine)
        ref_pat = ref["sub_block_4x4"][gi].astype(np.float32)
        errs.append(rmse(pred, ref_pat))
        errs_mean.append(rmse(_subblock_mean_from_blocks(blocks), ref_pat))
        n += 1
    if n == 0:
        return
    print(f"\n=== Verify vs {npz_path.name} (n={n}) ===")
    print(f"  position-wise composition RMSE: {float(np.mean(errs)):.4f} dB (median {float(np.median(errs)):.4f})")
    print(f"  mean(b0..b3) re-peaknorm RMSE:   {float(np.mean(errs_mean)):.4f} dB")


def build_composed_4x4_from_h5(
    h5_path: Path,
    th: np.ndarray | None = None,
    ph: np.ndarray | None = None,
    elem_combine: str = "median",
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Position-wise 4x4 composition indexed by HDF5 global row."""
    import h5py

    if th is None or ph is None:
        th, ph = build_grids()
    with h5py.File(h5_path, "r") as f:
        n_total = int(f["subblock_4x4"].shape[0])
        idx_list = (
            np.arange(n_total, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        out = np.zeros((len(idx_list), N_THETA, N_PHI), dtype=np.float32)
        meta = f["metadata"]
        sb = f["subblock_4x4"]
        for j, gi in enumerate(tqdm(idx_list, desc="Compose from HDF5")):
            blocks = sb[int(gi)].astype(np.float32)
            m = meta[int(gi)].astype(np.float64)
            out[j] = compose_4x4_from_blocks(
                blocks, float(m[0]), float(m[1]), th, ph, elem_combine=elem_combine,
            )
        if indices is None:
            return out
        full = np.zeros((n_total, N_THETA, N_PHI), dtype=np.float32)
        full[idx_list] = out
        return full


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="first file, first 3 samples")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all sub-block CSV files")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all samples in selected files")
    ap.add_argument(
        "--elem-combine",
        choices=("median", "mean"),
        default="median",
        help="how to merge recovered element patterns across b0..b3",
    )
    ap.add_argument("--write-npz", type=Path, default=None, help="write subblock_compositions.npz")
    ap.add_argument("--verify-npz", type=Path, default=None, help="compare to existing npz")
    ap.add_argument("--verify-n", type=int, default=50)
    ap.add_argument("--compare-hfss", action="store_true", help="RMSE vs datasets_4x4consistent_hfss")
    ap.add_argument("--n-vis", type=int, default=0, help="save comparison PNGs if > 0")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "compose_subblocks_4x4")
    args = ap.parse_args()

    sub_paths = _list_subblock_files()
    max_files = 1 if args.smoke else (args.max_files or None)
    max_per_file = 3 if args.smoke else (args.max_samples or None)

    th, ph = build_grids()
    n_total = len(sub_paths[:max_files] if max_files else sub_paths) * N_CONFIGS_PER_FILE
    if max_per_file:
        n_total = min(n_total, (max_files or len(sub_paths)) * max_per_file)

    composed_list = []
    mean_blk_list = []
    blocks_list = []
    idx_1_list = []
    dpx_list = []
    dpy_list = []
    col_list = []
    hfss_comp_rmse: list[float] = []
    hfss_mean_rmse: list[float] = []

    samples = list(iter_samples(sub_paths, max_files=max_files, max_per_file=max_per_file))
    for gi, file_idx, col, blocks, dpx, dpy in tqdm(samples, desc="Compose 4x4"):
        comp = compose_4x4_from_blocks(blocks, dpx, dpy, th, ph, elem_combine=args.elem_combine)
        mean_p = _subblock_mean_from_blocks(blocks)
        composed_list.append(comp)
        mean_blk_list.append(mean_p)
        blocks_list.append(blocks)
        idx_1_list.append(gi + 1)
        dpx_list.append(dpx)
        dpy_list.append(dpy)
        col_list.append(col)

        if args.compare_hfss and H4X4_DIR.is_dir():
            p_h4 = get_file_path(H4X4_DIR, file_idx)
            if p_h4.exists():
                d_h4 = load_single_csv(p_h4, file_index=file_idx)
                ci = gi % N_CONFIGS_PER_FILE
                hfss4 = _peak_norm_db(d_h4["patterns"][ci].astype(np.float32))
                hfss_comp_rmse.append(rmse(comp, hfss4))
                hfss_mean_rmse.append(rmse(mean_p, hfss4))

    composed = np.stack(composed_list, axis=0).astype(np.float16)
    idx_1 = np.array(idx_1_list, dtype=np.int64)
    dpx_a = np.array(dpx_list, dtype=np.float64)
    dpy_a = np.array(dpy_list, dtype=np.float64)

    print(f"\nComposed {len(composed)} samples  (elem_combine={args.elem_combine})", flush=True)
    print("  Block origins (b0..b3):", BLOCK_ORIGINS, flush=True)

    if hfss_comp_rmse:
        print("\n=== vs datasets_4x4consistent_hfss (peak-norm) ===")
        print(f"  position-wise composition RMSE: {float(np.mean(hfss_comp_rmse)):.4f} dB")
        print(f"  mean(b0..b3) RMSE:               {float(np.mean(hfss_mean_rmse)):.4f} dB")

    if args.verify_npz is not None:
        verify_against_npz(sub_paths, args.verify_npz, args.verify_n, args.elem_combine)

    if args.write_npz is not None:
        args.write_npz.parent.mkdir(parents=True, exist_ok=True)
        theta = np.arange(0, N_THETA, dtype=np.float32)
        phi = np.arange(-179.5, 180.0, 1.0, dtype=np.float32)
        np.savez(
            args.write_npz,
            idx=idx_1,
            dpx=dpx_a,
            dpy=dpy_a,
            sub_block_4x4=composed,
            theta=theta,
            phi=phi,
        )
        print(f"\nWrote {args.write_npz}  shape={composed.shape}", flush=True)

    if args.n_vis > 0:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipped figures", flush=True)
        else:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            extent = [-179.5, 179.5, 180, 0]
            for i in range(min(args.n_vis, len(composed_list))):
                blocks = blocks_list[i]
                fig, axes = plt.subplots(2, 3, figsize=(16, 9))
                for k, ax in enumerate(axes.flat[:4]):
                    r0, c0 = BLOCK_ORIGINS[k]
                    im = ax.imshow(blocks[k], aspect="auto", extent=extent, vmin=-40, vmax=0, cmap="jet")
                    ax.set_title(f"b{k} @ ({r0},{c0})")
                    plt.colorbar(im, ax=ax)
                for ax, data, title in zip(
                    axes.flat[4:],
                    [mean_blk_list[i], composed_list[i]],
                    ["mean(b0..b3)", f"composed ({args.elem_combine})"],
                ):
                    im = ax.imshow(data, aspect="auto", extent=extent, vmin=-40, vmax=0, cmap="jet")
                    ax.set_title(title)
                    plt.colorbar(im, ax=ax)
                fig.suptitle(
                    f"{col_list[i]}  dpx={dpx_list[i]:.2f} dpy={dpy_list[i]:.2f}",
                    y=1.02,
                )
                plt.tight_layout()
                plt.savefig(args.out_dir / f"compose_{i:03d}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
            print(f"Figures: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
