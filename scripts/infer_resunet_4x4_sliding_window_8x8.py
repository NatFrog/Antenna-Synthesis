"""
Sliding-window 8x8 HFSS inference via the trained 4x4 sub-block coupling ResUNet.

Per sample (subblock_compositions.npz):
  1. Global 8x8 composition baseline (peak-norm sub_block_8x8) and steering (dpx, dpy).
  2. Recover element gain from the composition + full 8x8 array factor (same idea as derive_*).
  3. For each of 25 overlapping 4x4 element windows on the 8x8 grid:
       - Per-window baseline = simulate that 4x4 patch using the composition element pattern.
       - MATLAB ch0 = ideal 4x4 patch (analytical element) at window position.
       - Sub-block ch1-4 = four copies of the per-window baseline (peak-norm), Z-scored like b0..b3.
       - ch5-7 = global 2x2 residual + dphase from subblock HDF5.
       - Predict coupling residual; window HFSS = peak_norm(baseline_win + coupling_pred).
  4. Stitch window HFSS maps with Gaussian weights (centre windows weigh more on the 8x8 lattice).

Optional: --export-matrix writes a 64x64 coupling-strength proxy (not used for scoring).

Usage:
    python -m scripts.infer_resunet_4x4_sliding_window_8x8
    python -m scripts.infer_resunet_4x4_sliding_window_8x8 --max-samples 100 --save-npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.config import BATCH_SIZE, CHECKPOINTS_DIR, DEVICE, N_THETA, PROCESSED_DIR, RESULTS_DIR
from src.training.metrics import compute_batch_metrics, compute_pattern_metrics, rmse
from scripts.train_cgan_2to4_fusion_no_m4 import ATTN_HEADS, GEN_BASE, EnhancedResUNetGenerator
from scripts.train_resunet_4x4_subblock_coupling import (
    HDF5 as H5_SUB,
    IN_CH,
    NORM_2X2_HFSS,
    NORM_2X2_MATLAB,
    NORM_CPL,
    NORM_SUB,
    NORM_4X4,
    OUT_CH,
)

C_LIGHT = 3e8
FREQ = 2.4e9
LAM = C_LIGHT / FREQ
K = 2 * np.pi / LAM
DX = 0.5 * LAM
DY = 0.5 * LAM
PARENT = 8
WIN = 4
ARRAY_CENTRE = ((PARENT - 1) / 2.0, (PARENT - 1) / 2.0)
N_WINDOWS = (PARENT - WIN + 1) ** 2

COMP_NPZ = PROCESSED_DIR / "subblock_compositions.npz"
H5_8X8 = PROCESSED_DIR / "antenna_data_8x8.h5"
ELEM_NPZ = PROJECT_ROOT / "datasets_2x2_from_4x4_workspace_elem_recovered.npz"
CKPT = CHECKPOINTS_DIR / "resunet_4x4_subblock_coupling_b4" / "best_generator.pt"
DEFAULT_OUT = RESULTS_DIR / "resunet_4x4_sliding_window_8x8"

WINDOW_ORIGINS = [(r, c) for r in range(PARENT - WIN + 1) for c in range(PARENT - WIN + 1)]
WIN_CENTRE = ((WIN - 1) / 2.0, (WIN - 1) / 2.0)


def _peak_norm_db(pat: np.ndarray) -> np.ndarray:
    p = pat.astype(np.float32)
    return p - float(p.max())


def _zscore(pat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((pat - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def _build_grids() -> tuple[np.ndarray, np.ndarray]:
    theta = np.arange(0, N_THETA, dtype=np.float64)
    phi = np.arange(-179.5, 180.0, 1.0, dtype=np.float64)
    TH, PH = np.meshgrid(np.deg2rad(theta), np.deg2rad(phi), indexing="ij")
    return TH.astype(np.float32), PH.astype(np.float32)


def _load_elem_lin_ideal() -> np.ndarray:
    if not ELEM_NPZ.exists():
        raise FileNotFoundError(f"Missing {ELEM_NPZ}")
    return np.power(10.0, np.load(ELEM_NPZ)["elem_dB"].astype(np.float32) / 10.0)


def _af_complex_window(
    dphase_x: float,
    dphase_y: float,
    TH: np.ndarray,
    PH: np.ndarray,
    r0: int,
    c0: int,
    parent: int = PARENT,
) -> np.ndarray:
    af = np.zeros_like(TH, dtype=np.complex64)
    for li in range(WIN):
        for lj in range(WIN):
            gi, gj = r0 + li, c0 + lj
            mx = (gi + 1) - (parent + 1) / 2.0
            ny = (gj + 1) - (parent + 1) / 2.0
            beta = np.deg2rad(mx * dphase_x + ny * dphase_y)
            xm = (gi - (parent - 1) / 2.0) * DX
            yn = (gj - (parent - 1) / 2.0) * DY
            phase = K * (xm * np.sin(TH) * np.cos(PH) + yn * np.sin(TH) * np.sin(PH)) + beta
            af += np.exp(1j * phase)
    return af


def _af_complex_full(
    dphase_x: float,
    dphase_y: float,
    TH: np.ndarray,
    PH: np.ndarray,
    parent: int = PARENT,
) -> np.ndarray:
    af = np.zeros_like(TH, dtype=np.complex64)
    for gi in range(parent):
        for gj in range(parent):
            mx = (gi + 1) - (parent + 1) / 2.0
            ny = (gj + 1) - (parent + 1) / 2.0
            beta = np.deg2rad(mx * dphase_x + ny * dphase_y)
            xm = (gi - (parent - 1) / 2.0) * DX
            yn = (gj - (parent - 1) / 2.0) * DY
            phase = K * (xm * np.sin(TH) * np.cos(PH) + yn * np.sin(TH) * np.sin(PH)) + beta
            af += np.exp(1j * phase)
    return af


def recover_elem_lin_from_composition(
    g8_db: np.ndarray,
    dphase_x: float,
    dphase_y: float,
    TH: np.ndarray,
    PH: np.ndarray,
) -> np.ndarray:
    """elem_lin from peak-norm 8x8 composition: G ~ elem * |AF_8x8|^2."""
    af = _af_complex_full(dphase_x, dphase_y, TH, PH)
    af2_db = 10.0 * np.log10(np.abs(af) ** 2 + np.finfo(float).eps)
    h_db = g8_db.astype(np.float64) - af2_db
    mask = af2_db > (float(af2_db.max()) - 30.0)
    h_db = np.where(mask, h_db, np.nan)
    if np.isnan(h_db).any():
        from scipy.ndimage import distance_transform_edt

        _, idx = distance_transform_edt(np.isnan(h_db), return_indices=True)
        h_db = h_db[tuple(idx)]
    h_db = h_db - float(np.nanmax(h_db))
    return np.power(10.0, h_db / 10.0).astype(np.float32)


def simulate_patch_db(
    dphase_x: float,
    dphase_y: float,
    elem_lin: np.ndarray,
    TH: np.ndarray,
    PH: np.ndarray,
    r0: int,
    c0: int,
) -> np.ndarray:
    af = _af_complex_window(dphase_x, dphase_y, TH, PH, r0, c0)
    p = elem_lin * (np.abs(af) ** 2)
    g_db = 10.0 * np.log10(p + np.finfo(float).eps)
    return _peak_norm_db(g_db)


def window_lattice_weight(r0: int, c0: int, sigma: float) -> float:
    """Higher weight for windows whose centre is near the 8x8 array centre."""
    cx = r0 + WIN_CENTRE[0]
    cy = c0 + WIN_CENTRE[1]
    d2 = (cx - ARRAY_CENTRE[0]) ** 2 + (cy - ARRAY_CENTRE[1]) ** 2
    return float(np.exp(-0.5 * d2 / (sigma**2)))


def build_window_batch_inputs(
    mat4_batch: np.ndarray,
    baseline_win_batch: np.ndarray,
    res2_n: np.ndarray,
    dpx: float,
    dpy: float,
    mean_4: np.ndarray,
    std_4: np.ndarray,
    mean_sub: np.ndarray,
    std_sub: np.ndarray,
) -> np.ndarray:
    """(n_win, 8, H, W) for one sample."""
    n_win = mat4_batch.shape[0]
    m4_n = _zscore(mat4_batch, mean_4[None], std_4[None])
    sb_stack = np.broadcast_to(
        baseline_win_batch[:, None], (n_win, 4, N_THETA, baseline_win_batch.shape[-1])
    ).astype(np.float32)
    sb_n = _zscore(sb_stack, mean_sub[None], std_sub[None])
    res2_plane = np.broadcast_to(
        res2_n[None, None], (n_win, 1, N_THETA, res2_n.shape[-1])
    )
    dpx_p = np.full((n_win, 1, N_THETA, mat4_batch.shape[-1]), dpx / 180.0, dtype=np.float32)
    dpy_p = np.full((n_win, 1, N_THETA, mat4_batch.shape[-1]), dpy / 180.0, dtype=np.float32)
    return np.concatenate([m4_n[:, None], sb_n, res2_plane, dpx_p, dpy_p], axis=1).astype(np.float32)


def stitch_farfield(
    maps: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted blend of (n_win, H, W) patterns."""
    w = weights.astype(np.float64)[:, None, None]
    acc = (maps.astype(np.float64) * w).sum(axis=0)
    wsum = w.sum()
    return (acc / max(wsum, 1e-12)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=COMP_NPZ)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--stitch-sigma", type=float, default=1.25, help="Window weight sigma in element indices")
    ap.add_argument("--n-vis", type=int, default=4)
    ap.add_argument("--save-npz", action="store_true")
    ap.add_argument("--export-matrix", action="store_true", help="Optional 64x64 proxy (not used for HFSS stitch)")
    args = ap.parse_args()

    for p in (args.npz, args.ckpt, H5_SUB, NORM_4X4, NORM_SUB, NORM_CPL, NORM_2X2_MATLAB, NORM_2X2_HFSS):
        if not p.exists():
            raise FileNotFoundError(p)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    TH, PH = _build_grids()
    elem_ideal = _load_elem_lin_ideal()
    win_weights = np.array(
        [window_lattice_weight(r, c, args.stitch_sigma) for r, c in WINDOW_ORIGINS],
        dtype=np.float32,
    )

    s4 = np.load(NORM_4X4)
    ss = np.load(NORM_SUB)
    sc = np.load(NORM_CPL)
    sm2 = np.load(NORM_2X2_MATLAB)
    sh2 = np.load(NORM_2X2_HFSS)
    mean_4 = s4["mean"].astype(np.float32)
    std_4 = np.maximum(s4["std"].astype(np.float32), 1e-6)
    mean_sub = ss["mean"].astype(np.float32)
    std_sub = np.maximum(ss["std"].astype(np.float32), 1e-6)
    mean_cpl = sc["mean"].astype(np.float32)
    std_cpl = np.maximum(sc["std"].astype(np.float32), 1e-6)
    mean_m2 = sm2["mean"].astype(np.float32)
    std_m2 = np.maximum(sm2["std"].astype(np.float32), 1e-6)
    mean_h2 = sh2["mean"].astype(np.float32)
    std_h2 = np.maximum(sh2["std"].astype(np.float32), 1e-6)

    comp = np.load(args.npz)
    n_all = len(comp["idx"])
    n_use = n_all if args.max_samples <= 0 else min(args.max_samples, n_all)
    idx_1 = comp["idx"][:n_use].astype(np.int64)
    gi0 = idx_1 - 1
    sb8 = comp["sub_block_8x8"][:n_use].astype(np.float32)
    dpx_all = comp["dpx"][:n_use].astype(np.float32)
    dpy_all = comp["dpy"][:n_use].astype(np.float32)
    baseline_8 = np.stack([_peak_norm_db(sb8[i]) for i in range(n_use)], axis=0)

    hfss8_truth = None
    if H5_8X8.exists():
        with h5py.File(H5_8X8, "r") as f8:
            if gi0.max() < f8["hfss_patterns"].shape[0]:
                hfss8_truth = np.stack(
                    [_peak_norm_db(f8["hfss_patterns"][int(gi)].astype(np.float32)) for gi in gi0],
                    axis=0,
                )

    G = EnhancedResUNetGenerator(in_ch=IN_CH, out_ch=OUT_CH, base=GEN_BASE, attn_heads=ATTN_HEADS).to(DEVICE)
    G.load_state_dict(torch.load(args.ckpt, map_location=DEVICE, weights_only=True))
    G.eval()
    print(
        f"Loaded {args.ckpt.name} | {n_use} samples x {N_WINDOWS} windows | "
        f"stitch_sigma={args.stitch_sigma} | device={DEVICE}",
        flush=True,
    )

    pred_hfss_list = []
    pred_hfss_global_base_list = []

    for si in tqdm(range(n_use), desc="Samples"):
        gi = int(gi0[si])
        dpx, dpy = float(dpx_all[si]), float(dpy_all[si])
        base8 = baseline_8[si]

        elem_comp = recover_elem_lin_from_composition(base8, dpx, dpy, TH, PH)

        with h5py.File(H5_SUB, "r") as fs:
            if gi >= fs["subblock_4x4"].shape[0]:
                raise IndexError(f"global idx {gi} not in subblock HDF5 (n={fs['subblock_4x4'].shape[0]})")
            m2 = fs["matlab_2x2"][gi].astype(np.float32)
            h2 = fs["hfss_2x2"][gi].astype(np.float32)

        res2_n = (_zscore(m2, mean_m2, std_m2) - _zscore(h2, mean_h2, std_h2)).astype(np.float32)

        mat4_batch = np.stack(
            [simulate_patch_db(dpx, dpy, elem_ideal, TH, PH, r0, c0) for r0, c0 in WINDOW_ORIGINS],
            axis=0,
        )
        baseline_win_batch = np.stack(
            [simulate_patch_db(dpx, dpy, elem_comp, TH, PH, r0, c0) for r0, c0 in WINDOW_ORIGINS],
            axis=0,
        )

        x_batch = build_window_batch_inputs(
            mat4_batch, baseline_win_batch, res2_n, dpx, dpy,
            mean_4, std_4, mean_sub, std_sub,
        )
        x = torch.from_numpy(x_batch).to(DEVICE)
        with torch.no_grad():
            pred_n = G(x).cpu().numpy()[:, 0]
        pred_cpl = pred_n * std_cpl[None] + mean_cpl[None]

        hfss_win = np.stack(
            [_peak_norm_db(baseline_win_batch[w] + pred_cpl[w]) for w in range(N_WINDOWS)],
            axis=0,
        )
        hfss_hat = _peak_norm_db(stitch_farfield(hfss_win, win_weights))

        cpl_stitched = stitch_farfield(pred_cpl.astype(np.float32), win_weights)
        hfss_global_stitch = _peak_norm_db(base8 + cpl_stitched)

        pred_hfss_list.append(hfss_hat)
        pred_hfss_global_base_list.append(hfss_global_stitch)

    pred_hfss = np.stack(pred_hfss_list, axis=0)
    pred_global = np.stack(pred_hfss_global_base_list, axis=0)

    metrics_lines = [
        f"checkpoint: {args.ckpt}",
        f"npz: {args.npz}",
        f"n_samples: {n_use}",
        f"n_windows: {N_WINDOWS}",
        f"stitch_sigma: {args.stitch_sigma}",
        "method_primary: per_window_composition_baseline + stitch_hfss_win",
        "method_alt: global_baseline_8 + stitch_coupling",
        "",
    ]

    if hfss8_truth is not None:
        pm = compute_batch_metrics(pred_hfss, hfss8_truth)
        pm_alt = compute_batch_metrics(pred_global, hfss8_truth)
        pb = compute_batch_metrics(baseline_8, hfss8_truth)
        print("\n=== Stitched per-window HFSS (composition baseline each window) ===")
        for k, v in pm.items():
            print(f"  {k}: {v:.6f}")
        print("\n=== Alt: global baseline_8 + stitched coupling ===")
        for k, v in pm_alt.items():
            print(f"  {k}: {v:.6f}")
        print("\n=== Baseline composition only ===")
        for k, v in pb.items():
            print(f"  {k}: {v:.6f}")
        for prefix, d in (("pred_win_stitch", pm), ("pred_global_cpl_stitch", pm_alt), ("baseline_comp8", pb)):
            for k, v in d.items():
                metrics_lines.append(f"{prefix}_{k}: {v:.6f}")

    metrics_path = args.out_dir / "metrics_sliding_window.txt"
    metrics_path.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    if args.save_npz:
        np.savez_compressed(
            args.out_dir / "sliding_window_outputs.npz",
            idx=idx_1,
            hfss_hat_farfield=pred_hfss,
            hfss_hat_global_coupling_stitch=pred_global,
            baseline_8x8=baseline_8,
            dpx=dpx_all,
            dpy=dpy_all,
            window_weights=win_weights,
        )

    rows = ["global_idx,hfss_rmse_win_stitch,hfss_rmse_global_cpl_stitch,baseline_rmse_db"]
    for i in range(n_use):
        gi = int(idx_1[i])
        if hfss8_truth is None:
            rows.append(f"{gi},nan,nan,nan")
            continue
        rows.append(
            f"{gi},"
            f"{compute_pattern_metrics(pred_hfss[i], hfss8_truth[i])['rmse_db']:.6f},"
            f"{compute_pattern_metrics(pred_global[i], hfss8_truth[i])['rmse_db']:.6f},"
            f"{compute_pattern_metrics(baseline_8[i], hfss8_truth[i])['rmse_db']:.6f}"
        )
    (args.out_dir / "per_sample.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    if args.n_vis > 0 and hfss8_truth is not None:
        try:
            import matplotlib.pyplot as plt

            extent = [-179.5, 179.5, 180, 0]
            for pi, i in enumerate(np.linspace(0, n_use - 1, min(args.n_vis, n_use), dtype=int)):
                fig, axes = plt.subplots(1, 4, figsize=(22, 5))
                panels = [
                    (baseline_8[i], "Comp 8x8"),
                    (pred_hfss[i], "Win-stitch pred"),
                    (hfss8_truth[i], "True HFSS 8x8"),
                ]
                for ax, (data, title) in zip(axes[:3], panels):
                    im = ax.imshow(data, aspect="auto", extent=extent, vmin=-40, vmax=0, cmap="jet")
                    ax.set_title(title)
                    plt.colorbar(im, ax=ax)
                err = pred_hfss[i] - hfss8_truth[i]
                em = max(abs(err.min()), abs(err.max()), 1.0)
                im = axes[3].imshow(err, aspect="auto", extent=extent, vmin=-em, vmax=em, cmap="RdBu_r")
                axes[3].set_title(f"Error RMSE={rmse(pred_hfss[i], hfss8_truth[i]):.2f} dB")
                plt.colorbar(im, ax=ax)
                fig.suptitle(f"idx={idx_1[i]} sliding-window (composition baselines)")
                plt.tight_layout()
                plt.savefig(args.out_dir / f"comparison_{pi:02d}.png", dpi=150, bbox_inches="tight")
                plt.close()
        except ImportError:
            pass

    print(f"\nWrote {metrics_path}", flush=True)
    print(f"Outputs under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
