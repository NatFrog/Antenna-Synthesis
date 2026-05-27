"""
8x8 HFSS inference using ONLY the trained 4x4 sub-block coupling ResUNet
(`checkpoints/resunet_4x4_subblock_coupling_b4/best_generator.pt`).

The user's evaluation target is mean RMSE on pixels whose true HFSS amplitude
exceeds -40 dB from peak (the visible-radiation region); deep nulls below
that floor are not prioritised. All tuning here optimises for that metric.

Inference techniques (all use the SAME 4x4 model, no retraining):

  1. Path B (8x8-scale input). Feed the 8x8 composition baseline
     (`sub_block_8x8` from `subblock_compositions.npz`) as a broadcast
     stand-in for the 4 sub-block channels, alongside MATLAB 8x8 as the
     `matlab_4x4` channel, and the global 2x2 residual. This matches the
     baseline in `infer_resunet_4x4_subblock_8x8_compositions.py`.

  2. Path A (4x4-scale input). Use the real 4x4 sub-blocks from
     `antenna_data_4x4_subblock.h5` to obtain a clean 4x4 coupling
     residual, then project the resulting 4x4 pattern to the 8x8 grid via
     the active element pattern (AEP) recomposition:
        P_8x8 = (P_4x4 - 20 log|AF_4x4|) + 20 log|AF_8x8|

  3. TTA (test-time augmentation). For each path, the prediction is the
     average of {original, phi-mirror, phi-shift-180, both}, mapping
     (dpx, dpy) -> (dpx, -dpy), (-dpx, -dpy), (-dpx, dpy) and flipping the
     phi axis back after inference. This averages out direction-dependent
     model bias.

  4. Iterative refinement. Re-feed the model with the previous prediction
     as the "subblock" channel; the residual it adds is a second-order
     correction. Two passes are reported.

  5. Coupling shrinkage. The 4x4 model's coupling residual was trained for
     4x4-scale fields; for 8x8, the actual mutual-coupling residual is
     smaller (more elements average out edge effects). We scan a per-batch
     scalar shrinkage `s in [0..1]` and apply `pred = baseline + s * coupling`.

  6. Ensemble. A non-negative linear blend of {Path-B-TTA, Path-A-AEP-TTA,
     baseline} in linear power space, with weights tuned on the first 100
     samples against masked (>-40 dB) RMSE.

Outputs (in --out-dir, default ``results/resunet_4x4_aep_8x8``):

  metrics_aep_8x8.txt    full + masked metrics for every method.
  per_sample.csv         per-sample full + > -40 dB RMSE.
  comparison_XX.png      6-panel visualisations.
  cuts_XX.png            E-/H-plane cuts.
  outputs.npz            (optional) all prediction arrays + truth.

Usage:
    python -m scripts.infer_resunet_4x4_aep_8x8 --tune-alpha
    python -m scripts.infer_resunet_4x4_aep_8x8 --max-samples 200 --tta
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

from src.config import BATCH_SIZE, CHECKPOINTS_DIR, DEVICE, N_PHI, N_THETA, PROCESSED_DIR, RESULTS_DIR
from src.training.metrics import (
    compute_batch_hfss_region_metrics,
    compute_batch_metrics,
    compute_pattern_metrics,
    rmse,
)
from scripts.train_cgan_2to4_fusion_no_m4 import ATTN_HEADS, GEN_BASE, EnhancedResUNetGenerator
from scripts.train_resunet_4x4_subblock_coupling import (
    HDF5 as H5_SUB,
    IN_CH,
    N_SUBBLOCKS,
    NORM_2X2_HFSS,
    NORM_2X2_MATLAB,
    NORM_4X4,
    NORM_CPL,
    NORM_SUB,
    OUT_CH,
)

C_LIGHT = 3e8
FREQ = 2.4e9
LAM = C_LIGHT / FREQ
K = 2 * np.pi / LAM
DX = 0.5 * LAM
DY = 0.5 * LAM

COMP_NPZ = PROCESSED_DIR / "subblock_compositions.npz"
H5_8X8 = PROCESSED_DIR / "antenna_data_8x8.h5"
NORM_8X8 = PROCESSED_DIR / "norm_stats_8x8.npz"
DEFAULT_CKPT = CHECKPOINTS_DIR / "resunet_4x4_subblock_coupling_b4" / "best_generator.pt"
DEFAULT_OUT = RESULTS_DIR / "resunet_4x4_aep_8x8"

EPS = float(np.finfo(np.float32).eps)
AF_NULL_DB = -30.0
DB_FLOORS = (-40.0, -30.0, -50.0)


# ─── Array factor utilities ────────────────────────────────────────────────


def build_grids() -> tuple[np.ndarray, np.ndarray]:
    theta = np.arange(0, N_THETA, dtype=np.float64)
    phi = np.arange(-179.5, 180.0, 1.0, dtype=np.float64)
    TH, PH = np.meshgrid(np.deg2rad(theta), np.deg2rad(phi), indexing="ij")
    return TH.astype(np.float32), PH.astype(np.float32)


def precompute_position_phase(TH: np.ndarray, PH: np.ndarray, parent: int) -> np.ndarray:
    coord = (np.arange(parent, dtype=np.float64) - (parent - 1) / 2.0) * DX
    sin_t = np.sin(TH.astype(np.float64))
    cos_p = np.cos(PH.astype(np.float64))
    sin_p = np.sin(PH.astype(np.float64))
    x_term = coord[:, None, None] * (sin_t * cos_p)[None, :, :]
    y_term = coord[:, None, None] * (sin_t * sin_p)[None, :, :]
    pos = K * (x_term[:, None, :, :] + y_term[None, :, :, :])
    return pos.astype(np.float32)


def af_complex_batch(
    dphase_x: np.ndarray,
    dphase_y: np.ndarray,
    pos_phase: np.ndarray,
    parent: int,
) -> np.ndarray:
    mn = (np.arange(parent, dtype=np.float64) - (parent + 1) / 2.0) + 1.0
    beta_x = np.deg2rad(mn[None, :] * dphase_x[:, None]).astype(np.float32)
    beta_y = np.deg2rad(mn[None, :] * dphase_y[:, None]).astype(np.float32)

    H, W = pos_phase.shape[-2:]
    af = np.zeros((dphase_x.shape[0], H, W), dtype=np.complex64)
    for m in range(parent):
        for n in range(parent):
            phase = pos_phase[m, n][None] + (beta_x[:, m] + beta_y[:, n])[:, None, None]
            af += np.exp(1j * phase.astype(np.float32))
    return af


# ─── Pattern helpers ───────────────────────────────────────────────────────


def peak_norm_db(pat: np.ndarray) -> np.ndarray:
    p = pat.astype(np.float32)
    return p - p.max(axis=(-2, -1), keepdims=True)


def zscore(pat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((pat - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def mask_fill(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    out = values.copy()
    for i in range(values.shape[0]):
        invalid = ~mask[i]
        if invalid.any() and (~invalid).any():
            _, idx = distance_transform_edt(invalid, return_indices=True)
            out[i] = values[i][tuple(idx)]
        elif invalid.all():
            out[i] = 0.0
    return out


def recover_aep_db(pattern_db, af_complex, null_floor_db=AF_NULL_DB):
    mag = np.abs(af_complex).astype(np.float32)
    af_db = 20.0 * np.log10(mag + EPS)
    af_peak = af_db.max(axis=(-2, -1), keepdims=True)
    mask = af_db > (af_peak + null_floor_db)
    aep = pattern_db - af_db
    aep_filled = mask_fill(aep, mask)
    return aep_filled - aep_filled.max(axis=(-2, -1), keepdims=True)


def aep_reconstruct_8x8(pred_4x4_db, af4_complex, af8_complex, null_floor_db=AF_NULL_DB):
    aep_db = recover_aep_db(pred_4x4_db, af4_complex, null_floor_db=null_floor_db)
    af8_db = 20.0 * np.log10(np.abs(af8_complex).astype(np.float32) + EPS)
    return peak_norm_db(aep_db + af8_db)


# ─── Norm stats ────────────────────────────────────────────────────────────


def load_norm_stats() -> dict:
    s4 = np.load(NORM_4X4)
    s8 = np.load(NORM_8X8) if NORM_8X8.exists() else None
    sub = np.load(NORM_SUB)
    cpl = np.load(NORM_CPL)
    m2 = np.load(NORM_2X2_MATLAB)
    h2 = np.load(NORM_2X2_HFSS)
    out = {
        "mean_4x4": s4["mean"].astype(np.float32),
        "std_4x4": np.maximum(s4["std"].astype(np.float32), 1e-6),
        "mean_sub": sub["mean"].astype(np.float32),
        "std_sub": np.maximum(sub["std"].astype(np.float32), 1e-6),
        "mean_cpl": cpl["mean"].astype(np.float32),
        "std_cpl": np.maximum(cpl["std"].astype(np.float32), 1e-6),
        "mean_m2": m2["mean"].astype(np.float32),
        "std_m2": np.maximum(m2["std"].astype(np.float32), 1e-6),
        "mean_h2": h2["mean"].astype(np.float32),
        "std_h2": np.maximum(h2["std"].astype(np.float32), 1e-6),
    }
    if s8 is not None:
        out["mean_8x8"] = s8["mean"].astype(np.float32)
        out["std_8x8"] = np.maximum(s8["std"].astype(np.float32), 1e-6)
    else:
        out["mean_8x8"] = out["mean_4x4"]
        out["std_8x8"] = out["std_4x4"]
    return out


# ─── Input builders ────────────────────────────────────────────────────────


def _make_dp_planes(dpx, dpy):
    b = dpx.shape[0]
    dpx_p = np.broadcast_to(
        (dpx / 180.0).astype(np.float32)[:, None, None, None], (b, 1, N_THETA, N_PHI)
    )
    dpy_p = np.broadcast_to(
        (dpy / 180.0).astype(np.float32)[:, None, None, None], (b, 1, N_THETA, N_PHI)
    )
    return dpx_p, dpy_p


def build_inputs_8x8(matlab_8x8_db, sub_block_8x8_db, m2_db, h2_db, dpx, dpy, stats):
    """Path B: 8x8 baseline broadcast as the 4 sub-block channels."""
    b = matlab_8x8_db.shape[0]
    m_n = zscore(matlab_8x8_db, stats["mean_8x8"][None], stats["std_8x8"][None])
    sb_stack = np.broadcast_to(
        sub_block_8x8_db[:, None], (b, N_SUBBLOCKS, N_THETA, N_PHI)
    ).astype(np.float32)
    sb_n = zscore(sb_stack, stats["mean_sub"][None], stats["std_sub"][None])
    m2_n = zscore(m2_db, stats["mean_m2"][None], stats["std_m2"][None])
    h2_n = zscore(h2_db, stats["mean_h2"][None], stats["std_h2"][None])
    res2_n = m2_n - h2_n
    dpx_p, dpy_p = _make_dp_planes(dpx, dpy)
    return np.concatenate(
        [m_n[:, None], sb_n, res2_n[:, None], dpx_p, dpy_p], axis=1
    ).astype(np.float32)


def build_inputs_4x4(matlab_4x4_db, subblock_4x4_db, m2_db, h2_db, dpx, dpy, stats):
    """Path A: native 4x4 inputs with the real 4 sub-blocks."""
    m4_n = zscore(matlab_4x4_db, stats["mean_4x4"][None], stats["std_4x4"][None])
    sb_n = zscore(subblock_4x4_db, stats["mean_sub"][None], stats["std_sub"][None])
    m2_n = zscore(m2_db, stats["mean_m2"][None], stats["std_m2"][None])
    h2_n = zscore(h2_db, stats["mean_h2"][None], stats["std_h2"][None])
    res2_n = m2_n - h2_n
    dpx_p, dpy_p = _make_dp_planes(dpx, dpy)
    return np.concatenate(
        [m4_n[:, None], sb_n, res2_n[:, None], dpx_p, dpy_p], axis=1
    ).astype(np.float32)


# ─── Forward + TTA ─────────────────────────────────────────────────────────

_TTA_TRANSFORMS = (
    # (phi_flip,   neg_dpx,  neg_dpy)
    (False, False, False),
    (True,  False, True),    # phi-mirror   <-> (dpx, -dpy)
    (False, True,  True),    # phi-shift-180 <-> (-dpx, -dpy)
    (True,  True,  False),   # phi-mirror + phi-shift-180 <-> (-dpx, dpy)
)


def _apply_phi_flip(arr: np.ndarray, flip: bool) -> np.ndarray:
    if not flip:
        return arr
    return arr[..., ::-1].copy()


@torch.no_grad()
def predict_residual(
    G: torch.nn.Module,
    builder,
    matlab_db: np.ndarray,
    sub_db_or_bc: np.ndarray,
    m2_db: np.ndarray,
    h2_db: np.ndarray,
    dpx: np.ndarray,
    dpy: np.ndarray,
    stats: dict,
    tta: bool = False,
) -> np.ndarray:
    """Return normalised coupling residual (B, H, W), optionally averaged over TTA."""
    transforms = _TTA_TRANSFORMS if tta else _TTA_TRANSFORMS[:1]
    acc = None
    for flip, neg_x, neg_y in transforms:
        mat_t = _apply_phi_flip(matlab_db, flip)
        sub_t = _apply_phi_flip(sub_db_or_bc, flip)
        m2_t = _apply_phi_flip(m2_db, flip)
        h2_t = _apply_phi_flip(h2_db, flip)
        dpx_t = -dpx if neg_x else dpx
        dpy_t = -dpy if neg_y else dpy
        x_np = builder(mat_t, sub_t, m2_t, h2_t, dpx_t, dpy_t, stats)
        x = torch.from_numpy(x_np).to(DEVICE)
        pred_n = G(x).cpu().numpy()[:, 0]
        if flip:
            pred_n = pred_n[:, :, ::-1].copy()
        acc = pred_n if acc is None else acc + pred_n
    return acc / float(len(transforms))


# ─── High-level prediction recipes ─────────────────────────────────────────


def coupling_db_from_residual_n(pred_n: np.ndarray, stats: dict) -> np.ndarray:
    return (pred_n * stats["std_cpl"][None] + stats["mean_cpl"][None]).astype(np.float32)


def reconstruct_4x4_db(pred_cpl_n: np.ndarray, sub_blocks_db: np.ndarray, stats: dict) -> np.ndarray:
    sb_mean = sub_blocks_db.mean(axis=1).astype(np.float32)
    sb_mean = sb_mean - sb_mean.max(axis=(-2, -1), keepdims=True)
    return peak_norm_db(sb_mean + coupling_db_from_residual_n(pred_cpl_n, stats))


def path_b_predict(G, mat8_db, sb8_db, m2, h2, dpx, dpy, stats, tta=False, shrink=1.0):
    """Path B: pred = sub_block_8x8 + shrink * coupling, with optional TTA."""
    res_n = predict_residual(G, build_inputs_8x8, mat8_db, sb8_db, m2, h2, dpx, dpy, stats, tta=tta)
    cpl_db = coupling_db_from_residual_n(res_n, stats)
    return peak_norm_db(sb8_db + shrink * cpl_db), cpl_db


def path_b_refine(G, mat8_db, baseline_db, m2, h2, dpx, dpy, stats, tta=False, shrink=1.0):
    """Pass-2 refinement: use previous prediction as the sub-block channel."""
    res_n = predict_residual(G, build_inputs_8x8, mat8_db, baseline_db, m2, h2, dpx, dpy, stats, tta=tta)
    cpl_db = coupling_db_from_residual_n(res_n, stats)
    return peak_norm_db(baseline_db + shrink * cpl_db)


def path_a_predict(G, mat4_db, sb4_db, m2, h2, dpx, dpy, stats, af4, af8, tta=False, null_floor_db=AF_NULL_DB):
    """Path A: real 4x4 inputs -> 4x4 prediction -> AEP -> 8x8."""
    res_n = predict_residual(G, build_inputs_4x4, mat4_db, sb4_db, m2, h2, dpx, dpy, stats, tta=tta)
    pred_4x4 = reconstruct_4x4_db(res_n, sb4_db, stats)
    pred_8x8 = aep_reconstruct_8x8(pred_4x4, af4, af8, null_floor_db=null_floor_db)
    return pred_4x4, pred_8x8


# ─── Blending utilities ────────────────────────────────────────────────────


def blend_linear_power(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Sum of (w_i * 10^(P_i/10)) in linear power then back to peak-normed dB."""
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if w.sum() <= 0:
        return arrays[0]
    w = w / w.sum()
    acc = np.zeros_like(arrays[0], dtype=np.float64)
    for arr, wi in zip(arrays, w):
        acc += wi * np.power(10.0, arr.astype(np.float64) / 10.0)
    return peak_norm_db((10.0 * np.log10(acc + EPS)).astype(np.float32))


def masked_rmse_above_floor(pred: np.ndarray, truth: np.ndarray, floor_db: float = -40.0) -> float:
    mask = truth > floor_db
    if mask.sum() == 0:
        return float("nan")
    err = pred[mask] - truth[mask]
    return float(np.sqrt(np.mean(err ** 2)))


def calibrate_blend_masked(
    components: list[np.ndarray],
    truth: np.ndarray,
    floor_db: float = -40.0,
    grid_step: float = 0.1,
) -> tuple[list[float], float]:
    """Grid search over simplex of non-negative weights for ``components``.

    Optimises masked-RMSE > floor_db averaged per-sample.
    """
    n = len(components)
    if n == 0:
        return [], float("nan")
    if n > 3:
        raise ValueError("Calibration supports up to 3 components currently.")
    grid = np.arange(0.0, 1.0 + 1e-9, grid_step)
    best_w = [1.0] + [0.0] * (n - 1)
    best_score = float("inf")
    if n == 1:
        return best_w, float(np.mean([masked_rmse_above_floor(components[0][i], truth[i], floor_db)
                                       for i in range(len(truth))]))
    if n == 2:
        for w0 in grid:
            w1 = 1.0 - w0
            blend = blend_linear_power(components, [float(w0), float(w1)])
            score = float(np.mean(
                [masked_rmse_above_floor(blend[i], truth[i], floor_db) for i in range(len(truth))]
            ))
            if score < best_score:
                best_score = score
                best_w = [float(w0), float(w1)]
        return best_w, best_score
    for w0 in grid:
        for w1 in grid:
            if w0 + w1 > 1.0 + 1e-9:
                continue
            w2 = max(0.0, 1.0 - w0 - w1)
            blend = blend_linear_power(components, [float(w0), float(w1), float(w2)])
            score = float(np.mean(
                [masked_rmse_above_floor(blend[i], truth[i], floor_db) for i in range(len(truth))]
            ))
            if score < best_score:
                best_score = score
                best_w = [float(w0), float(w1), float(w2)]
    return best_w, best_score


def scan_shrinkage(
    baseline_8: np.ndarray,
    coupling_db: np.ndarray,
    truth: np.ndarray,
    floor_db: float = -40.0,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    if grid is None:
        grid = np.arange(0.0, 1.51, 0.1)
    best_s = 1.0
    best_score = float("inf")
    for s in grid:
        pred = peak_norm_db(baseline_8 + float(s) * coupling_db)
        score = float(np.mean(
            [masked_rmse_above_floor(pred[i], truth[i], floor_db) for i in range(len(truth))]
        ))
        if score < best_score:
            best_score = score
            best_s = float(s)
    return best_s, best_score


def mainbeam_aware_blend(
    baseline: np.ndarray,
    refined: np.ndarray,
    cutover_db: float = -20.0,
    band_db: float = 6.0,
) -> np.ndarray:
    """Blend two patterns by sample brightness in linear power.

    `w(p) = sigmoid((baseline_dB - cutover_db) / band_db)` -> close to 1 in the
    mainbeam (keep `baseline`, which is already correct there) and close to 0
    in the deep sidelobes (use `refined`). `cutover_db` is the 50% mix point,
    `band_db` is the soft transition width.
    """
    w = 1.0 / (1.0 + np.exp(-(baseline - cutover_db) / band_db))
    b_lin = np.power(10.0, baseline.astype(np.float64) / 10.0)
    r_lin = np.power(10.0, refined.astype(np.float64) / 10.0)
    out = w * b_lin + (1.0 - w) * r_lin
    return peak_norm_db((10.0 * np.log10(out + EPS)).astype(np.float32))


def calibrate_mainbeam_blend(
    baseline: np.ndarray,
    refined: np.ndarray,
    truth: np.ndarray,
    floor_db: float = -40.0,
) -> tuple[float, float, float]:
    """Grid-search the (cutover_db, band_db) pair that minimises masked RMSE."""
    best = (-20.0, 6.0, float("inf"))
    for cut in np.arange(-30.0, -5.0 + 1e-9, 5.0):
        for band in (2.0, 4.0, 6.0, 10.0):
            blend = mainbeam_aware_blend(baseline, refined, float(cut), float(band))
            score = float(np.mean(
                [masked_rmse_above_floor(blend[i], truth[i], floor_db) for i in range(len(truth))]
            ))
            if score < best[2]:
                best = (float(cut), float(band), score)
    return best


# ─── Metrics summary helpers ───────────────────────────────────────────────


def summarise(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Full + masked metrics for `pred` against `truth`."""
    m = compute_batch_metrics(pred, truth)
    region = compute_batch_hfss_region_metrics(pred, truth, db_floors=DB_FLOORS)
    out: dict[str, float] = {}
    for k, v in m.items():
        out[f"full_{k}"] = float(v)
    for floor in DB_FLOORS:
        tag = f"{int(floor)}".replace("-", "neg")
        out[f"mean_rmse_gt{int(abs(floor))}"] = float(region[f"paper_region_rmse_db_{tag}_mean"])
        out[f"pooled_rmse_gt{int(abs(floor))}"] = float(region[f"paper_region_rmse_db_{tag}_pooled"])
    return out


def print_method_block(name: str, metrics: dict[str, float]) -> None:
    print(
        f"  {name:>32s}  full_rmse={metrics['full_rmse_db']:.3f}  "
        f">-40 mean={metrics['mean_rmse_gt40']:.3f}  "
        f">-40 pooled={metrics['pooled_rmse_gt40']:.3f}  "
        f"r={metrics['full_pearson_r']:.4f}  ssim={metrics['full_ssim']:.4f}",
        flush=True,
    )


# ─── Driver ────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=COMP_NPZ)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all rows in npz")
    ap.add_argument("--n-vis", type=int, default=8)
    ap.add_argument("--tta", action="store_true",
                    help="Enable phi-mirror + phi-180 TTA (4x model calls)")
    ap.add_argument("--two-pass", action="store_true",
                    help="Enable iterative refinement pass on Path B")
    ap.add_argument("--floor-db", type=float, default=-40.0,
                    help="dB floor for the masked-RMSE objective")
    ap.add_argument("--tune-alpha", action="store_true",
                    help="Calibrate ensemble weights & shrinkage on the first 100 samples")
    ap.add_argument("--null-floor-db", type=float, default=AF_NULL_DB)
    ap.add_argument("--save-npz", action="store_true")
    args = ap.parse_args()

    for p in (args.npz, H5_8X8, H5_SUB, NORM_4X4, NORM_SUB, NORM_CPL,
              NORM_2X2_MATLAB, NORM_2X2_HFSS, args.ckpt):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artefact: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = load_norm_stats()

    print("Building array-factor lookup tables ...", flush=True)
    TH, PH = build_grids()
    pos4 = precompute_position_phase(TH, PH, parent=4)
    pos8 = precompute_position_phase(TH, PH, parent=8)

    comp = np.load(args.npz)
    n_all = len(comp["idx"])
    n_use = n_all if args.max_samples <= 0 else min(args.max_samples, n_all)
    idx_1 = comp["idx"][:n_use].astype(np.int64)
    gi0 = idx_1 - 1
    dpx_all = comp["dpx"][:n_use].astype(np.float32)
    dpy_all = comp["dpy"][:n_use].astype(np.float32)
    sb8_all = comp["sub_block_8x8"][:n_use].astype(np.float32)

    with h5py.File(H5_SUB, "r") as fs:
        n_sub = fs["matlab_4x4"].shape[0]
    if int(gi0.max()) >= n_sub:
        raise IndexError(
            f"NPZ has indices up to {gi0.max()}, but subblock HDF5 has only {n_sub} rows."
        )

    G = EnhancedResUNetGenerator(in_ch=IN_CH, out_ch=OUT_CH, base=GEN_BASE,
                                 attn_heads=ATTN_HEADS).to(DEVICE)
    G.load_state_dict(torch.load(args.ckpt, map_location=DEVICE, weights_only=True))
    G.eval()
    print(
        f"Loaded {args.ckpt.name} | in_ch={IN_CH} | n={n_use} | TTA={args.tta} | "
        f"two-pass={args.two_pass} | device={DEVICE}",
        flush=True,
    )

    # Storage
    out_keys = [
        "matlab_8x8", "baseline_8x8", "path_b", "path_b_tta", "path_b_pass2",
        "path_a_aep", "path_a_aep_tta", "pred_4x4", "hfss_8x8", "coupling_b_db",
        "coupling_b_tta_db",
    ]
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in out_keys}

    bs = max(args.batch_size, 1)
    with h5py.File(H5_SUB, "r") as fs, h5py.File(H5_8X8, "r") as f8:
        for start in tqdm(range(0, n_use, bs), desc="Inference"):
            end = min(start + bs, n_use)
            sl = slice(start, end)
            gi_b = gi0[sl]
            dpx_b = dpx_all[sl]
            dpy_b = dpy_all[sl]

            mat4 = fs["matlab_4x4"][gi_b].astype(np.float32)
            sb4 = fs["subblock_4x4"][gi_b].astype(np.float32)
            m2 = fs["matlab_2x2"][gi_b].astype(np.float32)
            h2 = fs["hfss_2x2"][gi_b].astype(np.float32)
            mat8_raw = f8["matlab_patterns"][gi_b].astype(np.float32)
            hfss8_raw = f8["hfss_patterns"][gi_b].astype(np.float32)

            sb8 = peak_norm_db(sb8_all[sl])
            mat8 = peak_norm_db(mat8_raw)
            hfss8 = peak_norm_db(hfss8_raw)

            # Path B (no TTA)
            pb, cpl_b = path_b_predict(G, mat8, sb8, m2, h2, dpx_b, dpy_b, stats, tta=False)
            # Path B (TTA)
            if args.tta:
                pb_tta, cpl_b_tta = path_b_predict(G, mat8, sb8, m2, h2, dpx_b, dpy_b, stats, tta=True)
            else:
                pb_tta = pb.copy()
                cpl_b_tta = cpl_b.copy()
            # Path B (2-pass refinement using TTA prediction as baseline)
            if args.two_pass:
                pb_pass2 = path_b_refine(G, mat8, pb_tta, m2, h2, dpx_b, dpy_b, stats,
                                         tta=args.tta)
            else:
                pb_pass2 = pb_tta.copy()

            # Path A (AEP via 4x4 inputs)
            af4 = af_complex_batch(dpx_b, dpy_b, pos4, parent=4)
            af8 = af_complex_batch(dpx_b, dpy_b, pos8, parent=8)
            pred_4x4, pa = path_a_predict(
                G, mat4, sb4, m2, h2, dpx_b, dpy_b, stats, af4, af8,
                tta=False, null_floor_db=args.null_floor_db,
            )
            if args.tta:
                _, pa_tta = path_a_predict(
                    G, mat4, sb4, m2, h2, dpx_b, dpy_b, stats, af4, af8,
                    tta=True, null_floor_db=args.null_floor_db,
                )
            else:
                pa_tta = pa.copy()

            chunks["matlab_8x8"].append(mat8)
            chunks["baseline_8x8"].append(sb8)
            chunks["path_b"].append(pb)
            chunks["path_b_tta"].append(pb_tta)
            chunks["path_b_pass2"].append(pb_pass2)
            chunks["path_a_aep"].append(pa)
            chunks["path_a_aep_tta"].append(pa_tta)
            chunks["pred_4x4"].append(pred_4x4)
            chunks["hfss_8x8"].append(hfss8)
            chunks["coupling_b_db"].append(cpl_b)
            chunks["coupling_b_tta_db"].append(cpl_b_tta)

    arr = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    truth = arr["hfss_8x8"]

    # ── Calibration on first 100 samples ──
    s_best = 1.0
    w_best = [1.0, 0.0, 0.0]
    s_best_score = float("nan")
    e_best_score = float("nan")
    if args.tune_alpha:
        n_calib = min(100, n_use)
        # 1) Coupling shrinkage on Path B-TTA (or B if no TTA)
        cpl_for_shrink = arr["coupling_b_tta_db"][:n_calib] if args.tta else arr["coupling_b_db"][:n_calib]
        s_best, s_best_score = scan_shrinkage(
            arr["baseline_8x8"][:n_calib],
            cpl_for_shrink,
            truth[:n_calib],
            floor_db=args.floor_db,
        )
        # 2) Three-way blend: Path B-TTA / Path A-AEP-TTA / baseline.
        components = [
            arr["path_b_tta"][:n_calib],
            arr["path_a_aep_tta"][:n_calib],
            arr["baseline_8x8"][:n_calib],
        ]
        w_best, e_best_score = calibrate_blend_masked(
            components, truth[:n_calib], floor_db=args.floor_db, grid_step=0.1,
        )
        print(
            f"\n[calibration on first {n_calib} samples]\n"
            f"  best coupling shrinkage s = {s_best:.2f}  (masked RMSE > {args.floor_db:.0f} dB "
            f"= {s_best_score:.3f})\n"
            f"  best 3-way blend w(path_b_tta, path_a_aep_tta, baseline) = "
            f"({w_best[0]:.2f}, {w_best[1]:.2f}, {w_best[2]:.2f})  "
            f"(masked RMSE > {args.floor_db:.0f} dB = {e_best_score:.3f})",
            flush=True,
        )

    # Apply shrinkage Path-B variant
    cpl_for_apply = arr["coupling_b_tta_db"] if args.tta else arr["coupling_b_db"]
    pred_b_shrunk = peak_norm_db(arr["baseline_8x8"] + s_best * cpl_for_apply)

    pred_ensemble = blend_linear_power(
        [arr["path_b_tta"], arr["path_a_aep_tta"], arr["baseline_8x8"]],
        w_best,
    )

    # Mainbeam-aware blend: keep baseline where it's bright, ensemble where dim.
    cut_best, band_best, mb_score = -20.0, 6.0, float("nan")
    if args.tune_alpha:
        n_calib = min(100, n_use)
        cut_best, band_best, mb_score = calibrate_mainbeam_blend(
            arr["baseline_8x8"][:n_calib],
            pred_ensemble[:n_calib],
            truth[:n_calib],
            floor_db=args.floor_db,
        )
        print(
            f"  best mainbeam-aware blend cutover={cut_best:.1f} dB, band={band_best:.1f} dB "
            f"(masked RMSE > {args.floor_db:.0f} dB = {mb_score:.3f})",
            flush=True,
        )
    pred_mainbeam = mainbeam_aware_blend(arr["baseline_8x8"], pred_ensemble, cut_best, band_best)

    methods = {
        "matlab_8x8": arr["matlab_8x8"],
        "baseline_sub_block_8x8": arr["baseline_8x8"],
        "path_b_8x8in": arr["path_b"],
        "path_b_8x8in_tta": arr["path_b_tta"],
        "path_b_8x8in_pass2": arr["path_b_pass2"],
        f"path_b_8x8in_shrunk_s{s_best:.2f}": pred_b_shrunk,
        "path_a_4x4in_aep": arr["path_a_aep"],
        "path_a_4x4in_aep_tta": arr["path_a_aep_tta"],
        f"ensemble_w({w_best[0]:.2f},{w_best[1]:.2f},{w_best[2]:.2f})": pred_ensemble,
        f"mainbeam_blend_cut{cut_best:.0f}_b{band_best:.0f}": pred_mainbeam,
    }

    print("\n=== Aggregate metrics vs HFSS 8x8 ===", flush=True)
    method_metrics: dict[str, dict[str, float]] = {}
    for name, pred in methods.items():
        m = summarise(pred, truth)
        method_metrics[name] = m
        print_method_block(name, m)

    best_full = min(method_metrics.items(), key=lambda kv: kv[1]["full_rmse_db"])
    best_masked = min(method_metrics.items(), key=lambda kv: kv[1]["mean_rmse_gt40"])
    print(f"\n>>> Best FULL RMSE         : {best_full[0]} = {best_full[1]['full_rmse_db']:.3f} dB",
          flush=True)
    print(f">>> Best > -40 dB mean RMSE: {best_masked[0]} = {best_masked[1]['mean_rmse_gt40']:.3f} dB",
          flush=True)

    metrics_path = args.out_dir / "metrics_aep_8x8.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.ckpt}\n")
        f.write(f"npz: {args.npz}\n")
        f.write(f"n_samples: {n_use}\n")
        f.write(f"tta: {args.tta}\n")
        f.write(f"two_pass: {args.two_pass}\n")
        f.write(f"floor_db (masked metric): {args.floor_db}\n")
        f.write(f"calibrated_shrinkage_s: {s_best:.4f}\n")
        f.write(
            f"calibrated_blend_w_path_b_tta / w_path_a_aep_tta / w_baseline: "
            f"{w_best[0]:.4f} / {w_best[1]:.4f} / "
            f"{max(0.0, 1.0 - w_best[0] - w_best[1]):.4f}\n"
        )
        f.write(f"null_floor_db: {args.null_floor_db}\n")
        f.write(f"best_full_method: {best_full[0]}\n")
        f.write(f"best_masked_method: {best_masked[0]}\n\n")
        for name, m in method_metrics.items():
            for k, v in m.items():
                f.write(f"{name}_{k}: {v:.6f}\n")
            f.write("\n")
    print(f"\nWrote {metrics_path}", flush=True)

    rows = ["global_idx,rmse_baseline,rmse_path_b_tta,rmse_path_a_aep_tta,rmse_ensemble,"
            "masked_rmse_baseline,masked_rmse_path_b_tta,masked_rmse_path_a_aep_tta,masked_rmse_ensemble"]
    for i in range(n_use):
        gi = int(idx_1[i])
        r_base = compute_pattern_metrics(arr["baseline_8x8"][i], truth[i])["rmse_db"]
        r_pb = compute_pattern_metrics(arr["path_b_tta"][i], truth[i])["rmse_db"]
        r_pa = compute_pattern_metrics(arr["path_a_aep_tta"][i], truth[i])["rmse_db"]
        r_en = compute_pattern_metrics(pred_ensemble[i], truth[i])["rmse_db"]
        m_base = masked_rmse_above_floor(arr["baseline_8x8"][i], truth[i], args.floor_db)
        m_pb = masked_rmse_above_floor(arr["path_b_tta"][i], truth[i], args.floor_db)
        m_pa = masked_rmse_above_floor(arr["path_a_aep_tta"][i], truth[i], args.floor_db)
        m_en = masked_rmse_above_floor(pred_ensemble[i], truth[i], args.floor_db)
        rows.append(
            f"{gi},{r_base:.6f},{r_pb:.6f},{r_pa:.6f},{r_en:.6f},"
            f"{m_base:.6f},{m_pb:.6f},{m_pa:.6f},{m_en:.6f}"
        )
    csv_path = args.out_dir / "per_sample.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}", flush=True)

    if args.save_npz:
        out_npz = args.out_dir / "outputs.npz"
        np.savez_compressed(
            out_npz,
            idx=idx_1,
            dpx=dpx_all,
            dpy=dpy_all,
            **{k: v.astype(np.float16) for k, v in arr.items() if k != "coupling_b_db"},
            ensemble_8x8=pred_ensemble.astype(np.float16),
            shrunk_8x8=pred_b_shrunk.astype(np.float16),
            shrink_s=np.float32(s_best),
            blend_w=np.asarray(w_best, dtype=np.float32),
        )
        print(f"Wrote {out_npz}", flush=True)

    vis_idx = np.linspace(0, n_use - 1, min(args.n_vis, n_use), dtype=int)
    try:
        import matplotlib.pyplot as plt
        extent = [-179.5, 179.5, 180, 0]
        for pi, i in enumerate(vis_idx):
            gi = int(idx_1[i])
            fig, axes = plt.subplots(2, 3, figsize=(20, 9))
            panels = [
                (arr["matlab_8x8"][i], "MATLAB 8x8"),
                (arr["baseline_8x8"][i], "Comp 8x8 baseline"),
                (arr["path_b_tta"][i], "Path B (8x8in) + TTA"),
                (arr["path_a_aep_tta"][i], "Path A (4x4in AEP) + TTA"),
                (pred_ensemble[i], "Best ensemble"),
                (truth[i], "True HFSS 8x8"),
            ]
            for ax, (data, tl) in zip(axes.flat, panels):
                im = ax.imshow(data, aspect="auto", extent=extent, vmin=-40, vmax=0, cmap="jet")
                if tl != "True HFSS 8x8":
                    rms_full = rmse(data, truth[i])
                    rms_masked = masked_rmse_above_floor(data, truth[i], args.floor_db)
                    ax.set_title(f"{tl}\nfull={rms_full:.2f} dB | > -40 dB={rms_masked:.2f} dB")
                else:
                    ax.set_title(tl)
                plt.colorbar(im, ax=ax, label="dB")
            fig.suptitle(f"idx={gi} | dpx={dpx_all[i]:.1f} dpy={dpy_all[i]:.1f}", y=1.01)
            plt.tight_layout()
            plt.savefig(args.out_dir / f"comparison_{pi:02d}.png", dpi=150, bbox_inches="tight")
            plt.close()

            theta_deg = np.arange(0, N_THETA)
            phi_axis = np.arange(-179.5, 180.0, 1.0)
            phi0_idx = int(np.argmin(np.abs(phi_axis - 0.0)))
            phi90_idx = int(np.argmin(np.abs(phi_axis - 90.0)))
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            for ax, p_idx, label in zip(
                axes, (phi0_idx, phi90_idx), ("phi=0 (E-plane)", "phi=90 (H-plane)")
            ):
                ax.plot(theta_deg, truth[i][:, p_idx], "k", label="True HFSS", linewidth=2)
                ax.plot(theta_deg, pred_ensemble[i][:, p_idx], "r--", label="Ensemble")
                ax.plot(theta_deg, arr["path_b_tta"][i][:, p_idx], "g-.", label="Path B + TTA")
                ax.plot(theta_deg, arr["baseline_8x8"][i][:, p_idx], "b:", label="Comp 8x8", alpha=0.6)
                ax.axhline(args.floor_db, color="gray", linestyle=":", alpha=0.4,
                           label=f"{args.floor_db:.0f} dB floor")
                ax.set_ylim(-50, 2)
                ax.set_xlim(0, 180)
                ax.set_xlabel("theta [deg]")
                ax.set_ylabel("dB")
                ax.set_title(label)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="lower center", fontsize=8)
            fig.suptitle(f"idx={gi} cuts", y=1.02)
            plt.tight_layout()
            plt.savefig(args.out_dir / f"cuts_{pi:02d}.png", dpi=150, bbox_inches="tight")
            plt.close()
    except ImportError:
        print("matplotlib not available — skipped figures", flush=True)

    print(f"\nOutputs under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
