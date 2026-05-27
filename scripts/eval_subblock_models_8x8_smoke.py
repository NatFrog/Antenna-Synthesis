"""
Small fixed 8×8 evaluation: compare resunet_4x4_subblock_coupling_b4 vs stage1_comp5ch.

Uses the same tensor construction and recompositions as:
  - scripts.infer_resunet_4x4_subblock_8x8_compositions (8-ch b0..b3 path)
  - scripts.infer_subblocks_stage1_8x8_compositions (5-ch composition path)

Data sources (per global idx in subblock_compositions.npz):
  - matlab_8x8, hfss_8x8 : processed/antenna_data_8x8.h5
  - sub_block_8x8, dpx, dpy : processed/subblock_compositions.npz
  - matlab_2x2, hfss_2x2 : processed/antenna_data_4x4_subblock.h5 (same global idx)

Pattern metrics are on absolute dB (same convention as scripts/evaluate_resunet_4x4_subblock_coupling.py).

Usage:
    python -m scripts.eval_subblock_models_8x8_smoke --n-samples 8
    python -m scripts.eval_subblock_models_8x8_smoke --n-samples 16 --out-dir results/subblock_8x8_smoke
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

from src.config import BATCH_SIZE, CHECKPOINTS_DIR, DEVICE, N_THETA, PROCESSED_DIR, RESULTS_DIR
from src.evaluation.antenna_metrics import compute_antenna_metric_errors
from src.training.metrics import compute_batch_metrics, compute_pattern_metrics
from scripts.evaluate_resunet_4x4_subblock_coupling import compute_null_metrics
from scripts.infer_resunet_4x4_subblock_8x8_compositions import (
    build_batch_inputs_8ch,
    recompose_hfss_8ch,
)
from scripts.infer_subblocks_stage1_8x8_compositions import (
    build_batch_inputs_5ch_8x8,
    recompose_hfss_8x8,
)
from scripts.train_cgan_2to4_fusion_no_m4 import ATTN_HEADS, GEN_BASE, EnhancedResUNetGenerator
from scripts.train_resunet_4x4_subblock_coupling import NORM_2X2_HFSS as NORM_2X2_HFSS_B4
from scripts.train_resunet_4x4_subblock_coupling import NORM_2X2_MATLAB as NORM_2X2_MATLAB_B4
from scripts.train_resunet_4x4_subblock_coupling import NORM_CPL as NORM_CPL_B4
from scripts.train_resunet_4x4_subblock_coupling import NORM_SUB as NORM_SUB_B4
from scripts.train_resunet_4x4_subblock_coupling import OUT_CH
from scripts.train_subblocks_stage1 import CKPT_DIR as STAGE1_CKPT_DIR
from scripts.train_subblocks_stage1 import IN_CH as STAGE1_IN_CH

COMP_NPZ = PROCESSED_DIR / "subblock_compositions.npz"
H5_8X8 = PROCESSED_DIR / "antenna_data_8x8.h5"
H5_SUB = PROCESSED_DIR / "antenna_data_4x4_subblock.h5"
NORM_8X8 = PROCESSED_DIR / "norm_stats_8x8.npz"

CKPT_B4 = CHECKPOINTS_DIR / "resunet_4x4_subblock_coupling_b4" / "best_generator.pt"
CKPT_STAGE1 = STAGE1_CKPT_DIR / "best_generator.pt"


def _load_norms() -> dict:
    s8 = np.load(NORM_8X8)
    ss = np.load(NORM_SUB_B4)
    sc = np.load(NORM_CPL_B4)
    sm2 = np.load(NORM_2X2_MATLAB_B4)
    sh2 = np.load(NORM_2X2_HFSS_B4)
    return {
        "mean_8": s8["mean"].astype(np.float32),
        "std_8": np.maximum(s8["std"].astype(np.float32), 1e-6),
        "mean_sub": ss["mean"].astype(np.float32),
        "std_sub": np.maximum(ss["std"].astype(np.float32), 1e-6),
        "mean_cpl": sc["mean"].astype(np.float32),
        "std_cpl": np.maximum(sc["std"].astype(np.float32), 1e-6),
        "mean_m2": sm2["mean"].astype(np.float32),
        "std_m2": np.maximum(sm2["std"].astype(np.float32), 1e-6),
        "mean_h2": sh2["mean"].astype(np.float32),
        "std_h2": np.maximum(sh2["std"].astype(np.float32), 1e-6),
    }


def _mean_antenna(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    rows = []
    for i in range(len(pred)):
        try:
            rows.append(compute_antenna_metric_errors(pred[i], true[i]))
        except Exception:
            continue
    if not rows:
        return {}
    out: dict[str, float] = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def _infer_b4(
    G: torch.nn.Module,
    mat8: np.ndarray,
    sb8: np.ndarray,
    m2: np.ndarray,
    h2: np.ndarray,
    dpx: np.ndarray,
    dpy: np.ndarray,
    norms: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_hfss_dB, coupling_pred_dB, baseline_sb_mean_dB)."""
    parts = []
    bs = BATCH_SIZE
    for start in range(0, len(mat8), bs):
        sl = slice(start, min(start + bs, len(mat8)))
        x_np = build_batch_inputs_8ch(
            mat8[sl],
            sb8[sl],
            m2[sl],
            h2[sl],
            dpx[sl],
            dpy[sl],
            norms["mean_8"],
            norms["std_8"],
            norms["mean_sub"],
            norms["std_sub"],
            norms["mean_m2"],
            norms["std_m2"],
            norms["mean_h2"],
            norms["std_h2"],
        )
        xt = torch.from_numpy(x_np).to(DEVICE)
        with torch.no_grad():
            pred_n = G(xt).cpu().numpy()[:, 0]
        pred_hfss, sb_mean = recompose_hfss_8ch(
            x_np, pred_n, norms["mean_sub"], norms["std_sub"], norms["mean_cpl"], norms["std_cpl"]
        )
        cpl_pred = pred_hfss - sb_mean
        parts.append((pred_hfss, cpl_pred, sb_mean))

    pred_hfss = np.concatenate([p[0] for p in parts])
    cpl_pred = np.concatenate([p[1] for p in parts])
    sb_mean = np.concatenate([p[2] for p in parts])
    return pred_hfss, cpl_pred, sb_mean


def _infer_stage1(
    G: torch.nn.Module,
    mat8: np.ndarray,
    sb8: np.ndarray,
    m2: np.ndarray,
    h2: np.ndarray,
    dpx: np.ndarray,
    dpy: np.ndarray,
    norms: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_hfss_dB, coupling_pred_dB, composition_baseline_dB)."""
    parts = []
    bs = BATCH_SIZE
    for start in range(0, len(mat8), bs):
        sl = slice(start, min(start + bs, len(mat8)))
        x_np = build_batch_inputs_5ch_8x8(
            mat8[sl],
            sb8[sl],
            m2[sl],
            h2[sl],
            dpx[sl],
            dpy[sl],
            norms["mean_8"],
            norms["std_8"],
            norms["mean_sub"],
            norms["std_sub"],
            norms["mean_m2"],
            norms["std_m2"],
            norms["mean_h2"],
            norms["std_h2"],
        )
        xt = torch.from_numpy(x_np).to(DEVICE)
        with torch.no_grad():
            pred_n = G(xt).cpu().numpy()[:, 0]
        pred_hfss, comp_db = recompose_hfss_8x8(
            x_np,
            pred_n,
            norms["mean_sub"],
            norms["std_sub"],
            norms["mean_cpl"],
            norms["std_cpl"],
        )
        cpl_pred = pred_hfss - comp_db
        parts.append((pred_hfss, cpl_pred, comp_db))

    pred_hfss = np.concatenate([p[0] for p in parts])
    cpl_pred = np.concatenate([p[1] for p in parts])
    comp_db = np.concatenate([p[2] for p in parts])
    return pred_hfss, cpl_pred, comp_db


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR / "subblock_8x8_smoke",
        help="Write metrics_txt here",
    )
    args = ap.parse_args()

    need = (
        COMP_NPZ,
        H5_8X8,
        H5_SUB,
        NORM_8X8,
        NORM_SUB_B4,
        NORM_CPL_B4,
        NORM_2X2_MATLAB_B4,
        NORM_2X2_HFSS_B4,
        CKPT_B4,
        CKPT_STAGE1,
    )
    for p in need:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    norms = _load_norms()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    comp = np.load(COMP_NPZ)
    n_use = min(args.n_samples, len(comp["idx"]))
    idx_1 = comp["idx"][:n_use].astype(np.int64)
    gi0 = idx_1 - 1

    sb8 = comp["sub_block_8x8"][:n_use].astype(np.float32)
    dpx = comp["dpx"][:n_use].astype(np.float32)
    dpy = comp["dpy"][:n_use].astype(np.float32)

    with h5py.File(H5_8X8, "r") as f8, h5py.File(H5_SUB, "r") as fs:
        mat8 = f8["matlab_patterns"][gi0].astype(np.float32)
        hfss8 = f8["hfss_patterns"][gi0].astype(np.float32)
        n_sub = fs["matlab_2x2"].shape[0]

        m2 = np.zeros((n_use, N_THETA, sb8.shape[-1]), dtype=np.float32)
        h2 = np.zeros_like(m2)
        for i, gi in enumerate(gi0):
            if 0 <= int(gi) < n_sub:
                m2[i] = fs["matlab_2x2"][gi]
                h2[i] = fs["hfss_2x2"][gi]

    Gb = EnhancedResUNetGenerator(in_ch=8, out_ch=OUT_CH, base=GEN_BASE, attn_heads=ATTN_HEADS).to(DEVICE)
    Gb.load_state_dict(torch.load(CKPT_B4, map_location=DEVICE, weights_only=True))
    Gb.eval()

    Gs = EnhancedResUNetGenerator(in_ch=STAGE1_IN_CH, out_ch=OUT_CH, base=GEN_BASE, attn_heads=ATTN_HEADS).to(
        DEVICE
    )
    Gs.load_state_dict(torch.load(CKPT_STAGE1, map_location=DEVICE, weights_only=True))
    Gs.eval()

    pred_b4, cpl_b4, sb_mean_b4 = _infer_b4(Gb, mat8, sb8, m2, h2, dpx, dpy, norms)
    pred_s1, cpl_s1, comp_s1 = _infer_stage1(Gs, mat8, sb8, m2, h2, dpx, dpy, norms)

    true_cpl_b4 = hfss8 - sb_mean_b4
    true_cpl_s1 = hfss8 - comp_s1

    lines: list[str] = []
    lines.append("8x8 subblock model smoke evaluation (correct channel pipelines)")
    lines.append(f"n_samples: {n_use}")
    lines.append(f"global_idx (1-based, first n): {idx_1.tolist()}")
    lines.append(f"checkpoint_b4: {CKPT_B4}")
    lines.append(f"checkpoint_stage1: {CKPT_STAGE1}")
    lines.append("")

    def block(title: str, pred: np.ndarray, cpl_p: np.ndarray, cpl_t: np.ndarray, ref_for_null: np.ndarray):
        lines.append(f"=== {title} ===")
        pm = compute_batch_metrics(pred, hfss8)
        for k, v in pm.items():
            lines.append(f"  hfss_{k}: {v:.6f}")
        cm = compute_batch_metrics(cpl_p, cpl_t)
        for k, v in cm.items():
            lines.append(f"  coupling_{k}: {v:.6f}")
        am = _mean_antenna(pred, hfss8)
        for k, v in am.items():
            lines.append(f"  antenna_{k}: {v:.6f}")
        nm = compute_null_metrics(pred, hfss8, ref_for_null)
        for k, v in nm.items():
            lines.append(f"  null_ref_{k}: {v:.6f}")
        lines.append("")

    block("b4 (8-ch, mean(b0..b3) baseline)", pred_b4, cpl_b4, true_cpl_b4, mat8)
    block("stage1_comp5ch (5-ch, denorm composition baseline)", pred_s1, cpl_s1, true_cpl_s1, mat8)

    lines.append("=== Baselines (HFSS 8x8 target) ===")
    bl_comp = compute_batch_metrics(sb8, hfss8)
    bl_mat = compute_batch_metrics(mat8, hfss8)
    bl_sbmean = compute_batch_metrics(sb_mean_b4, hfss8)
    for k, v in bl_comp.items():
        lines.append(f"  baseline_npz_sub_block_8x8_raw_{k}: {v:.6f}")
    for k, v in bl_sbmean.items():
        lines.append(f"  baseline_b4_recomposed_sb_mean_{k}: {v:.6f}")
    for k, v in bl_mat.items():
        lines.append(f"  baseline_matlab_8x8_{k}: {v:.6f}")
    lines.append("")

    lines.append("=== Per-sample HFSS RMSE (dB) ===")
    lines.append("idx1,label,b4_rmse,stage1_rmse,matlab_rmse,npz_comp_rmse")
    for j in range(n_use):
        gi = int(idx_1[j])
        label = f"s{gi:05d}"
        lines.append(
            f"{gi},{label},"
            f"{compute_pattern_metrics(pred_b4[j], hfss8[j])['rmse_db']:.4f},"
            f"{compute_pattern_metrics(pred_s1[j], hfss8[j])['rmse_db']:.4f},"
            f"{compute_pattern_metrics(mat8[j], hfss8[j])['rmse_db']:.4f},"
            f"{compute_pattern_metrics(sb8[j], hfss8[j])['rmse_db']:.4f}"
        )

    out_path = args.out_dir / f"metrics_8x8_smoke_n{n_use}.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
