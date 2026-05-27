"""
Compare bootstrap Phase A vs Phase C on held-out **test** splits at 4x4, 8x8, and 16x16.

- 4x4: ``split_indices_2to4['test']``, targets ``hfss_4x4``, anchor ``hfss_pred_2x2``.
- 8x8: ``split_indices_4to8['test']``, targets ``hfss_8x8``, anchor ``hfss_pred_4x4``.
- 16x16: second half of sorted 4to8 test = ``test_16x16`` (same protocol as
  ``evaluate_residual_multiscale``), anchor ``hfss_pred_8x8``.

Usage::
    python -m scripts.evaluate_bootstrap_phase_table
    python -m scripts.evaluate_bootstrap_phase_table --out results/bootstrap_phase_table.md
    python -m scripts.evaluate_bootstrap_phase_table \\
        --checkpoint checkpoints/residual_physics_2to4_only/best_generator.pt \\
        --out results/residual_physics_2to4_table.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
import torch
from tqdm import tqdm

from scipy.ndimage import minimum_filter

from src.config import BATCH_SIZE, DEVICE, NULL_THRESHOLD_DB, PROCESSED_DIR
from src.evaluation.antenna_metrics import compute_antenna_metric_errors
from src.training.metrics import compute_batch_hfss_region_metrics, compute_batch_metrics
from scripts.train_cgan_2to4_fusion_no_m4 import (
    ATTN_HEADS,
    GEN_BASE,
    EnhancedResUNetGenerator,
)
from scripts.train_residual_multiscale import (
    AMP_DEVICE_TYPE,
    AMP_DTYPE,
    ResidualReconLoss,
    USE_AMP,
)

H5_2TO4 = PROCESSED_DIR / "antenna_data_2to4.h5"
H5_4TO8 = PROCESSED_DIR / "antenna_data_4to8.h5"
H5_16X16 = PROCESSED_DIR / "antenna_data_8to16_subarray.h5"
M16_TEST = PROCESSED_DIR / "matlab_16x16_test.npz"
NORM_COMBINED = PROCESSED_DIR / "norm_stats_matlab_combined.npz"
SPLITS_2TO4 = PROCESSED_DIR / "split_indices_2to4.npz"
SPLITS_4TO8 = PROCESSED_DIR / "split_indices_4to8.npz"
CKPT_A = PROJECT_ROOT / "checkpoints/residual_bootstrap_phase_a/best_generator.pt"
CKPT_C = PROJECT_ROOT / "checkpoints/residual_bootstrap_phase_c/best_generator.pt"


def _synth_scale(
    G: torch.nn.Module,
    recon_loss: ResidualReconLoss,
    mat: np.ndarray,
    anchor: np.ndarray,
    meta: np.ndarray,
    mean_mat: np.ndarray,
    std_mat: np.ndarray,
    scale_n: int,
) -> np.ndarray:
    """Batched residual forward at scale ``scale_n`` in ``{4, 8, 16}``."""
    n = len(mat)
    preds = np.zeros_like(mat, dtype=np.float32)
    mean_b = mean_mat.astype(np.float32)
    std_b = np.maximum(std_mat.astype(np.float32), 1e-6)
    tok_v = float(scale_n) / 16.0
    G.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n, BATCH_SIZE), desc=f"N={scale_n}x{scale_n}", leave=False):
            j = min(i + BATCH_SIZE, n)
            mat_batch = mat[i:j]
            anchor_batch = anchor[i:j]
            mat_n = (mat_batch - mean_b[None]) / std_b[None]
            anchor_n = (anchor_batch - mean_b[None]) / std_b[None]
            dpx = (meta[i:j, 0:1, None] / 180.0).astype(np.float32) * np.ones_like(mat_n)
            dpy = (meta[i:j, 1:2, None] / 180.0).astype(np.float32) * np.ones_like(mat_n)
            stk = np.full_like(mat_n, fill_value=tok_v, dtype=np.float32)
            x = np.stack([mat_n, dpx, dpy, stk, anchor_n], axis=1)
            xt = torch.from_numpy(x).to(DEVICE)
            mt = torch.from_numpy(mat_batch[:, None]).to(DEVICE)
            with torch.amp.autocast(
                device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP
            ):
                delta_n = G(xt)
                pred_dB = recon_loss.compose(delta_n, mt)
            preds[i:j] = pred_dB.float().cpu().numpy()[:, 0]
    return preds


def _build_metrics(
    preds: np.ndarray,
    truth_hfss: np.ndarray,
    matlab: np.ndarray,
) -> dict[str, float]:
    """Pattern + antenna + null metrics (same style as ``evaluate_residual_multiscale.report``)."""
    pm = compute_batch_metrics(preds, truth_hfss)
    ae = []
    for i in range(len(preds)):
        try:
            ae.append(compute_antenna_metric_errors(preds[i], truth_hfss[i]))
        except Exception:
            pass
    avg_ant: dict[str, float] = {}
    if ae:
        for key in ae[0]:
            vals = [e[key] for e in ae if not np.isnan(e[key])]
            avg_ant[key] = float(np.mean(vals)) if vals else float("nan")

    null_rmses, depth_errs = [], []
    nfa = nft = 0
    for i in range(len(preds)):
        mat = matlab[i]
        mask_null = mat < (mat.max() + NULL_THRESHOLD_DB)
        if mask_null.sum() > 0:
            null_rmses.append(
                float(
                    np.sqrt(np.mean((preds[i][mask_null] - truth_hfss[i][mask_null]) ** 2))
                )
            )
            pf = preds[i][mask_null] - mat[mask_null]
            tf = truth_hfss[i][mask_null] - mat[mask_null]
            nfa += int((np.abs(pf - tf) < 2.0).sum())
            nft += int(mask_null.sum())
        lm = minimum_filter(mat, size=5)
        ilm = (mat == lm) & mask_null
        if ilm.sum() > 0:
            mp = np.argwhere(ilm)
            mv = mat[ilm]
            di = np.argsort(mv)[:10]
            for k in di:
                t, p = mp[k]
                depth_errs.append(abs(preds[i][t, p] - truth_hfss[i][t, p]))
    nm = {
        "rmse_at_nulls_db": float(np.mean(null_rmses)) if null_rmses else float("nan"),
        "depth_error_db": float(np.mean(depth_errs)) if depth_errs else float("nan"),
        "fill_accuracy_pct": float(nfa / max(nft, 1) * 100),
    }
    paper = compute_batch_hfss_region_metrics(preds, truth_hfss, db_floors=(-40.0, -50.0))
    out: dict[str, float] = {f"pattern_{k}": v for k, v in pm.items()}
    out.update({f"antenna_{k}": v for k, v in avg_ant.items()})
    out.update({f"null_{k}": v for k, v in nm.items()})
    out.update(paper)
    return out


def _load_g(ckpt: Path, sigma_res: float) -> tuple:
    G = EnhancedResUNetGenerator(
        in_ch=5, out_ch=1, base=GEN_BASE, attn_heads=ATTN_HEADS
    ).to(DEVICE)
    G.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    recon = ResidualReconLoss(sigma_res=sigma_res).to(DEVICE)
    return G, recon


def _fmt_metric(x: float) -> str:
    return f"{x:.4f}" if np.isfinite(x) else "nan"


# Internal metric key -> readable row label (shown in markdown tables)
METRIC_LABELS: dict[str, str] = {
    "pattern_rmse_db": "RMSE (full pattern, dB)",
    "pattern_mae_db": "MAE (full pattern, dB)",
    "pattern_pearson_r": "Pearson r (full pattern)",
    "pattern_ssim": "SSIM (full pattern)",
    "paper_region_rmse_db_neg40_mean": "RMSE where HFSS > -40 dB (mean over test patterns, dB)",
    "paper_region_mae_db_neg40_mean": "MAE where HFSS > -40 dB (mean over test patterns, dB)",
    "paper_region_frac_sphere_neg40": "Fraction of sphere with HFSS > -40 dB",
    "paper_region_rmse_db_neg50_mean": "RMSE where HFSS > -50 dB (mean over test patterns, dB)",
    "paper_region_mae_db_neg50_mean": "MAE where HFSS > -50 dB (mean over test patterns, dB)",
    "paper_region_frac_sphere_neg50": "Fraction of sphere with HFSS > -50 dB",
    "antenna_peak_gain_error_db": "Peak gain error (dB; ~0 after max-norm)",
    "antenna_peak_direction_error_deg": "Peak direction error (deg)",
    "antenna_hpbw_e_error_pct": "E-plane HPBW error (%)",
    "antenna_hpbw_h_error_pct": "H-plane HPBW error (%)",
    "antenna_sll_error_db": "First sidelobe level error (dB)",
    "null_rmse_at_nulls_db": "RMSE in nulls (Matlab < peak - 20 dB, dB)",
    "null_depth_error_db": "Null depth error (deepest nulls, dB)",
    "null_fill_accuracy_pct": "Null fill accuracy (%, correction within 2 dB)",
}

METRIC_ROW_ORDER: list[str] = [
    "pattern_rmse_db",
    "pattern_mae_db",
    "pattern_pearson_r",
    "pattern_ssim",
    "paper_region_rmse_db_neg40_mean",
    "paper_region_mae_db_neg40_mean",
    "paper_region_frac_sphere_neg40",
    "paper_region_rmse_db_neg50_mean",
    "paper_region_mae_db_neg50_mean",
    "paper_region_frac_sphere_neg50",
    "antenna_peak_direction_error_deg",
    "antenna_hpbw_e_error_pct",
    "antenna_hpbw_h_error_pct",
    "antenna_sll_error_db",
    "null_rmse_at_nulls_db",
    "null_depth_error_db",
    "null_fill_accuracy_pct",
]


def _label(key: str) -> str:
    return METRIC_LABELS.get(key, key)


def _parse_scales(spec: str) -> list[tuple[str, str]]:
    """Return (short header, internal scale label) for each requested scale."""
    all_scales = {
        "4": ("4x4", "4x4 (2to4 test)"),
        "4x4": ("4x4", "4x4 (2to4 test)"),
        "8": ("8x8", "8x8 (4to8 test)"),
        "8x8": ("8x8", "8x8 (4to8 test)"),
        "16": ("16x16", "16x16 (test_16x16)"),
        "16x16": ("16x16", "16x16 (test_16x16)"),
    }
    if spec.strip().lower() == "all":
        return [all_scales["4x4"], all_scales["8x8"], all_scales["16x16"]]
    parts = [p.strip().lower() for p in spec.split(",")]
    out = []
    for p in parts:
        if p not in all_scales:
            raise ValueError(f"Unknown scale {p!r}; use 4x4, 8x8, 16x16, or all")
        pair = all_scales[p]
        if pair not in out:
            out.append(pair)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional path to write the markdown table",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Single EMA checkpoint. Empty = bootstrap A vs C.",
    )
    ap.add_argument(
        "--wide",
        action="store_true",
        help="One table: readable metric rows; columns = scales from --scales (needs 2+ scales).",
    )
    ap.add_argument(
        "--scales",
        type=str,
        default="16x16",
        help="Comma-separated scales to evaluate: 4x4, 8x8, 16x16, or all (default: 16x16 only).",
    )
    args = ap.parse_args()
    scale_cols = _parse_scales(args.scales)
    if not scale_cols:
        scale_cols = _parse_scales("16x16")

    req_data = (
        H5_2TO4,
        H5_4TO8,
        H5_16X16,
        M16_TEST,
        NORM_COMBINED,
        SPLITS_2TO4,
        SPLITS_4TO8,
    )
    for p in req_data:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    s = np.load(NORM_COMBINED)
    mean_mat = s["mean"].astype(np.float32)
    std_mat = s["std"].astype(np.float32)
    sigma_res = float(s["residual_std"])

    sp24 = np.load(SPLITS_2TO4)
    sp48 = np.load(SPLITS_4TO8)
    te4 = np.sort(sp24["test"].astype(np.int64))
    test48 = np.sort(sp48["test"].astype(np.int64))
    test16_idx = test48[100:]

    with h5py.File(H5_2TO4, "r") as f:
        mat4 = f["matlab_4x4"][te4].astype(np.float32)
        hf4 = f["hfss_4x4"][te4].astype(np.float32)
        a4 = f["hfss_pred_2x2"][te4].astype(np.float32)
        meta4 = f["metadata"][te4].astype(np.float32)

    with h5py.File(H5_4TO8, "r") as f:
        mat8 = f["matlab_8x8"][test48].astype(np.float32)
        hf8 = f["hfss_8x8"][test48].astype(np.float32)
        a8 = f["hfss_pred_4x4"][test48].astype(np.float32)
        meta8 = f["metadata"][test48].astype(np.float32)

    pack = np.load(M16_TEST)
    if not np.array_equal(pack["test_idx"], test48):
        raise RuntimeError("matlab_16x16 test_idx mismatch")
    mat16_all = pack["arr"].astype(np.float32)
    test_mat16 = mat16_all[100:]
    with h5py.File(H5_16X16, "r") as f:
        hf16 = f["hfss_16x16"][test16_idx].astype(np.float32)
        a16 = f["hfss_pred_8x8"][test16_idx].astype(np.float32)
    with h5py.File(H5_4TO8, "r") as f:
        meta16 = f["metadata"][test16_idx].astype(np.float32)

    all_keys = METRIC_ROW_ORDER

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_absolute():
            ckpt = PROJECT_ROOT / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        print(f"\n=== Evaluating {ckpt} ===", flush=True)
        G, recon = _load_g(ckpt, sigma_res)
        one: dict[str, dict[str, float]] = {}
        scale_notes: list[str] = []
        for short, scale_label in scale_cols:
            if scale_label.startswith("4x4"):
                pred = _synth_scale(G, recon, mat4, a4, meta4, mean_mat, std_mat, 4)
                one[scale_label] = _build_metrics(pred, hf4, mat4)
                scale_notes.append(f"- **4x4**: n={len(te4)} (2to4 test), truth `hfss_4x4`, anchor `hfss_pred_2x2`")
            elif scale_label.startswith("8x8"):
                pred = _synth_scale(G, recon, mat8, a8, meta8, mean_mat, std_mat, 8)
                one[scale_label] = _build_metrics(pred, hf8, mat8)
                scale_notes.append(
                    f"- **8x8**: n={len(test48)} (4to8 test), truth `hfss_8x8`, anchor `hfss_pred_4x4`"
                )
            else:
                pred = _synth_scale(G, recon, test_mat16, a16, meta16, mean_mat, std_mat, 16)
                one[scale_label] = _build_metrics(pred, hf16, test_mat16)
                scale_notes.append(
                    f"- **16x16**: n={len(test16_idx)} (held-out test_16x16), truth `hfss_16x16`, "
                    f"anchor `hfss_pred_8x8`, analytical `matlab_16x16`"
                )
        lines = [
            "# Residual model evaluation (held-out test)\n",
            f"- Checkpoint: `{ckpt.relative_to(PROJECT_ROOT)}`\n",
            *scale_notes,
            "- **Region metrics**: RMSE/MAE only where ground-truth HFSS exceeds the dB floor "
            "(main beam + sidelobes, not deep nulls).\n",
            "- **Null metrics**: pixels where Matlab pattern is more than 20 dB below its peak.\n",
        ]
        use_wide = args.wide and len(scale_cols) > 1
        if use_wide:
            hdr = " | ".join(short for short, _ in scale_cols)
            sep = " | ".join("---" for _ in scale_cols)
            lines.append(f"\n| Metric | {hdr} |")
            lines.append(f"|--------|{sep}|")
            for k in all_keys:
                cells = [_fmt_metric(one[label].get(k, float("nan"))) for _, label in scale_cols]
                lines.append(f"| {_label(k)} | {' | '.join(cells)} |")
        else:
            for short, scale_label in scale_cols:
                title = f"16x16 extrapolation (test_16x16)" if "16x16" in scale_label else scale_label
                lines.append(f"\n## {title}\n")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for k in all_keys:
                    v = one[scale_label].get(k, float("nan"))
                    lines.append(f"| {_label(k)} | {_fmt_metric(v)} |")
        md = "\n".join(lines)
        print(md, flush=True)
        if args.out:
            out_p = Path(args.out)
            if not out_p.is_absolute():
                out_p = PROJECT_ROOT / out_p
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(md, encoding="utf-8")
            print(f"\nWrote {out_p}", flush=True)
        return

    for p in (CKPT_A, CKPT_C):
        if not Path(p).exists():
            raise FileNotFoundError(p)

    results: dict[str, dict[str, dict[str, float]]] = {
        label: {"Phase A": {}, "Phase C": {}} for _, label in scale_cols
    }

    for ckpt_name, ckpt in [("Phase A", CKPT_A), ("Phase C", CKPT_C)]:
        print(f"\n=== Evaluating {ckpt_name}: {ckpt} ===", flush=True)
        G, recon = _load_g(ckpt, sigma_res)
        for _short, scale_label in scale_cols:
            if scale_label.startswith("4x4"):
                pred = _synth_scale(G, recon, mat4, a4, meta4, mean_mat, std_mat, 4)
                results[scale_label][ckpt_name] = _build_metrics(pred, hf4, mat4)
            elif scale_label.startswith("8x8"):
                pred = _synth_scale(G, recon, mat8, a8, meta8, mean_mat, std_mat, 8)
                results[scale_label][ckpt_name] = _build_metrics(pred, hf8, mat8)
            else:
                pred = _synth_scale(G, recon, test_mat16, a16, meta16, mean_mat, std_mat, 16)
                results[scale_label][ckpt_name] = _build_metrics(pred, hf16, test_mat16)

    lines = [
        "# Bootstrap Phase A vs Phase C (held-out test splits)\n",
        f"- Checkpoints: `{CKPT_A.relative_to(PROJECT_ROOT)}`, `{CKPT_C.relative_to(PROJECT_ROOT)}`\n",
        f"- 4x4: n={len(te4)} (2to4 test), 8x8: n={len(test48)} (4to8 test), "
        f"16x16: n={len(test16_idx)} (honest test_16x16)\n",
    ]

    bootstrap_scales = [label for _, label in scale_cols if label in results]
    if not bootstrap_scales:
        bootstrap_scales = list(results.keys())

    for scale_label in bootstrap_scales:
        title = "16x16 extrapolation (test_16x16)" if "16x16" in scale_label else scale_label
        lines.append(f"\n## {title}\n")
        lines.append("| Metric | Phase A | Phase C |")
        lines.append("|--------|---------|---------|")
        for k in all_keys:
            va = results[scale_label]["Phase A"].get(k, float("nan"))
            vc = results[scale_label]["Phase C"].get(k, float("nan"))
            lines.append(f"| {_label(k)} | {_fmt_metric(va)} | {_fmt_metric(vc)} |")

    md = "\n".join(lines)
    print(md, flush=True)
    if args.out:
        out_p = Path(args.out)
        if not out_p.is_absolute():
            out_p = PROJECT_ROOT / out_p
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md, encoding="utf-8")
        print(f"\nWrote {out_p}", flush=True)


if __name__ == "__main__":
    main()
