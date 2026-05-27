"""
Pooled pixel-level error report for multi-scale residual cGAN checkpoints
(e.g. residual_bootstrap_phase_c).

Reads the same 16×16 protocol as scripts/evaluate_residual_multiscale.py (val_16x16 /
test_16x16 split from 4to8 test indices, anchor from antenna_data_8to16_subarray.h5).

Also reports **model-generated 8×8** pooled metrics on the **same indices** used for the
16×16 script (subset of sorted 4to8 ``test``):

  - Forward this checkpoint at **N=8**: ``matlab_8x8`` + anchor ``hfss_pred_4x4``
    + ``scale_token = 8/16`` + steering (matches ``evaluate_bootstrap_phase_table._synth_scale``).
  - Truth: ``hfss_8x8`` from ``antenna_data_4to8.h5``.
  - Optional baseline: ``matlab_8x8`` vs ``hfss_8x8`` on the test half.

For each chosen split, computes over **all selected pixels** (flattened):

  RMSE, MAE, MedAE, p90(|error|), p99(|error|), max(|error|), Pearson r, R², bias.

Usage:

    python -m scripts.report_residual_checkpoint_pixel_metrics
    python -m scripts.report_residual_checkpoint_pixel_metrics ^
        --checkpoint checkpoints/residual_bootstrap_phase_c/best_generator.pt ^
        --out-dir results/residual_bootstrap_phase_c ^
        --splits val test baseline_test

Skip 8x8 block (16x16 only):

    python -m scripts.report_residual_checkpoint_pixel_metrics --skip-8x8

Metric definitions match the extended error report spec (MedAE / pxx / pooled R² / bias).
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

from src.config import DEVICE
from scripts.evaluate_bootstrap_phase_table import _synth_scale
from scripts.evaluate_residual_multiscale import (
    H5_16X16,
    H5_4TO8,
    M16_TEST,
    NORM_COMBINED,
    SPLITS_4TO8,
    synth_at_16x16,
)
from scripts.train_cgan_2to4_fusion_no_m4 import EnhancedResUNetGenerator, GEN_BASE, ATTN_HEADS
from scripts.train_residual_multiscale import ResidualReconLoss

DEFAULT_CKPT = PROJECT_ROOT / "checkpoints/residual_bootstrap_phase_c/best_generator.pt"
DEFAULT_OUT = PROJECT_ROOT / "results/residual_bootstrap_phase_c"


def pooled_pixel_metric_bundle(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """
    pred, true: (N, H, W), same shape, dB.

    All summaries use the same multiset of residuals (flattened batch × H × W).
    """
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch pred {pred.shape} vs true {true.shape}")

    diff = (pred.astype(np.float64) - true.astype(np.float64)).ravel()
    abs_e = np.abs(diff)
    flat_p = pred.astype(np.float64).ravel()
    flat_t = true.astype(np.float64).ravel()

    sse = float(np.sum(diff**2))
    mean_t = float(flat_t.mean())
    ss_tot = float(np.sum((flat_t - mean_t) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 1e-30 else float("nan")

    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(abs_e))
    medae = float(np.median(abs_e))
    p90 = float(np.percentile(abs_e, 90))
    p99 = float(np.percentile(abs_e, 99))
    max_abs = float(np.max(abs_e))
    bias = float(np.mean(diff))

    if flat_p.std() < 1e-18 or flat_t.std() < 1e-18:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(flat_p, flat_t)[0, 1])

    n_pix = diff.size
    return {
        "n_patterns": float(pred.shape[0]),
        "n_pixels": float(n_pix),
        "rmse_db": rmse,
        "mae_db": mae,
        "medae_db": medae,
        "p90_abs_err_db": p90,
        "p99_abs_err_db": p99,
        "max_abs_err_db": max_abs,
        "pearson_r": pearson,
        "r2": r2,
        "bias_db": bias,
        "truth_mean_db": mean_t,
        "mse_db2": sse / max(n_pix, 1),
    }


def _format_section(title: str, m: dict[str, float]) -> str:
    order = (
        ("n_patterns", "n_patterns"),
        ("n_pixels", "n_pixels"),
        ("rmse_db", "RMSE"),
        ("mae_db", "MAE"),
        ("medae_db", "MedAE"),
        ("p90_abs_err_db", "p90_|err|"),
        ("p99_abs_err_db", "p99_|err|"),
        ("max_abs_err_db", "max_|err|"),
        ("pearson_r", "Pearson_r"),
        ("r2", "R2"),
        ("bias_db", "bias_mean(pred-true)"),
        ("truth_mean_db", "truth_mean_db"),
        ("mse_db2", "MSE"),
    )
    lines = [f"--- {title} ---"]
    for key, label in order:
        v = m.get(key)
        if v is None:
            continue
        if key in ("n_patterns", "n_pixels"):
            lines.append(f"  {label}: {int(v)}")
        else:
            lines.append(f"  {label}: {v:.6f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test", "baseline_test"],
        choices=("val", "test", "baseline_test"),
        help="val = model on first 100; test = model on honest 100; baseline_test = matlab vs hfss test",
    )
    ap.add_argument(
        "--skip-8x8",
        action="store_true",
        help="Omit 8x8 pooled metrics (checkpoint forward at N=8 vs hfss_8x8).",
    )
    args = ap.parse_args()

    include_8x8 = not args.skip_8x8

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    s = np.load(NORM_COMBINED)
    mean_mat = s["mean"].astype(np.float32)
    std_mat = s["std"].astype(np.float32)
    sigma_res = float(s["residual_std"])

    sp = np.load(SPLITS_4TO8)
    test48 = np.sort(sp["test"].astype(np.int64))
    val16_idx = test48[:100]
    test16_idx = test48[100:]

    pack = np.load(M16_TEST)
    if not np.array_equal(pack["test_idx"], test48):
        raise RuntimeError("matlab_16x16_test.npz test_idx mismatch")
    mat16_all = pack["arr"].astype(np.float32)
    val_mat16 = mat16_all[:100]
    test_mat16 = mat16_all[100:]

    with h5py.File(H5_16X16, "r") as f16:
        val_hfss = f16["hfss_16x16"][val16_idx].astype(np.float32)
        test_hfss = f16["hfss_16x16"][test16_idx].astype(np.float32)
        val_hfss_pred_8x8 = f16["hfss_pred_8x8"][val16_idx].astype(np.float32)
        test_hfss_pred_8x8 = f16["hfss_pred_8x8"][test16_idx].astype(np.float32)

    val_mat8 = val_hfss8 = val_anchor4 = test_mat8 = test_hfss8 = test_anchor4 = None
    with h5py.File(H5_4TO8, "r") as f48:
        val_meta = f48["metadata"][val16_idx].astype(np.float32)
        test_meta = f48["metadata"][test16_idx].astype(np.float32)
        if include_8x8:
            val_mat8 = f48["matlab_8x8"][val16_idx].astype(np.float32)
            test_mat8 = f48["matlab_8x8"][test16_idx].astype(np.float32)
            val_hfss8 = f48["hfss_8x8"][val16_idx].astype(np.float32)
            test_hfss8 = f48["hfss_8x8"][test16_idx].astype(np.float32)
            val_anchor4 = f48["hfss_pred_4x4"][val16_idx].astype(np.float32)
            test_anchor4 = f48["hfss_pred_4x4"][test16_idx].astype(np.float32)

    if include_8x8:
        for lab, ma, anc, tg in (
            ("val N=8", val_mat8, val_anchor4, val_hfss8),
            ("test N=8", test_mat8, test_anchor4, test_hfss8),
        ):
            if ma.shape != anc.shape or ma.shape != tg.shape:
                raise ValueError(f"{lab}: shape mismatch mat {ma.shape} anchor {anc.shape} hfss {tg.shape}")
        if "baseline_test" in args.splits and test_mat8.shape != test_hfss8.shape:
            raise ValueError(
                f"baseline test 8x8 shape mismatch matlab {test_mat8.shape} vs hfss {test_hfss8.shape}"
            )

    G = EnhancedResUNetGenerator(in_ch=5, out_ch=1, base=GEN_BASE, attn_heads=ATTN_HEADS).to(DEVICE)
    state = torch.load(args.checkpoint, map_location=DEVICE)
    G.load_state_dict(state)
    G.eval()
    recon_loss = ResidualReconLoss(sigma_res=sigma_res).to(DEVICE)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    header = [
        "Pooled pixel metrics (flattened residuals across patterns × θ × φ)",
        f"checkpoint: {args.checkpoint.resolve()}",
        f"sigma_residual: {sigma_res:.6f} dB",
        "",
    ]

    if "val" in args.splits:
        preds = synth_at_16x16(
            G, recon_loss, val_mat16, val_hfss_pred_8x8, val_hfss, val_meta, mean_mat, std_mat
        )
        m = pooled_pixel_metric_bundle(preds, val_hfss)
        txt = _format_section("VAL_16X16 - model preds vs hfss_16x16 (pooled pixels)", m)
        print("\n" + txt + "\n", flush=True)
        sections.append(txt)

    if "test" in args.splits:
        preds = synth_at_16x16(
            G, recon_loss, test_mat16, test_hfss_pred_8x8, test_hfss, test_meta, mean_mat, std_mat
        )
        m = pooled_pixel_metric_bundle(preds, test_hfss)
        txt = _format_section("TEST_16X16 - model preds vs hfss_16x16 (pooled pixels)", m)
        print("\n" + txt + "\n", flush=True)
        sections.append(txt)

    if "baseline_test" in args.splits:
        m = pooled_pixel_metric_bundle(test_mat16, test_hfss)
        txt = _format_section("BASELINE_TEST - matlab_16x16 vs hfss (pooled pixels)", m)
        print("\n" + txt + "\n", flush=True)
        sections.append(txt)

    sections_8x8: list[str] = []
    if include_8x8:
        sections_8x8.extend(
            [
                "",
                "# " + "=" * 72,
                "# 8x8 FROM CHECKPOINT (N=8 forward)",
                "# pred = G(matlab_8x8_n, dphase, scale_tok=8/16, hfss_pred_4x4_n) composed",
                "#      via ResidualReconLoss (same as evaluate_bootstrap_phase_table._synth_scale).",
                "# true_hfss = hfss_8x8 (antenna_data_4to8.h5), same indices as val_16x16 / test_16x16.",
                "# " + "=" * 72,
                "",
            ]
        )
        if "val" in args.splits:
            pred8 = _synth_scale(
                G, recon_loss, val_mat8, val_anchor4, val_meta, mean_mat, std_mat, scale_n=8
            )
            m8 = pooled_pixel_metric_bundle(pred8, val_hfss8)
            txt = _format_section(
                "VAL_INDICES - checkpoint N=8 output vs hfss_8x8 (pooled pixels)", m8
            )
            print("\n" + txt + "\n", flush=True)
            sections_8x8.append(txt)
        if "test" in args.splits:
            pred8 = _synth_scale(
                G, recon_loss, test_mat8, test_anchor4, test_meta, mean_mat, std_mat, scale_n=8
            )
            m8 = pooled_pixel_metric_bundle(pred8, test_hfss8)
            txt = _format_section(
                "TEST_INDICES - checkpoint N=8 output vs hfss_8x8 (pooled pixels)", m8
            )
            print("\n" + txt + "\n", flush=True)
            sections_8x8.append(txt)
        if "baseline_test" in args.splits:
            m8 = pooled_pixel_metric_bundle(test_mat8, test_hfss8)
            txt = _format_section(
                "BASELINE_TEST - matlab_8x8 vs hfss_8x8 (pooled pixels)", m8
            )
            print("\n" + txt + "\n", flush=True)
            sections_8x8.append(txt)

    out_path = args.out_dir / "metrics_pixel_pooled.txt"
    out_path.write_text("\n".join(header + sections + sections_8x8) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
