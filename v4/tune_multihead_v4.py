"""
Hyperparameter tuning runner for multi-head v4.

Does **not** modify ``train_stage1_6x6_multihead_v4.py`` or the eval scripts.
Runs Tier-1 inference sweeps on an existing checkpoint and optional Tier-2
training trials via ``scripts.train_multihead_v4_trial``.

Usage:
    python -m scripts.tune_multihead_v4 --list-presets
    python -m scripts.tune_multihead_v4 --preset inference-default
    python -m scripts.tune_multihead_v4 --preset alpha-main-cap
    python -m scripts.tune_multihead_v4 --mode train --preset mainbeam-focus --init-v2
    python -m scripts.tune_multihead_v4 --mode all --preset mainbeam-focus --init-v2
    python -m scripts.tune_multihead_v4 --preset inference-default --ckpt checkpoints/stage1_6x6_multihead_v4/best_generator.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from scripts.evaluate_dual_residual_zeroshot import (
    _gap_stats,
    _load_phase_aware_scale,
    _pixel_stats,
)
from scripts.eval_multihead_v4 import run_inference as run_inference_8x8
from scripts.eval_multihead_v4_16x16 import run_inference as run_inference_16x16
from scripts.train_stage1_6x6_multihead_v2 import extend_stats_with_coupling_gap
from scripts.train_cgan_2to4_fusion_no_m4 import ATTN_HEADS, GEN_BASE
from src.config import CHECKPOINTS_DIR, DEVICE, PROCESSED_DIR
from src.evaluation.comprehensive_metrics import evaluate_predictions
from src.evaluation.zeroshot_varalpha_report import _masked_point_metrics
from src.models.multihead_v4 import (
    FocusedRegionalMultiHeadResUNet,
    TOP_K_NULLS,
    fit_ls_alpha,
    shield_main_beam_residual,
)
from src.models.multihead_v5 import apply_regional_alpha, fit_regional_ls_alphas

DEFAULT_CKPT = CHECKPOINTS_DIR / "stage1_6x6_multihead_v4" / "best_generator.pt"
NORM_6X6 = PROCESSED_DIR / "norm_stats_stage1_6x6.npz"
SPLIT_6X6 = PROCESSED_DIR / "split_indices_stage1_6x6.npz"
PA_8 = PROCESSED_DIR / "phase_aware_8x8_compose.npz"
PA_16 = PROCESSED_DIR / "phase_aware_16x16_compose.npz"
RESULTS_ROOT = PROJECT_ROOT / "results" / "tuning_v4"


@dataclass
class InferenceTrial:
    name: str
    scale: int = 8
    regional: bool = True
    shield: bool = True
    no_alpha: bool = False
    alpha_main_max: float | None = None


@dataclass
class TrainTrial:
    name: str
    lambda_mainbeam: float = 55.0
    lambda_composed_null: float = 24.0
    val_weight_main: float = 0.30
    val_weight_null: float = 0.40
    alpha_base_max: float = 0.35
    main_shield_margin: float = 2.5
    lr: float = 2e-4
    aux_ramp_epochs: int = 20
    max_epochs: int = 200
    early_stop_patience: int = 40
    extra_args: dict[str, Any] = field(default_factory=dict)


INFERENCE_PRESETS: dict[str, list[InferenceTrial]] = {
    "inference-default": [
        InferenceTrial("8x8_regional_shield", scale=8, regional=True, shield=True),
        InferenceTrial("8x8_global_shield", scale=8, regional=False, shield=True),
        InferenceTrial("8x8_regional_no_shield", scale=8, regional=True, shield=False),
        InferenceTrial("8x8_no_alpha", scale=8, regional=False, shield=True, no_alpha=True),
        InferenceTrial("16x16_global_shield", scale=16, regional=False, shield=True),
        InferenceTrial("16x16_regional_shield", scale=16, regional=True, shield=True),
        InferenceTrial("16x16_regional_no_shield", scale=16, regional=True, shield=False),
    ],
    "alpha-main-cap": [
        InferenceTrial(
            f"8x8_regional_cap_{cap if cap is not None else 'uncapped'}",
            scale=8, regional=True, shield=True, alpha_main_max=cap,
        )
        for cap in (None, 0.08, 0.12, 0.16)
    ] + [
        InferenceTrial(
            f"16x16_regional_cap_{cap if cap is not None else 'uncapped'}",
            scale=16, regional=True, shield=True, alpha_main_max=cap,
        )
        for cap in (None, 0.08, 0.12, 0.16)
    ],
}

TRAIN_PRESETS: dict[str, list[TrainTrial]] = {
    "mainbeam-focus": [
        TrainTrial(
            "mainbeam70_w035",
            lambda_mainbeam=70.0,
            val_weight_main=0.35,
            alpha_base_max=0.35,
        ),
        TrainTrial(
            "mainbeam80_w040_base025",
            lambda_mainbeam=80.0,
            val_weight_main=0.40,
            alpha_base_max=0.25,
        ),
        TrainTrial(
            "mainbeam70_shield30",
            lambda_mainbeam=70.0,
            val_weight_main=0.35,
            main_shield_margin=3.0,
            alpha_base_max=0.25,
        ),
    ],
    "null-focus": [
        TrainTrial(
            "null_comp32_w050",
            lambda_composed_null=32.0,
            val_weight_null=0.50,
            val_weight_main=0.20,
        ),
        TrainTrial(
            "null_comp32_w055_base020",
            lambda_composed_null=32.0,
            val_weight_null=0.55,
            val_weight_main=0.20,
            alpha_base_max=0.20,
        ),
    ],
}


def _peaknorm(x: np.ndarray) -> np.ndarray:
    return (x - x.max(axis=(1, 2), keepdims=True)).astype(np.float32)


def _load_stats() -> dict:
    extras = np.load(PROCESSED_DIR / "stage1_6x6_extras.npz")
    return extend_stats_with_coupling_gap(
        dict(np.load(NORM_6X6)),
        extras["sub_block_6x6"],
        extras["matlab_6x6"],
        np.load(SPLIT_6X6)["train"],
    )


def _load_model(stats: dict, ckpt: Path) -> FocusedRegionalMultiHeadResUNet:
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    g = FocusedRegionalMultiHeadResUNet(
        base=GEN_BASE,
        attn_heads=ATTN_HEADS,
        top_k=TOP_K_NULLS,
        sb_mean=stats["sub_block_6x6_mean"],
        sb_std=stats["sub_block_6x6_std"],
        res_mean=stats["residual_mean"],
        res_std=stats["residual_std"],
    ).to(DEVICE)
    g.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
    g.eval()
    return g


def _fit_alphas(
    true_res: np.ndarray,
    pred_res: np.ndarray,
    sub_block: np.ndarray,
    *,
    regional: bool,
    alpha_main_max: float | None,
) -> dict[str, float] | None:
    if regional:
        alphas = dict(fit_regional_ls_alphas(true_res, pred_res, sub_block))
        if alpha_main_max is not None:
            alphas["main"] = min(alphas["main"], alpha_main_max)
        return alphas
    alpha = fit_ls_alpha(true_res, pred_res)
    return {"global": alpha, "main": alpha, "sl": alpha, "null": alpha}


def _scale_test_residuals(
    pred_raw: np.ndarray,
    sub_block: np.ndarray,
    indices: np.ndarray,
    alphas: dict[str, float] | None,
    *,
    regional: bool,
    apply_shield: bool,
) -> np.ndarray:
    out = np.empty_like(pred_raw)
    for i, s in enumerate(indices):
        s = int(s)
        raw = pred_raw[i]
        sb = sub_block[s]
        if alphas is None:
            scaled = raw
        elif regional:
            scaled = apply_regional_alpha(raw, sb, alphas, apply_shield=apply_shield)
        else:
            scaled = alphas["global"] * raw
            if apply_shield:
                scaled = shield_main_beam_residual(scaled, sb)
        out[i] = scaled.astype(np.float32)
    return out


def _load_scale_data(scale: int) -> dict:
    pa = PA_8 if scale == 8 else PA_16
    if not pa.exists():
        raise FileNotFoundError(
            f"{pa} missing — run prep_phase_aware_compose and prep_zeroshot_eval_cache --scale {scale}"
        )
    return _load_phase_aware_scale(pa, scale, None, None, None)


def _region_metrics(pred_pn: np.ndarray, true_pn: np.ndarray) -> dict[str, float]:
    full = _masked_point_metrics(pred_pn, true_pn, None)
    main = _masked_point_metrics(pred_pn, true_pn, -10.0)
    null = _masked_point_metrics(pred_pn, true_pn, "null")
    return {
        "full_mae": full["mae"],
        "full_rmse": full["rmse"],
        "main_mae": main["mae"],
        "main_rmse": main["rmse"],
        "null_mae": null["mae"],
        "null_rmse": null["rmse"],
    }


def run_inference_trial(
    trial: InferenceTrial,
    *,
    ckpt: Path,
    stats: dict,
    model: FocusedRegionalMultiHeadResUNet | None = None,
) -> dict[str, Any]:
    data = _load_scale_data(trial.scale)
    matlab = data["matlab"]
    sub_block = data["sub_block"]
    hfss = data["hfss"]
    fingerprint = data["fingerprint"]
    dpx = data["dpx"]
    dpy = data["dpy"]
    tr = data["train_idx"]
    te = data["test_idx"]

    m_mean, m_std = _pixel_stats(matlab, tr)
    sb_mean, sb_std = _pixel_stats(sub_block, tr)
    gap_mean, gap_std = _gap_stats(matlab, sub_block, tr)

    g = model if model is not None else _load_model(stats, ckpt)
    run_inference = run_inference_8x8 if trial.scale == 8 else run_inference_16x16
    infer_shield_on_raw = trial.scale == 16

    pred_res_tr, _ = run_inference(
        g, stats,
        matlab=matlab, sub_block=sub_block, fingerprint=fingerprint,
        dpx=dpx, dpy=dpy,
        m_mean=m_mean, m_std=m_std, sb_mean=sb_mean, sb_std=sb_std,
        gap_mean=gap_mean, gap_std=gap_std,
        indices=tr, apply_shield=infer_shield_on_raw and trial.shield,
    )
    true_res_tr = (hfss[tr] - sub_block[tr]).astype(np.float32)

    alphas: dict[str, float] | None = None
    if not trial.no_alpha:
        alphas = _fit_alphas(
            true_res_tr, pred_res_tr, sub_block[tr],
            regional=trial.regional,
            alpha_main_max=trial.alpha_main_max,
        )

    pred_res_te, _ = run_inference(
        g, stats,
        matlab=matlab, sub_block=sub_block, fingerprint=fingerprint,
        dpx=dpx, dpy=dpy,
        m_mean=m_mean, m_std=m_std, sb_mean=sb_mean, sb_std=sb_std,
        gap_mean=gap_mean, gap_std=gap_std,
        indices=te, apply_shield=False,
    )

    if trial.no_alpha:
        pred_scaled = pred_res_te
        if trial.shield:
            pred_scaled = np.stack([
                shield_main_beam_residual(pred_res_te[i], sub_block[int(te[i])])
                for i in range(len(te))
            ])
    elif trial.regional and alphas is not None:
        pred_scaled = _scale_test_residuals(
            pred_res_te, sub_block, te, alphas,
            regional=True, apply_shield=trial.shield,
        )
    elif alphas is not None:
        pred_scaled = alphas["global"] * pred_res_te
        if trial.shield:
            pred_scaled = np.stack([
                shield_main_beam_residual(pred_scaled[i], sub_block[int(te[i])])
                for i in range(len(te))
            ])
    else:
        pred_scaled = pred_res_te

    pred_raw = (sub_block[te] + pred_scaled).astype(np.float32)
    pred_pn = _peaknorm(pred_raw)
    true_pn = _peaknorm(hfss[te])
    sb_pn = _peaknorm(sub_block[te])

    metrics_model = evaluate_predictions(pred_pn, true_pn, matlab[te])
    metrics_baseline = evaluate_predictions(sb_pn, true_pn, matlab[te])
    region = _region_metrics(pred_pn, true_pn)

    return {
        "trial": asdict(trial),
        "scale": trial.scale,
        "n_test": int(len(te)),
        "alphas": alphas,
        "region_metrics": region,
        "metrics_model": metrics_model,
        "metrics_baseline": metrics_baseline,
        "learned_alpha_base": float(g.alpha_base),
        "learned_alpha_null": float(g.alpha_null),
        "learned_alpha_e": float(g.alpha_e),
        "learned_alpha_h": float(g.alpha_h),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "kind", "name", "scale", "regional", "shield", "no_alpha", "alpha_main_max",
        "full_mae", "main_mae", "null_mae", "null_fill_pct", "sll_error_db",
        "alpha_main", "alpha_sl", "alpha_null", "alpha_global",
        "checkpoint", "best_val_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_row_from_inference(result: dict[str, Any], ckpt: Path) -> dict[str, Any]:
    trial = result["trial"]
    mm = result["metrics_model"]
    alphas = result.get("alphas") or {}
    return {
        "kind": "inference",
        "name": trial["name"],
        "scale": trial["scale"],
        "regional": trial["regional"],
        "shield": trial["shield"],
        "no_alpha": trial["no_alpha"],
        "alpha_main_max": trial["alpha_main_max"],
        "full_mae": result["region_metrics"]["full_mae"],
        "main_mae": result["region_metrics"]["main_mae"],
        "null_mae": result["region_metrics"]["null_mae"],
        "null_fill_pct": mm.get("null_null_fill_accuracy_pct"),
        "sll_error_db": mm.get("antenna_sll_error_db"),
        "alpha_main": alphas.get("main"),
        "alpha_sl": alphas.get("sl"),
        "alpha_null": alphas.get("null"),
        "alpha_global": alphas.get("global"),
        "checkpoint": str(ckpt),
        "best_val_score": None,
    }


def run_inference_sweep(
    trials: list[InferenceTrial],
    *,
    ckpt: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    stats = _load_stats()
    model = _load_model(stats, ckpt)
    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for trial in trials:
        print(f"\n[inference] {trial.name} (scale={trial.scale})", flush=True)
        result = run_inference_trial(trial, ckpt=ckpt, stats=stats, model=model)
        results.append(result)
        trial_dir = out_dir / "inference" / trial.name
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary_rows.append(_summary_row_from_inference(result, ckpt))
        rm = result["region_metrics"]
        print(
            f"  full MAE={rm['full_mae']:.3f}  main MAE={rm['main_mae']:.3f}  "
            f"null MAE={rm['null_mae']:.3f}  null_fill="
            f"{result['metrics_model'].get('null_null_fill_accuracy_pct', float('nan')):.1f}%",
            flush=True,
        )

    _write_summary_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "inference_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    return results


def run_train_sweep(
    trials: list[TrainTrial],
    *,
    out_dir: Path,
    init_v2: bool,
    inference_after: list[InferenceTrial] | None = None,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    train_results: list[dict[str, Any]] = []

    for trial in trials:
        print(f"\n[train] {trial.name}", flush=True)
        cmd = [
            sys.executable, "-m", "scripts.train_multihead_v4_trial",
            "--trial-id", trial.name,
            "--max-epochs", str(trial.max_epochs),
            "--early-stop-patience", str(trial.early_stop_patience),
            "--lr", str(trial.lr),
            "--aux-ramp-epochs", str(trial.aux_ramp_epochs),
            "--lambda-mainbeam", str(trial.lambda_mainbeam),
            "--lambda-composed-null", str(trial.lambda_composed_null),
            "--val-weight-main", str(trial.val_weight_main),
            "--val-weight-null", str(trial.val_weight_null),
            "--alpha-base-max", str(trial.alpha_base_max),
            "--main-shield-margin", str(trial.main_shield_margin),
        ]
        if init_v2:
            cmd.append("--init-v2")
        for key, value in trial.extra_args.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

        ckpt = CHECKPOINTS_DIR / "tuning_v4" / trial.name / "best_generator.pt"
        summary_path = CHECKPOINTS_DIR / "tuning_v4" / trial.name / "train_summary.json"
        train_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        train_results.append({"trial": asdict(trial), "train_summary": train_summary})

        infer_trials = inference_after or [
            InferenceTrial(f"{trial.name}_8x8_regional", scale=8, regional=True, shield=True),
            InferenceTrial(f"{trial.name}_16x16_regional", scale=16, regional=True, shield=True),
        ]
        infer_out = out_dir / "after_train" / trial.name
        infer_results = run_inference_sweep(infer_trials, ckpt=ckpt, out_dir=infer_out)
        for ir in infer_results:
            row = _summary_row_from_inference(ir, ckpt)
            row["name"] = f"{trial.name}/{row['name']}"
            row["best_val_score"] = train_summary.get("best_val_score")
            summary_rows.append(row)

    _write_summary_csv(out_dir / "train_summary.csv", summary_rows)
    (out_dir / "train_results.json").write_text(
        json.dumps(train_results, indent=2), encoding="utf-8",
    )
    return train_results


def _resolve_preset(mode: str, preset: str) -> tuple[list[InferenceTrial], list[TrainTrial]]:
    infer: list[InferenceTrial] = []
    train: list[TrainTrial] = []
    if mode == "inference":
        if preset not in INFERENCE_PRESETS:
            raise KeyError(
                f"Unknown inference preset {preset!r}. "
                f"Available: {', '.join(sorted(INFERENCE_PRESETS))}"
            )
        infer = INFERENCE_PRESETS[preset]
    elif mode == "train":
        if preset not in TRAIN_PRESETS:
            raise KeyError(
                f"Unknown train preset {preset!r}. "
                f"Available: {', '.join(sorted(TRAIN_PRESETS))}"
            )
        train = TRAIN_PRESETS[preset]
    elif mode == "all":
        if preset not in TRAIN_PRESETS:
            raise KeyError(
                f"Unknown train preset {preset!r} for --mode all. "
                f"Available: {', '.join(sorted(TRAIN_PRESETS))}"
            )
        train = TRAIN_PRESETS[preset]
    return infer, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune multi-head v4 hyperparameters")
    parser.add_argument(
        "--mode",
        choices=("inference", "train", "all"),
        default="inference",
        help="inference=sweep eval knobs; train=retrain trials; all=train then eval",
    )
    parser.add_argument(
        "--preset",
        default="inference-default",
        help="Preset name (inference or train depending on mode)",
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--run-id", default=None, help="Output subfolder under results/tuning_v4/")
    parser.add_argument("--init-v2", action="store_true", help="Warm-start train trials from v2")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list_presets:
        print("Inference presets:")
        for name, trials in INFERENCE_PRESETS.items():
            print(f"  {name} ({len(trials)} trials)")
        print("\nTrain presets:")
        for name, trials in TRAIN_PRESETS.items():
            print(f"  {name} ({len(trials)} trials)")
        return

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "mode": args.mode,
        "preset": args.preset,
        "checkpoint": str(args.ckpt),
        "init_v2": args.init_v2,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    infer_trials, train_trials = _resolve_preset(args.mode, args.preset)

    if args.dry_run:
        print(json.dumps({"meta": meta, "inference": [asdict(t) for t in infer_trials],
                          "train": [asdict(t) for t in train_trials]}, indent=2))
        return

    if args.mode in ("inference", "all") and infer_trials:
        run_inference_sweep(infer_trials, ckpt=args.ckpt, out_dir=out_dir)

    if args.mode in ("train", "all") and train_trials:
        run_train_sweep(train_trials, out_dir=out_dir, init_v2=args.init_v2)

    print(f"\nTuning run complete: {out_dir}", flush=True)
    print(f"  summary CSV: {out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
