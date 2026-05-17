"""
Unified **stage** residual trainer (same G / D / losses as bootstrap + multiscale).

Compared to ``train_residual_bootstrap`` phase A, N=4 supervision uses **simulated HFSS**
ground truth (``hfss_4x4``) rather than preferring ``hfss_pred_4x4``. The project's
standard 2->4 HDF5 does not ship a separate ``hfss_2x2`` array, so N=2 supervision stays
on ``hfss_pred_2x2`` by default (override with ``--n2-hfss-key`` if your H5 has more).

**Anchors (ch4)** stay cascade-style (``hfss_pred_2x2`` / ``hfss_pred_4x4``) so behaviour
at inference matches the multiscale pipeline.

End-to-end (one command)::

    python -m scripts.train_residual_stage run

This runs: phase A (N=2+4, no 8x8 loss); export pseudo 8x8 from phase-A EMA on 4->8 rows;
phase C (N=4 + N=8 with pseudo 8x8 targets). Checkpoints go to
``checkpoints/residual_stage_phase_{a,c}/`` by default.

**4070-oriented defaults** (overridable): TF32 on, bf16 AMP (from multiscale), 4 dataloader
workers, validation every 2 epochs, ``torch.compile`` when Inductor/Triton works (skipped on
Windows; install Triton on Linux for speedup), fused CUDA Adam when available.

For manual single phases (same hyperparameters as this script's defaults), use
``python -m scripts.train_residual_bootstrap`` with the new flags documented there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.config import CHECKPOINTS_DIR, PROCESSED_DIR, RANDOM_SEED
from scripts import train_residual_bootstrap as boot
from scripts.train_residual_multiscale import AMP_DTYPE, BATCH, USE_AMP

CKPT_STAGE_A = CHECKPOINTS_DIR / "residual_stage_phase_a"
CKPT_STAGE_C = CHECKPOINTS_DIR / "residual_stage_phase_c"
DEFAULT_PSEUDO = PROCESSED_DIR / "hfss_8x8_pseudo_stage.npz"


def _default_perf_kwargs(
    *,
    num_workers: int | None,
    batch_size: int | None,
    validate_every: int | None,
    use_compile: bool,
    fused_adam: bool,
) -> dict:
    return {
        "hfss_supervision_n4": "hfss_4x4",
        "anchor_key_n4": "hfss_pred_2x2",
        "num_workers": 4 if num_workers is None else int(num_workers),
        "batch_size": batch_size,
        "validate_every": 2 if validate_every is None else int(validate_every),
        "compile": use_compile,
        "fused_adam": fused_adam,
    }


def cmd_run_all(args: argparse.Namespace) -> None:
    pseudo = Path(args.pseudo)
    perf = _default_perf_kwargs(
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        validate_every=args.validate_every,
        use_compile=not args.no_compile,
        fused_adam=not args.no_fused_adam,
    )

    ns_a = argparse.Namespace(
        skip_n2=args.skip_n2,
        og_sized_epochs=args.og_sized_epochs,
        epochs=args.epochs_a,
        n2_hfss_key=args.n2_hfss_key,
        ckpt_dir_phase_a=str(CKPT_STAGE_A),
        ema_decay=args.ema_decay,
        **perf,
    )
    print(
        "\n"
        + "=" * 60
        + "\n[stage] Phase A - N=2+4, N=4 targets = hfss_4x4 (no 8x8 supervision)\n"
        + "=" * 60,
        flush=True,
    )
    boot.cmd_phase_a(ns_a)

    gen_a = CKPT_STAGE_A / "best_generator.pt"
    if not gen_a.exists():
        raise FileNotFoundError(f"Phase A did not produce {gen_a}")

    ns_ex = argparse.Namespace(
        generator=str(gen_a),
        output=str(pseudo),
        split=args.pseudo_split,
        batch_size=args.export_batch_size,
    )
    print("\n" + "=" * 60 + "\n[stage] Export pseudo 8x8 (EMA generator, 4->8 rows)\n" + "=" * 60, flush=True)
    boot.cmd_export_pseudo(ns_ex)

    ns_c = argparse.Namespace(
        pseudo=str(pseudo),
        init_generator=str(gen_a),
        epochs=args.epochs_c,
        ckpt_dir_phase_c=str(CKPT_STAGE_C),
        ema_decay=args.ema_decay,
        **perf,
    )
    print(
        "\n"
        + "=" * 60
        + "\n[stage] Phase C - N=4 (hfss_4x4) + N=8 (pseudo 8x8 targets)\n"
        + "=" * 60,
        flush=True,
    )
    boot.cmd_phase_c(ns_c)

    print(
        f"\n[stage] Done. Phase A: {CKPT_STAGE_A} | Phase C: {CKPT_STAGE_C} | pseudo: {pseudo}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage residual trainer: HFSS 2-4, predict 8x8, train 4-8 (single script)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run phase A, export pseudo 8x8, then phase C")
    run.add_argument(
        "--epochs-a",
        type=int,
        default=None,
        help="Phase A epochs (defaults to train_residual_multiscale.EPOCHS)",
    )
    run.add_argument(
        "--epochs-c",
        type=int,
        default=None,
        help="Phase C epochs (same default as phase A if unset)",
    )
    run.add_argument("--skip-n2", action="store_true", help="Train N=4 only in phase A")
    run.add_argument(
        "--og-sized-epochs",
        action="store_true",
        help="Phase A: len(tr4)+len(tr8) samples/epoch, rotating N=2 block",
    )
    run.add_argument("--n2-hfss-key", default="hfss_pred_2x2", help="N=2 supervision HDF5 key")
    run.add_argument(
        "--pseudo",
        type=str,
        default=str(DEFAULT_PSEUDO),
        help="Written by export; read again in phase C",
    )
    run.add_argument(
        "--pseudo-split",
        choices=["train", "val", "trainval", "all"],
        default="trainval",
        help="Which 4->8 indices receive pseudo labels (trainval matches bootstrap phase C)",
    )
    run.add_argument("--export-batch-size", type=int, default=BATCH)
    run.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default 4; set 0 if HDF5 workers misbehave on your setup)",
    )
    run.add_argument("--batch-size", type=int, default=None, help="Training batch size override")
    run.add_argument(
        "--validate-every",
        type=int,
        default=None,
        help="Validate every N epochs (default 2 for faster epochs)",
    )
    run.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (on Linux, Inductor needs Triton; Windows skips compile automatically)",
    )
    run.add_argument(
        "--no-fused-adam",
        action="store_true",
        help="Disable fused CUDA Adam",
    )
    run.add_argument(
        "--ema-decay",
        type=float,
        default=0.9995,
        help="EMA decay (slightly smoother than bootstrap default 0.999; set lower to match bootstrap)",
    )
    run.set_defaults(func=cmd_run_all)

    args = ap.parse_args()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_dtype_str = str(AMP_DTYPE).replace("torch.", "") if USE_AMP else "off"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[stage] torch={torch.__version__} cuda={torch.version.cuda} device={dev} "
        f"AMP={USE_AMP} dtype={amp_dtype_str}",
        flush=True,
    )
    args.func(args)


if __name__ == "__main__":
    main()
