"""
Train the residual U-Net **only** on 2x2 and 4x4 data (``antenna_data_2to4.h5``).

This is the small-footprint "learn antenna physics from coarse arrays" setup:

- **Supervision**: N=4 uses simulated HFSS ``hfss_4x4``. N=2 uses ``hfss_pred_2x2``
  by default (the fusion HDF5 has no separate ``hfss_2x2`` field; change with
  ``--n2-hfss-key`` if your file does).
- **Anchors (ch4)**: cascade field ``hfss_pred_2x2`` at N=4 (override with ``--anchor-key-n4``).
- **No 8x8 / 16x16 HFSS in the loss** — larger arrays are out-of-distribution;
  ``val_16x16`` is still monitored for checkpoint selection (same protocol as bootstrap phase A).
- **Larger arrays at run time**: use ``ResidualPhysicsPredictor`` with ``scale_n`` in ``{8, 16}`` and
  an anchor one scale below. Official 16x16 test::

    python -m scripts.evaluate_residual_multiscale \\
        --checkpoint checkpoints/residual_physics_2to4_only/best_generator.pt \\
        --results results/residual_physics_2to4_only

Usage::

    python -m scripts.train_residual_physics_2to4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.config import CHECKPOINTS_DIR, RANDOM_SEED
from scripts import train_residual_bootstrap as boot

CKPT_PHYSICS_2TO4 = CHECKPOINTS_DIR / "residual_physics_2to4_only"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Residual generator on 2x2+4x4 only (HFSS 4x4); extrapolate to 8x8/16x16 at inference."
    )
    ap.add_argument("--epochs", type=int, default=None, help="default: train_residual_multiscale.EPOCHS")
    ap.add_argument("--skip-n2", action="store_true", help="Train N=4 branch only")
    ap.add_argument(
        "--og-sized-epochs",
        action="store_true",
        help="Match multiscale sample count per epoch (rotating N=2 block)",
    )
    ap.add_argument("--n2-hfss-key", default="hfss_pred_2x2", help="N=2 supervision HDF5 key")
    ap.add_argument(
        "--anchor-key-n4",
        default="hfss_pred_2x2",
        help="ch4 anchor at N=4 (must exist in 2to4 HDF5)",
    )
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--validate-every", type=int, default=1)
    ap.add_argument("--compile", action="store_true", help="torch.compile if Inductor/Triton available")
    ap.add_argument("--fused-adam", action="store_true")
    ap.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="EMA decay (default: EMA_DECAY from train_cgan_2to4_fusion_no_m4)",
    )
    args = ap.parse_args()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ns = argparse.Namespace(
        skip_n2=args.skip_n2,
        og_sized_epochs=args.og_sized_epochs,
        epochs=args.epochs,
        n2_hfss_key=args.n2_hfss_key,
        ckpt_dir_phase_a=str(CKPT_PHYSICS_2TO4),
        hfss_supervision_n4="hfss_4x4",
        anchor_key_n4=args.anchor_key_n4,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        validate_every=args.validate_every,
        compile=args.compile,
        fused_adam=args.fused_adam,
        ema_decay=args.ema_decay,
    )

    from scripts.train_residual_multiscale import AMP_DTYPE, USE_AMP

    amp_dtype_str = str(AMP_DTYPE).replace("torch.", "") if USE_AMP else "off"
    print(
        f"[physics_2to4] torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={boot.DEVICE} AMP={USE_AMP} dtype={amp_dtype_str}\n"
        f"[physics_2to4] ckpt_dir={CKPT_PHYSICS_2TO4} | supervision: N=4 hfss_4x4, "
        f"N=2 {args.n2_hfss_key!r}",
        flush=True,
    )

    boot.cmd_phase_a(ns)


if __name__ == "__main__":
    main()
