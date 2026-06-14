"""
Parametric Stage-1 training for multi-head v4 tuning trials.

Separate from ``train_stage1_6x6_multihead_v4.py`` so the baseline trainer stays
unchanged.  Invoked by ``scripts.tune_multihead_v4`` (train mode) or directly:

    python -m scripts.train_multihead_v4_trial --trial-id mainbeam70 --init-v2
    python -m scripts.train_multihead_v4_trial --trial-id quick --max-epochs 30 \\
        --lambda-mainbeam 70 --val-weight-main 0.35
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import src.models.multihead_v4 as mh4
from scripts.train_cgan_2to4_fusion_no_m4 import AMP_DEVICE_TYPE, EMA_DECAY, GEN_BASE, USE_AMP
from scripts.train_stage1_6x6_multihead_v2 import extend_stats_with_coupling_gap
from scripts.train_stage1_6x6_multihead_v4 import (
    NPZ_EXTRAS,
    NORM_6X6,
    SPLIT_6X6,
    Stage1_6x6_MultiHeadV4Dataset,
    _load_shared_encoder,
    train_one_epoch,
)
from src.config import BATCH_SIZE, CHECKPOINTS_DIR, DEVICE, MAX_EPOCHS, RANDOM_SEED
from src.models.multihead_v4 import EMA, FocusedRegionalLoss, FocusedRegionalMultiHeadResUNet
from src.training.metrics import mae, pearson_correlation, rmse
from scripts.train_cgan_2to4_fusion_no_m4 import ATTN_HEADS

CKPT_V2 = CHECKPOINTS_DIR / "stage1_6x6_multihead_v2" / "best_generator.pt"
TUNING_ROOT = CHECKPOINTS_DIR / "tuning_v4"

PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 4
PLATEAU_MIN_LR = 1e-6
NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda"
VALIDATE_EVERY = 2
SAVE_EVERY = 10


@dataclass
class TrialConfig:
    trial_id: str
    lr: float = 2e-4
    max_epochs: int = 200
    early_stop_patience: int = 40
    aux_ramp_epochs: int = 20
    lambda_mainbeam: float = 55.0
    lambda_composed_null: float = 24.0
    lambda_eplane: float = 8.0
    lambda_hplane: float = 8.0
    lambda_null: float = 8.0
    val_weight_main: float = 0.30
    val_weight_main_sl: float = 0.30
    val_weight_null: float = 0.40
    alpha_base_max: float = 0.35
    main_shield_margin: float = 2.5
    top_k_nulls: int = 10
    init_v2: bool = False


def _aux_scale(epoch: int, ramp_epochs: int) -> float:
    return min(1.0, epoch / ramp_epochs)


def _patch_module(cfg: TrialConfig) -> None:
    mh4.LAMBDA_MAINBEAM = cfg.lambda_mainbeam
    mh4.LAMBDA_COMPOSED_NULL = cfg.lambda_composed_null
    mh4.LAMBDA_EPLANE = cfg.lambda_eplane
    mh4.LAMBDA_HPLANE = cfg.lambda_hplane
    mh4.LAMBDA_NULL = cfg.lambda_null
    mh4.VAL_WEIGHT_MAIN = cfg.val_weight_main
    mh4.VAL_WEIGHT_MAIN_SL = cfg.val_weight_main_sl
    mh4.VAL_WEIGHT_NULL = cfg.val_weight_null
    mh4.ALPHA_BASE_MAX = cfg.alpha_base_max
    mh4.MAIN_SHIELD_MARGIN = cfg.main_shield_margin
    mh4.TOP_K_NULLS = cfg.top_k_nulls


@torch.no_grad()
def evaluate_composed_trial(model, ema, loader, res_mean, res_std, sb_mean, sb_std, cfg: TrialConfig):
    live = ema.swap_into(model)
    try:
        model.eval()
        main_err, main_sl_err, null_err, all_err = [], [], [], []
        for x, y, null_mask, *_ in loader:
            x = x.to(DEVICE, non_blocking=PIN_MEMORY)
            y = y.to(DEVICE, non_blocking=PIN_MEMORY)
            null_mask = null_mask.to(DEVICE, non_blocking=PIN_MEMORY)
            with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
                pred_n = model(x)
            pred_db = pred_n * res_std + res_mean
            tgt_db = y * res_std + res_mean
            sb_db = x[:, 1:2] * sb_std + sb_mean
            composed_pred = sb_db + pred_db
            composed_true = sb_db + tgt_db
            err = (composed_pred - composed_true).float()

            sb_peak = sb_db.amax(dim=(-2, -1), keepdim=True)
            main = composed_true >= (sb_peak - 3.0)
            main_sl = composed_true >= (sb_peak - 20.0)
            for i in range(err.shape[0]):
                e = err[i, 0].cpu().numpy()
                t = composed_true[i, 0].cpu().numpy()
                p = t + e
                nm = null_mask[i, 0].cpu().numpy() > 0.5
                mb = main[i, 0].cpu().numpy()
                ms = main_sl[i, 0].cpu().numpy()
                if mb.any():
                    main_err.append(rmse(p[mb], t[mb]))
                if ms.any():
                    main_sl_err.append(rmse(p[ms], t[ms]))
                if nm.any():
                    null_err.append(rmse(p[nm], t[nm]))
                all_err.append(rmse(p.ravel(), t.ravel()))

        main_rmse = float(np.mean(main_err)) if main_err else float("nan")
        main_sl_rmse = float(np.mean(main_sl_err)) if main_sl_err else float("nan")
        null_rmse = float(np.mean(null_err)) if null_err else float("nan")
        full_rmse = float(np.mean(all_err)) if all_err else float("nan")
        score = (
            cfg.val_weight_main * main_rmse
            + cfg.val_weight_main_sl * main_sl_rmse
            + cfg.val_weight_null * null_rmse
        )
        return {
            "score": score,
            "main_rmse": main_rmse,
            "main_sl_rmse": main_sl_rmse,
            "null_rmse": null_rmse,
            "full_rmse": full_rmse,
        }
    finally:
        ema.restore(model, live)


@torch.no_grad()
def evaluate_residual_trial(model, ema, loader):
    live = ema.swap_into(model)
    try:
        model.eval()
        yt, yp = [], []
        for x, y, *_ in loader:
            x = x.to(DEVICE, non_blocking=PIN_MEMORY)
            y = y.to(DEVICE, non_blocking=PIN_MEMORY)
            with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
                y_hat = model(x)
            yt.append(y.cpu().numpy())
            yp.append(y_hat.cpu().numpy())
        yt = np.concatenate(yt).reshape(-1)
        yp = np.concatenate(yp).reshape(-1)
        return {
            "rmse": float(rmse(yt, yp)),
            "mae": float(mae(yt, yp)),
            "pearson": float(pearson_correlation(yt, yp)),
        }
    finally:
        ema.restore(model, live)


def train_trial(cfg: TrialConfig) -> dict:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    _patch_module(cfg)
    ckpt_dir = TUNING_ROOT / cfg.trial_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    for p in (NPZ_EXTRAS, NORM_6X6, SPLIT_6X6):
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"Trial {cfg.trial_id}  ckpt={ckpt_dir}", flush=True)
    e = np.load(NPZ_EXTRAS)
    stats = extend_stats_with_coupling_gap(
        dict(np.load(NORM_6X6)),
        e["sub_block_6x6"],
        e["matlab_6x6"],
        np.load(SPLIT_6X6)["train"],
    )
    sp = np.load(SPLIT_6X6)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"Splits: {len(tr)} train / {len(va)} val / {len(te)} test", flush=True)

    tr_ds = Stage1_6x6_MultiHeadV4Dataset(
        tr, stats, e["matlab_6x6"], e["sub_block_6x6"], e["fingerprint"],
        e["hfss_6x6"], e["dpx"], e["dpy"], augment_noise=True, top_k=cfg.top_k_nulls,
    )
    va_ds = Stage1_6x6_MultiHeadV4Dataset(
        va, stats, e["matlab_6x6"], e["sub_block_6x6"], e["fingerprint"],
        e["hfss_6x6"], e["dpx"], e["dpy"], augment_noise=False, top_k=cfg.top_k_nulls,
    )
    tr_loader = DataLoader(
        tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, drop_last=True,
    )
    va_loader = DataLoader(
        va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    res_mean_t = torch.from_numpy(stats["residual_mean"].astype(np.float32))[None, None].to(DEVICE)
    res_std_t = torch.from_numpy(stats["residual_std"].astype(np.float32))[None, None].to(DEVICE)
    sb_mean_t = torch.from_numpy(stats["sub_block_6x6_mean"].astype(np.float32))[None, None].to(DEVICE)
    sb_std_t = torch.from_numpy(stats["sub_block_6x6_std"].astype(np.float32))[None, None].to(DEVICE)

    G = FocusedRegionalMultiHeadResUNet(
        base=GEN_BASE, attn_heads=ATTN_HEADS, top_k=cfg.top_k_nulls,
        sb_mean=stats["sub_block_6x6_mean"], sb_std=stats["sub_block_6x6_std"],
        res_mean=stats["residual_mean"], res_std=stats["residual_std"],
    ).to(DEVICE)
    if cfg.init_v2:
        _load_shared_encoder(G, CKPT_V2)

    opt = optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR,
    )
    scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
    criterion = FocusedRegionalLoss(
        stats["residual_mean"], stats["residual_std"],
        stats["sub_block_6x6_mean"], stats["sub_block_6x6_std"],
        lambda_mainbeam=cfg.lambda_mainbeam,
        lambda_composed_null=cfg.lambda_composed_null,
        lambda_eplane=cfg.lambda_eplane,
        lambda_hplane=cfg.lambda_hplane,
        lambda_null=cfg.lambda_null,
    ).to(DEVICE)
    ema = EMA(G, decay=EMA_DECAY)

    best_score = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    epochs = min(cfg.max_epochs, MAX_EPOCHS)

    # Patch aux ramp used by imported train_one_epoch via module attribute on train script.
    import scripts.train_stage1_6x6_multihead_v4 as train_v4
    train_v4.AUX_RAMP_EPOCHS = cfg.aux_ramp_epochs

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\n=== {cfg.trial_id} epoch {epoch}/{epochs} ===", flush=True)
        tr_stats = train_one_epoch(G, ema, tr_loader, opt, scaler, criterion, epoch)
        print(
            f"  train loss={tr_stats['loss']:.4f}  mainbeam={tr_stats['mainbeam']:.4f}  "
            f"null_c={tr_stats['composed_null']:.3f}",
            flush=True,
        )

        if epoch % VALIDATE_EVERY == 0 or epoch == 1 or epoch == epochs:
            comp = evaluate_composed_trial(
                G, ema, va_loader, res_mean_t, res_std_t, sb_mean_t, sb_std_t, cfg,
            )
            res_m = evaluate_residual_trial(G, ema, va_loader)
            print(
                f"  val score={comp['score']:.4f}  main={comp['main_rmse']:.4f}  "
                f"null={comp['null_rmse']:.4f}  res_rmse={res_m['rmse']:.4f}",
                flush=True,
            )
            if comp["score"] < best_score - 1e-5:
                best_score = comp["score"]
                best_epoch = epoch
                epochs_without_improve = 0
                torch.save(ema.state_dict(), ckpt_dir / "best_generator.pt")
            else:
                epochs_without_improve += VALIDATE_EVERY
            sched.step(comp["score"])

        torch.save(G.state_dict(), ckpt_dir / "last_generator.pt")
        torch.save(ema.state_dict(), ckpt_dir / "last_generator_ema.pt")
        if epoch % SAVE_EVERY == 0:
            torch.save(ema.state_dict(), ckpt_dir / f"generator_ema_epoch_{epoch:03d}.pt")

        if epochs_without_improve >= cfg.early_stop_patience:
            print(f"Early stop at epoch {epoch}; best={best_epoch} score={best_score:.5f}", flush=True)
            break

    summary = {
        "trial_id": cfg.trial_id,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "minutes": (time.time() - t0) / 60.0,
        "checkpoint": str(ckpt_dir / "best_generator.pt"),
    }
    (ckpt_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done: {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multi-head v4 tuning trial")
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--init-v2", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--early-stop-patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--aux-ramp-epochs", type=int, default=20)
    parser.add_argument("--lambda-mainbeam", type=float, default=55.0)
    parser.add_argument("--lambda-composed-null", type=float, default=24.0)
    parser.add_argument("--lambda-eplane", type=float, default=8.0)
    parser.add_argument("--lambda-hplane", type=float, default=8.0)
    parser.add_argument("--lambda-null", type=float, default=8.0)
    parser.add_argument("--val-weight-main", type=float, default=0.30)
    parser.add_argument("--val-weight-main-sl", type=float, default=0.30)
    parser.add_argument("--val-weight-null", type=float, default=0.40)
    parser.add_argument("--alpha-base-max", type=float, default=0.35)
    parser.add_argument("--main-shield-margin", type=float, default=2.5)
    parser.add_argument("--top-k-nulls", type=int, default=10)
    args = parser.parse_args()

    cfg = TrialConfig(
        trial_id=args.trial_id,
        lr=args.lr,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        aux_ramp_epochs=args.aux_ramp_epochs,
        lambda_mainbeam=args.lambda_mainbeam,
        lambda_composed_null=args.lambda_composed_null,
        lambda_eplane=args.lambda_eplane,
        lambda_hplane=args.lambda_hplane,
        lambda_null=args.lambda_null,
        val_weight_main=args.val_weight_main,
        val_weight_main_sl=args.val_weight_main_sl,
        val_weight_null=args.val_weight_null,
        alpha_base_max=args.alpha_base_max,
        main_shield_margin=args.main_shield_margin,
        top_k_nulls=args.top_k_nulls,
        init_v2=args.init_v2,
    )
    train_trial(cfg)


if __name__ == "__main__":
    main()
