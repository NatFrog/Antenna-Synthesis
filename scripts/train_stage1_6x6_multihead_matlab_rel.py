"""
Stage 1 at 6×6 — multi-head v4 with matlab-relative HFSS correction.

Trains on sub-block compositions from ``datasets_6x6sub-block_*_fixed`` (via
``prep_stage1_6x6``) while predicting ``hfss_6x6 − matlab_6x6``.  Sub-block
and coupling_gap (sub − matlab) remain input channels for coupling context.

Prep first (standalone 6×6 CSV pipeline — no 4×4 compose):
    python -m scripts.prep_stage1_6x6_matlab_rel

Usage:
    python -m scripts.train_stage1_6x6_multihead_matlab_rel
    python -m scripts.train_stage1_6x6_multihead_matlab_rel --init-v4
    python -m scripts.train_stage1_6x6_multihead_matlab_rel --resume

Checkpoints: checkpoints/stage1_6x6_multihead_matlab_rel/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from scripts.train_cgan_2to4_fusion_no_m4 import AMP_DEVICE_TYPE, NOISE_STD, USE_AMP
from scripts.train_stage1_6x6_multihead_v2 import extend_stats_with_coupling_gap
from src.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DEVICE,
    MAX_EPOCHS,
    NULL_THRESHOLD_DB,
    PROCESSED_DIR,
    RANDOM_SEED,
)
from src.models.multihead_v4 import (
    EMA,
    FocusedRegionalLoss,
    FocusedRegionalMultiHeadResUNet,
    TOP_K_NULLS,
    VAL_WEIGHT_MAIN,
    VAL_WEIGHT_MAIN_SL,
    VAL_WEIGHT_NULL,
    compute_topk_null_depths,
)
from src.training.metrics import mae, pearson_correlation, rmse
from src.training.stage1_6x6_common import evaluate_composed_matlab_rel
from scripts.train_cgan_2to4_fusion_no_m4 import ATTN_HEADS, EMA_DECAY, GEN_BASE

NPZ_EXTRAS = PROCESSED_DIR / "stage1_6x6_mr_extras.npz"
NORM_6X6 = PROCESSED_DIR / "norm_stats_stage1_6x6_mr.npz"
SPLIT_6X6 = PROCESSED_DIR / "split_indices_stage1_6x6_mr.npz"
CKPT_V4 = CHECKPOINTS_DIR / "stage1_6x6_multihead_v4" / "best_generator.pt"
CKPT_DIR = CHECKPOINTS_DIR / "stage1_6x6_multihead_matlab_rel"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = min(MAX_EPOCHS, 200)
BATCH = BATCH_SIZE
LR = 2e-4
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 4
PLATEAU_MIN_LR = 1e-6
NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda"
VALIDATE_EVERY = 2
EARLY_STOP_PATIENCE = 40
SAVE_EVERY = 10
STEP_PRINT_EVERY = 25
AUX_RAMP_EPOCHS = 20

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


class Stage1_6x6_MultiHeadV4Dataset(Dataset):
    def __init__(
        self,
        indices,
        stats,
        matlab_6x6,
        sub_block_6x6,
        fingerprint,
        hfss_6x6,
        dpx,
        dpy,
        augment_noise: bool = False,
        noise_std: float = NOISE_STD,
        top_k: int = TOP_K_NULLS,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.m6 = matlab_6x6
        self.sb6 = sub_block_6x6
        self.fp = fingerprint
        self.hf6 = hfss_6x6
        self.dpx = dpx
        self.dpy = dpy
        self.augment_noise = augment_noise
        self.noise_std = noise_std
        self.top_k = top_k
        self.s = {k: v.astype(np.float32) for k, v in stats.items()}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        m6 = self.m6[idx].astype(np.float32)
        sb6 = self.sb6[idx].astype(np.float32)
        fp = self.fp[idx].astype(np.float32)
        hf6 = self.hf6[idx].astype(np.float32)
        res = hf6 - m6

        m6_n = (m6 - self.s["matlab_6x6_mean"]) / self.s["matlab_6x6_std"]
        sb6_n = (sb6 - self.s["sub_block_6x6_mean"]) / self.s["sub_block_6x6_std"]
        fp_n = (fp - self.s["fingerprint_mean"]) / self.s["fingerprint_std"]
        res_n = (res - self.s["residual_mean"]) / self.s["residual_std"]
        gap_n = (
            (sb6 - m6 - self.s["coupling_gap_mean"]) / self.s["coupling_gap_std"]
        ).astype(np.float32)

        dpx_n = np.full_like(m6_n, self.dpx[idx] / 180.0, dtype=np.float32)
        dpy_n = np.full_like(m6_n, self.dpy[idx] / 180.0, dtype=np.float32)

        peak = float(hf6.max())
        null_mask = (hf6 < (peak + NULL_THRESHOLD_DB)).astype(np.float32)
        null_depths = compute_topk_null_depths(
            hf6, k=self.top_k, null_threshold_db=NULL_THRESHOLD_DB)

        x = np.stack([m6_n, sb6_n, fp_n, dpx_n, dpy_n, gap_n], axis=0)
        if self.augment_noise and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=(4,) + m6_n.shape).astype(np.float32)
            x[:4] += noise

        return (
            torch.from_numpy(x),
            torch.from_numpy(res_n[None, ...]),
            torch.from_numpy(null_mask[None, ...]),
            torch.from_numpy(null_depths),
        )


def aux_scale(epoch: int) -> float:
    return min(1.0, epoch / AUX_RAMP_EPOCHS)


def train_one_epoch(model, ema, loader, opt, scaler, criterion, epoch):
    model.train()
    scale = aux_scale(epoch)
    totals = {k: [] for k in (
        "loss", "l1", "ssim", "null", "mainbeam", "composed_null",
        "eplane", "hplane", "shield", "null_w",
    )}
    total_steps = len(loader)
    for step, (x, y, null_mask, null_depths) in enumerate(loader, 1):
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)
        null_mask = null_mask.to(DEVICE, non_blocking=PIN_MEMORY)
        null_depths = null_depths.to(DEVICE, non_blocking=PIN_MEMORY)

        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
            out = model(x, return_aux=True)
            loss, stats = criterion(
                out, y, null_mask, null_depths, x[:, 1:2], x[:, 0:1], aux_scale=scale)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        ema.update(model)

        for key in totals:
            k = "l1_db" if key == "l1" else key
            if k == "shield":
                k = "shield_mean"
            if k == "null_w":
                k = "null_w_mean"
            if k in stats:
                totals[key].append(float(stats[k]))
        if step == 1 or step % STEP_PRINT_EVERY == 0 or step == total_steps:
            print(
                f"  step {step:03d}/{total_steps} | loss={stats['loss'].item():.4f}"
                f" l1={stats['l1_db'].item():.3f} null_c={stats['composed_null'].item():.3f}"
                f" e={stats['eplane'].item():.3f} h={stats['hplane'].item():.3f}"
                f" mb={stats['mainbeam'].item():.4f}",
                flush=True,
            )
    return {k: float(np.mean(v)) for k, v in totals.items()}


@torch.no_grad()
def evaluate_residual(model, ema, loader):
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


def _load_shared_encoder(model: FocusedRegionalMultiHeadResUNet, ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        print(f"  init-v2: {ckpt_path} not found, skipping", flush=True)
        return
    src = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    dst = model.state_dict()
    shared = [
        k for k in dst
        if k in src and src[k].shape == dst[k].shape
        and k.split(".")[0] in {"d1", "d2", "d3", "d4", "bottleneck", "attn", "u1", "u2", "u3", "u4"}
    ]
    for k in shared:
        dst[k] = src[k]
    model.load_state_dict(dst)
    print(f"  Warm-started encoder from {ckpt_path.name} ({len(shared)} tensors)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train matlab-relative multi-head v4")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-v4", action="store_true",
        help="Warm-start shared encoder from stage1_6x6_multihead_v4/best_generator.pt",
    )
    args = parser.parse_args()

    for p in (NPZ_EXTRAS, NORM_6X6, SPLIT_6X6):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run: python -m scripts.prep_stage1_6x6_matlab_rel"
            )

    print("Loading data ...", flush=True)
    e = np.load(NPZ_EXTRAS)
    matlab_6x6 = e["matlab_6x6"]
    sub_block_6x6 = e["sub_block_6x6"]
    hfss_6x6 = e["hfss_6x6"]
    fingerprint = e["fingerprint"]
    dpx = e["dpx"]
    dpy = e["dpy"]

    stats = extend_stats_with_coupling_gap(
        dict(np.load(NORM_6X6)),
        sub_block_6x6,
        matlab_6x6,
        np.load(SPLIT_6X6)["train"],
    )
    sp = np.load(SPLIT_6X6)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"Splits: {len(tr)} train / {len(va)} val / {len(te)} test (test reserved)", flush=True)

    tr_ds = Stage1_6x6_MultiHeadV4Dataset(
        tr, stats, matlab_6x6, sub_block_6x6, fingerprint, hfss_6x6, dpx, dpy,
        augment_noise=True)
    va_ds = Stage1_6x6_MultiHeadV4Dataset(
        va, stats, matlab_6x6, sub_block_6x6, fingerprint, hfss_6x6, dpx, dpy,
        augment_noise=False)
    tr_loader = DataLoader(
        tr_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, drop_last=True)
    va_loader = DataLoader(
        va_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY)

    res_mean_t = torch.from_numpy(stats["residual_mean"].astype(np.float32))[None, None].to(DEVICE)
    res_std_t = torch.from_numpy(stats["residual_std"].astype(np.float32))[None, None].to(DEVICE)
    sb_mean_t = torch.from_numpy(stats["sub_block_6x6_mean"].astype(np.float32))[None, None].to(DEVICE)
    sb_std_t = torch.from_numpy(stats["sub_block_6x6_std"].astype(np.float32))[None, None].to(DEVICE)
    m_mean_t = torch.from_numpy(stats["matlab_6x6_mean"].astype(np.float32))[None, None].to(DEVICE)
    m_std_t = torch.from_numpy(stats["matlab_6x6_std"].astype(np.float32))[None, None].to(DEVICE)

    G = FocusedRegionalMultiHeadResUNet(
        base=GEN_BASE, attn_heads=ATTN_HEADS, top_k=TOP_K_NULLS,
        sb_mean=stats["sub_block_6x6_mean"], sb_std=stats["sub_block_6x6_std"],
        res_mean=stats["residual_mean"], res_std=stats["residual_std"],
    ).to(DEVICE)
    print(f"G params: {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M  in_ch=6", flush=True)
    if DEVICE.type == "cuda":
        print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print(f"Device: {DEVICE}  *** WARNING: training on CPU ***", flush=True)

    last_g = CKPT_DIR / "last_generator.pt"
    last_g_ema = CKPT_DIR / "last_generator_ema.pt"
    if args.resume and last_g.exists():
        G.load_state_dict(torch.load(last_g, map_location=DEVICE, weights_only=True), strict=False)
        print(f"Resumed from {last_g.name}", flush=True)
    elif args.init_v4:
        _load_shared_encoder(G, CKPT_V4)
    elif last_g.exists() and not args.resume:
        print("Fresh start (use --resume to continue). Ignoring old checkpoints.", flush=True)

    opt = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR)
    scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
    criterion = FocusedRegionalLoss(
        stats["residual_mean"], stats["residual_std"],
        stats["sub_block_6x6_mean"], stats["sub_block_6x6_std"],
        matlab_mean=stats["matlab_6x6_mean"],
        matlab_std=stats["matlab_6x6_std"],
        compose_base="matlab",
    ).to(DEVICE)
    ema = EMA(G, decay=EMA_DECAY)
    if args.resume and last_g_ema.exists():
        ema.shadow = torch.load(last_g_ema, map_location=DEVICE, weights_only=True)

    best_score = float("inf")
    best_epoch = -1
    epochs_without_improve = 0

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        cur_lr = opt.param_groups[0]["lr"]
        print(
            f"\n=== Epoch {epoch}/{EPOCHS}  (lr={cur_lr:.2e}, aux={aux_scale(epoch):.2f}) ===",
            flush=True,
        )
        tr_stats = train_one_epoch(G, ema, tr_loader, opt, scaler, criterion, epoch)
        print(
            f"  train: loss={tr_stats['loss']:.4f}  l1={tr_stats['l1']:.3f}  "
            f"null_c={tr_stats['composed_null']:.3f}  eplane={tr_stats['eplane']:.3f}  "
            f"hplane={tr_stats['hplane']:.3f}  mainbeam={tr_stats['mainbeam']:.4f}  "
            f"a_base={float(G.alpha_base):.3f}  a_null={float(G.alpha_null):.3f}  "
            f"a_e={float(G.alpha_e):.3f}  a_h={float(G.alpha_h):.3f}",
            flush=True,
        )

        if epoch % VALIDATE_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            comp = evaluate_composed_matlab_rel(
                G, ema, va_loader, res_mean_t, res_std_t, m_mean_t, m_std_t, sb_mean_t, sb_std_t,
                amp_device_type=AMP_DEVICE_TYPE, use_amp=USE_AMP, device=DEVICE,
                pin_memory=PIN_MEMORY,
                val_weight_main=VAL_WEIGHT_MAIN,
                val_weight_main_sl=VAL_WEIGHT_MAIN_SL,
                val_weight_null=VAL_WEIGHT_NULL,
            )
            res_m = evaluate_residual(G, ema, va_loader)
            print(
                f"  val (EMA): score={comp['score']:.4f}  "
                f"main={comp['main_rmse']:.4f}  main+SL={comp['main_sl_rmse']:.4f}  "
                f"null={comp['null_rmse']:.4f}  full={comp['full_rmse']:.4f}  |  "
                f"res_n rmse={res_m['rmse']:.4f}",
                flush=True,
            )
            if comp["score"] < best_score - 1e-5:
                best_score = comp["score"]
                best_epoch = epoch
                epochs_without_improve = 0
                torch.save(ema.state_dict(), CKPT_DIR / "best_generator.pt")
                print(f"  * new best (val score {best_score:.5f}) saved", flush=True)
            else:
                epochs_without_improve += VALIDATE_EVERY
            sched.step(comp["score"])

        torch.save(G.state_dict(), CKPT_DIR / "last_generator.pt")
        torch.save(ema.state_dict(), CKPT_DIR / "last_generator_ema.pt")
        if epoch % SAVE_EVERY == 0:
            torch.save(ema.state_dict(), CKPT_DIR / f"generator_ema_epoch_{epoch:03d}.pt")

        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}; best epoch {best_epoch} "
                f"(val score {best_score:.5f})",
                flush=True,
            )
            break

    print(
        f"\nTraining done in {(time.time() - t0) / 60:.1f} min. Best epoch {best_epoch} "
        f"val score {best_score:.5f}. Checkpoints in {CKPT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
