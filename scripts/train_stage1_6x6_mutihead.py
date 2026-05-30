"""
Stage 1 at 6x6 scale — multi-head gated ResUNet.

Usage:
    python -m scripts.train_stage1_6x6_multihead          # fresh start
    python -m scripts.train_stage1_6x6_multihead --resume # continue last run

Delete checkpoints/stage1_6x6_multihead/ before a fresh run if the gate
architecture changed — old weights are incompatible.
"""
import sys, time, argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, DEVICE,
    BATCH_SIZE, MAX_EPOCHS, RANDOM_SEED, NULL_THRESHOLD_DB,
)
from src.training.metrics import rmse, mae, pearson_correlation
from src.models.stage1_multihead_resunet import (
    GatedMultiHeadResUNet,
    MultiHeadStage1Loss,
    compute_psl_db,
    compute_topk_null_depths,
)
from scripts.train_cgan_2to4_fusion_no_m4 import (
    EMA, AMP_DEVICE_TYPE, GEN_BASE, ATTN_HEADS, EMA_DECAY, NOISE_STD, USE_AMP,
)

# ── Paths ───────────────────────────────────────────────────────────────────
NPZ_EXTRAS  = PROCESSED_DIR / "stage1_6x6_extras.npz"
NORM_6X6    = PROCESSED_DIR / "norm_stats_stage1_6x6.npz"
SPLIT_6X6   = PROCESSED_DIR / "split_indices_stage1_6x6.npz"
CKPT_DIR    = CHECKPOINTS_DIR / "stage1_6x6_multihead"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ─────────────────────────────────────────────────────────
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
TOP_K_NULLS = 10
LAMBDA_PSL = 2.0
LAMBDA_NULL = 5.0
LAMBDA_MAINBEAM = 10.0
AUX_RAMP_EPOCHS = 20

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


class Stage1_6x6_MultiHeadDataset(Dataset):
    """5-channel input, residual target, null mask, PSL + top-K null depth labels."""

    def __init__(self, indices, stats,
                 matlab_6x6, sub_block_6x6, fingerprint, hfss_6x6,
                 dpx, dpy, augment_noise=False, noise_std=NOISE_STD,
                 top_k=TOP_K_NULLS):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.m6   = matlab_6x6
        self.sb6  = sub_block_6x6
        self.fp   = fingerprint
        self.hf6  = hfss_6x6
        self.dpx  = dpx
        self.dpy  = dpy
        self.augment_noise = augment_noise
        self.noise_std = noise_std
        self.top_k = top_k
        self.s = {k: v.astype(np.float32) for k, v in stats.items()}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])

        m6  = self.m6[idx].astype(np.float32)
        sb6 = self.sb6[idx].astype(np.float32)
        fp  = self.fp[idx].astype(np.float32)
        hf6 = self.hf6[idx].astype(np.float32)
        res = hf6 - sb6

        m6_n  = (m6  - self.s["matlab_6x6_mean"])    / self.s["matlab_6x6_std"]
        sb6_n = (sb6 - self.s["sub_block_6x6_mean"]) / self.s["sub_block_6x6_std"]
        fp_n  = (fp  - self.s["fingerprint_mean"])   / self.s["fingerprint_std"]
        res_n = (res - self.s["residual_mean"])      / self.s["residual_std"]

        dpx_n = np.full_like(m6_n, self.dpx[idx] / 180.0, dtype=np.float32)
        dpy_n = np.full_like(m6_n, self.dpy[idx] / 180.0, dtype=np.float32)

        peak = float(hf6.max())
        null_mask = (hf6 < (peak + NULL_THRESHOLD_DB)).astype(np.float32)
        psl = np.float32(compute_psl_db(hf6))
        null_depths = compute_topk_null_depths(hf6, k=self.top_k, null_threshold_db=NULL_THRESHOLD_DB)

        x = np.stack([m6_n, sb6_n, fp_n, dpx_n, dpy_n], axis=0)

        if self.augment_noise and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std,
                                     size=(3,) + m6_n.shape).astype(np.float32)
            x[:3] += noise

        return (
            torch.from_numpy(x),
            torch.from_numpy(res_n[None, ...]),
            torch.from_numpy(null_mask[None, ...]),
            torch.tensor(psl),
            torch.from_numpy(null_depths),
        )


def aux_scale(epoch: int) -> float:
    return min(1.0, epoch / AUX_RAMP_EPOCHS)


def train_one_epoch(model, ema, loader, opt, scaler, criterion, epoch):
    model.train()
    scale = aux_scale(epoch)
    totals = {"loss": [], "l1": [], "ssim": [], "psl": [], "null": [],
              "mainbeam": [], "gate": []}
    total_steps = len(loader)
    for step, (x, y, null_mask, psl, null_depths) in enumerate(loader, 1):
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)
        null_mask = null_mask.to(DEVICE, non_blocking=PIN_MEMORY)
        psl = psl.to(DEVICE, non_blocking=PIN_MEMORY)
        null_depths = null_depths.to(DEVICE, non_blocking=PIN_MEMORY)

        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
            out = model(x, return_aux=True)
            loss, stats = criterion(
                out, y, null_mask, psl, null_depths, x[:, 1:2], aux_scale=scale)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        ema.update(model)

        totals["loss"].append(stats["loss"].item())
        for k in ("l1_db", "ssim", "psl", "null", "mainbeam", "gate_mean"):
            key = "l1" if k == "l1_db" else k.replace("_mean", "")
            totals[key].append(float(stats[k]))
        if step == 1 or step % STEP_PRINT_EVERY == 0 or step == total_steps:
            print(f"  step {step:03d}/{total_steps} | loss={stats['loss'].item():.4f}"
                  f" l1={stats['l1_db'].item():.3f} psl={stats['psl'].item():.3f}"
                  f" null={stats['null'].item():.3f} gate={stats['gate_mean'].item():.3f}",
                  flush=True)
    return {k: float(np.mean(v)) for k, v in totals.items()}


@torch.no_grad()
def evaluate_loader(model, ema, loader):
    live = ema.swap_into(model)
    try:
        model.eval()
        yt, yp = [], []
        for x, y, *_rest in loader:
            x = x.to(DEVICE, non_blocking=PIN_MEMORY)
            y = y.to(DEVICE, non_blocking=PIN_MEMORY)
            with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
                y_hat = model(x)
            yt.append(y.cpu().numpy())
            yp.append(y_hat.cpu().numpy())
        yt = np.concatenate(yt).reshape(-1)
        yp = np.concatenate(yp).reshape(-1)
        return {"rmse": float(rmse(yt, yp)),
                "mae":  float(mae(yt, yp)),
                "pearson": float(pearson_correlation(yt, yp))}
    finally:
        ema.restore(model, live)


def _checkpoint_compatible(path: Path) -> bool:
    """Reject checkpoints from the broken gate (learnable margin < 0)."""
    try:
        sd = torch.load(path, map_location="cpu")
        if "gate_margin" in sd and float(sd["gate_margin"]) < 0:
            return False
    except Exception:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last_generator.pt if compatible")
    args = parser.parse_args()

    for p in (NPZ_EXTRAS, NORM_6X6, SPLIT_6X6):
        if not p.exists():
            raise FileNotFoundError(p)

    print("Loading data ...", flush=True)
    e = np.load(NPZ_EXTRAS)
    matlab_6x6    = e["matlab_6x6"]
    sub_block_6x6 = e["sub_block_6x6"]
    hfss_6x6      = e["hfss_6x6"]
    fingerprint   = e["fingerprint"]
    dpx           = e["dpx"]
    dpy           = e["dpy"]

    stats = dict(np.load(NORM_6X6))
    sp = np.load(SPLIT_6X6)
    tr = sp["train"]; va = sp["val"]; te = sp["test"]
    print(f"Splits: {len(tr)} train / {len(va)} val / {len(te)} test (test reserved)",
          flush=True)

    tr_ds = Stage1_6x6_MultiHeadDataset(tr, stats, matlab_6x6, sub_block_6x6, fingerprint,
                                         hfss_6x6, dpx, dpy, augment_noise=True)
    va_ds = Stage1_6x6_MultiHeadDataset(va, stats, matlab_6x6, sub_block_6x6, fingerprint,
                                         hfss_6x6, dpx, dpy, augment_noise=False)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                           drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    G = GatedMultiHeadResUNet(
        in_ch=5, base=GEN_BASE, attn_heads=ATTN_HEADS, top_k=TOP_K_NULLS,
        sb_mean=stats["sub_block_6x6_mean"], sb_std=stats["sub_block_6x6_std"],
        res_mean=stats["residual_mean"], res_std=stats["residual_std"],
    ).to(DEVICE)
    print(f"G params: {sum(p.numel() for p in G.parameters())/1e6:.2f}M", flush=True)
    if DEVICE.type == "cuda":
        print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print(f"Device: {DEVICE}  *** WARNING: training on CPU ***", flush=True)

    last_g = CKPT_DIR / "last_generator.pt"
    last_g_ema = CKPT_DIR / "last_generator_ema.pt"
    if args.resume and last_g.exists() and _checkpoint_compatible(last_g):
        G.load_state_dict(torch.load(last_g, map_location=DEVICE))
        print(f"Resumed from {last_g.name}", flush=True)
    elif last_g.exists() and not args.resume:
        print("Fresh start (use --resume to continue). Ignoring old checkpoints.", flush=True)
    elif last_g.exists() and not _checkpoint_compatible(last_g):
        print("Old checkpoint has incompatible gate — starting fresh.", flush=True)

    opt = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR)
    scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
    criterion = MultiHeadStage1Loss(
        stats["residual_mean"], stats["residual_std"],
        stats["sub_block_6x6_mean"], stats["sub_block_6x6_std"],
        lambda_psl=LAMBDA_PSL, lambda_null=LAMBDA_NULL,
        lambda_mainbeam=LAMBDA_MAINBEAM,
    ).to(DEVICE)
    ema = EMA(G, decay=EMA_DECAY)
    if args.resume and last_g_ema.exists() and _checkpoint_compatible(last_g):
        ema.shadow = torch.load(last_g_ema, map_location=DEVICE)

    best_rmse = float("inf"); best_epoch = -1; epochs_without_improve = 0

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        cur_lr = opt.param_groups[0]["lr"]
        print(f"\n=== Epoch {epoch}/{EPOCHS}  (lr={cur_lr:.2e}, aux={aux_scale(epoch):.2f}) ===",
              flush=True)
        tr_stats = train_one_epoch(G, ema, tr_loader, opt, scaler, criterion, epoch)
        print(f"  train: loss={tr_stats['loss']:.4f}  l1={tr_stats['l1']:.3f}  "
              f"psl={tr_stats['psl']:.3f}  null={tr_stats['null']:.3f}  "
              f"mainbeam={tr_stats['mainbeam']:.4f}  gate={tr_stats['gate']:.3f}",
              flush=True)

        if epoch % VALIDATE_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            m = evaluate_loader(G, ema, va_loader)
            print(f"  val (EMA): rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  "
                  f"r={m['pearson']:.4f}  (zero baseline rmse≈0.989)", flush=True)
            if m["rmse"] < best_rmse - 1e-5:
                best_rmse = m["rmse"]; best_epoch = epoch; epochs_without_improve = 0
                torch.save(ema.state_dict(), CKPT_DIR / "best_generator.pt")
                print(f"  * new best (val rmse {best_rmse:.5f}) saved", flush=True)
            else:
                epochs_without_improve += VALIDATE_EVERY
            sched.step(m["rmse"])

        torch.save(G.state_dict(), CKPT_DIR / "last_generator.pt")
        torch.save(ema.state_dict(), CKPT_DIR / "last_generator_ema.pt")
        if epoch % SAVE_EVERY == 0:
            torch.save(ema.state_dict(), CKPT_DIR / f"generator_ema_epoch_{epoch:03d}.pt")

        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}; best epoch {best_epoch} "
                  f"(val rmse {best_rmse:.5f})", flush=True)
            break

    print(f"\nTraining done in {(time.time()-t0)/60:.1f} min. Best epoch {best_epoch} "
          f"val rmse {best_rmse:.5f}. Checkpoints in {CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
