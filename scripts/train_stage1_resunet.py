"""
Stage 1 at the 6x6 scale: ResUNet that learns the inter-block coupling residual
inside a 6x6 array. Same data and generator as train_stage1_6x6_cgan.py but
trained with reconstruction loss only (no PatchGAN discriminator).

Channel layout (5 input channels):
  ch0 : matlab_6x6_n       (analytical 6x6, peak-norm dB, z-score normalised)
  ch1 : sub_block_6x6_n    (within-array composition of the 9 sub-blocks)
  ch2 : fingerprint_n      (matlab_2x2 - hfss_2x2_mean, scale-invariant)
  ch3 : dphase_x / 180     (broadcast scalar)
  ch4 : dphase_y / 180

Target:
  residual_n = (hfss_6x6 - sub_block_6x6) normalised with residual stats
"""
import sys, time
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
from scripts.train_cgan_2to4_fusion_no_m4 import (
    EnhancedResUNetGenerator, EMA, ReconLoss,
    evaluate_loader, AMP_DEVICE_TYPE,
    GEN_BASE, ATTN_HEADS, EMA_DECAY,
    LAMBDA_L1, LAMBDA_SSIM, NULL_LOSS_WEIGHT, NOISE_STD,
    USE_AMP,
)

# ── Paths ───────────────────────────────────────────────────────────────────
NPZ_EXTRAS  = PROCESSED_DIR / "stage1_6x6_extras.npz"
NORM_6X6    = PROCESSED_DIR / "norm_stats_stage1_6x6.npz"
SPLIT_6X6   = PROCESSED_DIR / "split_indices_stage1_6x6.npz"
CKPT_DIR    = CHECKPOINTS_DIR / "stage1_6x6_resunet"
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

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


class Stage1_6x6_Dataset(Dataset):
    """5-channel input / 1-channel residual target at 6x6 scale."""

    def __init__(self, indices, stats,
                 matlab_6x6, sub_block_6x6, fingerprint, hfss_6x6,
                 dpx, dpy, augment_noise=False, noise_std=NOISE_STD):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.m6   = matlab_6x6
        self.sb6  = sub_block_6x6
        self.fp   = fingerprint
        self.hf6  = hfss_6x6
        self.dpx  = dpx
        self.dpy  = dpy
        self.augment_noise = augment_noise
        self.noise_std = noise_std
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

        x = np.stack([m6_n, sb6_n, fp_n, dpx_n, dpy_n], axis=0)

        if self.augment_noise and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std,
                                     size=(3,) + m6_n.shape).astype(np.float32)
            x[:3] += noise

        y = res_n[None, ...]
        null_mask = null_mask[None, ...]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(null_mask)


def train_one_epoch(generator, ema, loader, opt, scaler, recon_loss):
    generator.train()
    losses, l1s, ssims = [], [], []
    total_steps = len(loader)
    for step, (x, y, null_mask) in enumerate(loader, 1):
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)
        null_mask = null_mask.to(DEVICE, non_blocking=PIN_MEMORY)

        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
            y_hat = generator(x)
            loss, l1_val, ssim_val = recon_loss(y_hat, y, null_mask)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        ema.update(generator)

        losses.append(float(loss))
        l1s.append(float(l1_val))
        ssims.append(float(ssim_val))
        if step == 1 or step % STEP_PRINT_EVERY == 0 or step == total_steps:
            print(f"  step {step:03d}/{total_steps} | loss={loss.item():.4f}"
                  f" l1_dB={l1_val.item():.3f} ssim={ssim_val.item():.4f}", flush=True)
    return np.mean(losses), np.mean(l1s), np.mean(ssims)


def main():
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

    tr_ds = Stage1_6x6_Dataset(tr, stats, matlab_6x6, sub_block_6x6, fingerprint,
                                hfss_6x6, dpx, dpy, augment_noise=True,
                                noise_std=NOISE_STD)
    va_ds = Stage1_6x6_Dataset(va, stats, matlab_6x6, sub_block_6x6, fingerprint,
                                hfss_6x6, dpx, dpy, augment_noise=False)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                           drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    G = EnhancedResUNetGenerator(in_ch=5, out_ch=1, base=GEN_BASE,
                                  attn_heads=ATTN_HEADS).to(DEVICE)
    g_params = sum(p.numel() for p in G.parameters())
    print(f"G params: {g_params/1e6:.2f}M", flush=True)

    last_g     = CKPT_DIR / "last_generator.pt"
    last_g_ema = CKPT_DIR / "last_generator_ema.pt"
    if last_g.exists():
        G.load_state_dict(torch.load(last_g, map_location=DEVICE))
        print(f"Resumed G live weights from {last_g.name}", flush=True)

    opt = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR)
    scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
    recon_loss = ReconLoss(stats["residual_mean"], stats["residual_std"]).to(DEVICE)
    ema = EMA(G, decay=EMA_DECAY)
    if last_g_ema.exists():
        ema.shadow = torch.load(last_g_ema, map_location=DEVICE)
        print(f"Resumed EMA shadow from {last_g_ema.name}", flush=True)

    best_rmse = float("inf"); best_epoch = -1; epochs_without_improve = 0
    if last_g_ema.exists():
        seed_metrics = evaluate_loader(G, ema, va_loader)
        best_rmse = seed_metrics["rmse"]
        print(f"Seeded best_rmse from resumed EMA: {best_rmse:.5f}", flush=True)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        cur_lr = opt.param_groups[0]["lr"]
        print(f"\n=== Epoch {epoch}/{EPOCHS}  (lr={cur_lr:.2e}) ===", flush=True)
        loss_avg, l1_avg, ssim_avg = train_one_epoch(
            G, ema, tr_loader, opt, scaler, recon_loss)
        print(f"  train: loss={loss_avg:.4f}  l1_dB={l1_avg:.3f}  ssim={ssim_avg:.4f}",
              flush=True)

        if epoch % VALIDATE_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            m = evaluate_loader(G, ema, va_loader)
            print(f"  val (EMA): rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  "
                  f"r={m['pearson']:.4f}", flush=True)
            if m["rmse"] < best_rmse - 1e-5:
                best_rmse = m["rmse"]; best_epoch = epoch; epochs_without_improve = 0
                torch.save(ema.state_dict(), CKPT_DIR / "best_generator.pt")
                print(f"  * new best (val rmse {best_rmse:.5f}) saved", flush=True)
            else:
                epochs_without_improve += VALIDATE_EVERY
                print(f"  (no improvement for {epochs_without_improve} epochs)", flush=True)
            sched.step(m["rmse"])

        torch.save(G.state_dict(),  CKPT_DIR / "last_generator.pt")
        torch.save(ema.state_dict(), CKPT_DIR / "last_generator_ema.pt")
        if epoch % SAVE_EVERY == 0:
            torch.save(ema.state_dict(), CKPT_DIR / f"generator_ema_epoch_{epoch:03d}.pt")

        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}; best epoch {best_epoch} "
                  f"(val rmse {best_rmse:.5f})", flush=True)
            break

    total = (time.time() - t0) / 60
    print(f"\nTraining done in {total:.1f} min. Best epoch {best_epoch} "
          f"val rmse {best_rmse:.5f}. Checkpoints in {CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
