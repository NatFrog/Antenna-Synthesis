"""
Train an enhanced no-matlab_4x4 ablation of the synthetic 2-to-4 fusion cGAN.

Layers a stack of capacity / loss / training-procedure improvements on top of
train_cgan_2to4_fusion.py while keeping the no-m4 ablation premise:

  * Drops matlab_4x4 from the input stack (the ablation).
  * Adds a residual channel (matlab_2x2_n - hfss_pred_2x2_n) to compensate.
  * Wider generator (base=48) with multi-head self-attention at the bottleneck.
  * Reconstruction loss = null-weighted L1 (dB-space) + (1 - SSIM(dB)) * 0.5.
  * Lambda_L1 = 150 (was 100).
  * Pix2pix-style LR (2e-4 / 2e-5) with constant-then-linear-decay schedule.
  * EMA on generator weights; EMA copy used for val + final best checkpoint.
  * Longer training (EPOCHS=200, EARLY_STOP_PATIENCE=40).
  * Gaussian noise augmentation on the input channels (training only).

Inputs (channel-wise):
  ch0: matlab_2x2          (normalised with norm_stats_2x2.npz)
  ch1: hfss_pred_2x2       (normalised with norm_stats_2x2.npz)
  ch2: residual            (matlab_2x2_n - hfss_pred_2x2_n)
  ch3: dphase_x / 180      (broadcast scalar)
  ch4: dphase_y / 180      (broadcast scalar)
Target:
  hfss_4x4                 (normalised with norm_stats.npz)

Usage:
    python -m scripts.train_cgan_2to4_fusion_no_m4
"""

import sys
import time
import copy
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, DEVICE,
    BATCH_SIZE, MAX_EPOCHS, RANDOM_SEED, NULL_THRESHOLD_DB,
)
from src.training.metrics import rmse, mae, pearson_correlation
from scripts.train_cgan import PatchDiscriminator
from scripts.train_cgan_2to4_fusion import (
    ResidualBlock, DownRes, UpRes,  # building blocks of the original ResUNet
)

# ─── Paths ───────────────────────────────────────────────────────────────────
HDF5 = PROCESSED_DIR / "antenna_data_2to4.h5"
NORM_2X2 = PROCESSED_DIR / "norm_stats_2x2.npz"
NORM_4X4 = PROCESSED_DIR / "norm_stats.npz"
SPLITS = PROCESSED_DIR / "split_indices_2to4.npz"
CKPT_DIR = CHECKPOINTS_DIR / "cgan_resunet_patchgan_2to4_no_m4"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hyperparameters ────────────────────────────────────────────────────────
EPOCHS = min(MAX_EPOCHS, 200)
BATCH = BATCH_SIZE                   # 16, validated to fit at base=48 on 16 GB VRAM
LR_G = 2e-4                          # Pix2pix-standard
LR_D = 2e-5
# ReduceLROnPlateau: halve LR when val RMSE stops improving, no fixed schedule.
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 4                 # val checks (= 8 epochs at VALIDATE_EVERY=2)
PLATEAU_MIN_LR = 1e-6
LAMBDA_L1 = 150.0                    # heavier reconstruction anchor
LAMBDA_SSIM = 0.5                    # SSIM term weight (loss = 1 - SSIM)
NULL_LOSS_WEIGHT = 3.0               # null pixels weighted ~3x in L1
NULL_DB_THRESHOLD = NULL_THRESHOLD_DB # -20 dB from per-sample peak
LABEL_SMOOTH_REAL = 0.9
LABEL_SMOOTH_FAKE = 0.1
NOISE_STD = 0.05                     # input augmentation, normalised-space std
EMA_DECAY = 0.999
GEN_BASE = 48                        # was 32; safely fits at BATCH=16 on 16 GB
ATTN_HEADS = 8
NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda"
USE_AMP = False                      # match the rest of the project
VALIDATE_EVERY = 2
EARLY_STOP_PATIENCE = 40
SAVE_EVERY = 10
STEP_PRINT_EVERY = 25

# Reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# ─── Dataset ─────────────────────────────────────────────────────────────────
class FusionDatasetNoM4(Dataset):
    """4 + 1-residual = 5 input channels; returns (x, y, null_mask).

    null_mask is a binary float tensor (1.0 inside null regions, 0.0 elsewhere)
    in target dB space, used by the null-weighted L1 loss. Null = pixels below
    per-sample peak + NULL_DB_THRESHOLD (-20 dB).
    """

    def __init__(self, h5_path, indices,
                 mean_2x2, std_2x2, mean_4x4, std_4x4,
                 augment_noise=False, noise_std=NOISE_STD):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean_2x2 = mean_2x2.astype(np.float32)
        self.std_2x2 = std_2x2.astype(np.float32)
        self.mean_4x4 = mean_4x4.astype(np.float32)
        self.std_4x4 = std_4x4.astype(np.float32)
        self.augment_noise = augment_noise
        self.noise_std = noise_std
        self._file = None

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        f = self._get_file()
        idx = int(self.indices[i])

        m2 = f["matlab_2x2"][idx].astype(np.float32)
        hp2 = f["hfss_pred_2x2"][idx].astype(np.float32)
        h4 = f["hfss_4x4"][idx].astype(np.float32)
        meta = f["metadata"][idx].astype(np.float32)

        m2_n = (m2 - self.mean_2x2) / self.std_2x2
        hp2_n = (hp2 - self.mean_2x2) / self.std_2x2
        residual_n = m2_n - hp2_n
        h4_n = (h4 - self.mean_4x4) / self.std_4x4

        # Null mask in dB space (independent of normalization)
        peak = float(h4.max())
        null_mask = (h4 < (peak + NULL_DB_THRESHOLD)).astype(np.float32)

        dphase_x = np.full_like(m2_n, fill_value=meta[0] / 180.0, dtype=np.float32)
        dphase_y = np.full_like(m2_n, fill_value=meta[1] / 180.0, dtype=np.float32)

        x = np.stack([m2_n, hp2_n, residual_n, dphase_x, dphase_y], axis=0)  # (5, H, W)

        if self.augment_noise and self.noise_std > 0:
            # Apply only to the three pattern-derived channels (0, 1, 2).
            noise = np.random.normal(0, self.noise_std, size=(3,) + m2_n.shape).astype(np.float32)
            x[:3] += noise

        y = h4_n[None, ...]  # (1, H, W)
        null_mask = null_mask[None, ...]  # (1, H, W)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(null_mask)


# ─── Self-attention block (used at bottleneck) ───────────────────────────────
class SelfAttention2D(nn.Module):
    """Multi-head self-attention over flattened spatial dims (BCHW -> BNC).

    With the bottleneck spatial size of ~12 x 23 = 276 tokens at base=48, the
    attention matrix is small (276x276) and adds negligible memory.
    """

    def __init__(self, channels, num_heads=ATTN_HEADS):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} not divisible by heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        h_in = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, n)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                # (B, heads, head_dim, N)
        q = q.transpose(-1, -2)                                  # (B, heads, N, head_dim)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        out = (attn @ v).transpose(-1, -2).reshape(b, c, h, w)
        return h_in + self.proj(out)


# ─── Enhanced ResUNet generator (wider + attention bottleneck) ───────────────
class EnhancedResUNetGenerator(nn.Module):
    def __init__(self, in_ch=5, out_ch=1, base=GEN_BASE, attn_heads=ATTN_HEADS):
        super().__init__()
        self.d1 = DownRes(in_ch, base)
        self.d2 = DownRes(base, base * 2)
        self.d3 = DownRes(base * 2, base * 4)
        self.d4 = DownRes(base * 4, base * 8)
        self.bottleneck = ResidualBlock(base * 8, base * 16)
        self.attn = SelfAttention2D(base * 16, num_heads=attn_heads)
        self.u1 = UpRes(base * 16, base * 8, base * 8)
        self.u2 = UpRes(base * 8, base * 4, base * 4)
        self.u3 = UpRes(base * 4, base * 2, base * 2)
        self.u4 = UpRes(base * 2, base, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x_in = x
        s1, p1 = self.d1(x)
        s2, p2 = self.d2(p1)
        s3, p3 = self.d3(p2)
        s4, p4 = self.d4(p3)
        b = self.bottleneck(p4)
        b = self.attn(b)
        x = self.u1(b, s4)
        x = self.u2(x, s3)
        x = self.u3(x, s2)
        x = self.u4(x, s1)
        out = self.out(x)
        if out.shape[-2:] != x_in.shape[-2:]:
            out = F.interpolate(out, size=x_in.shape[-2:], mode="bilinear", align_corners=False)
        return out


# ─── EMA wrapper ─────────────────────────────────────────────────────────────
class EMA:
    """Exponential moving average over a module's parameters and buffers.

    Apply via context-manager-style swap: model uses EMA weights while the
    swap is active, then restores live weights on exit.
    """

    def __init__(self, module, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(module.state_dict())
        for v in self.shadow.values():
            if torch.is_tensor(v) and v.is_floating_point():
                v.requires_grad_(False)

    @torch.no_grad()
    def update(self, module):
        msd = module.state_dict()
        for k, v in msd.items():
            if not torch.is_tensor(v):
                continue
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return self.shadow

    @torch.no_grad()
    def swap_into(self, module):
        live = copy.deepcopy(module.state_dict())
        module.load_state_dict(self.shadow)
        return live

    @torch.no_grad()
    def restore(self, module, live):
        module.load_state_dict(live)


# ─── SSIM (single-scale, Gaussian-windowed, computed in dB space) ────────────
def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w_2d = g[:, None] * g[None, :]
    return w_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)


def ssim_dB(pred_db, target_db, data_range=60.0, window_size=11, sigma=1.5):
    """SSIM averaged over a batch of single-channel patterns in dB space."""
    win = _gaussian_window(window_size, sigma, pred_db.device, pred_db.dtype)
    pad = window_size // 2

    mu_p = F.conv2d(pred_db, win, padding=pad)
    mu_t = F.conv2d(target_db, win, padding=pad)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sig_p = F.conv2d(pred_db * pred_db, win, padding=pad) - mu_p2
    sig_t = F.conv2d(target_db * target_db, win, padding=pad) - mu_t2
    sig_pt = F.conv2d(pred_db * target_db, win, padding=pad) - mu_pt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / ((mu_p2 + mu_t2 + c1) * (sig_p + sig_t + c2))
    return ssim_map.mean()


# ─── Combined reconstruction loss (null-weighted L1 + SSIM, in dB space) ─────
class ReconLoss(nn.Module):
    """null-weighted L1 + (1 - SSIM), both evaluated in dB space.

    The training tensors live in normalised space; we denormalise with the per-
    pixel mean/std (registered as buffers) before computing the loss so the
    null mask threshold and SSIM data range have physical meaning.
    """

    def __init__(self, mean_4x4, std_4x4,
                 lambda_l1=LAMBDA_L1, lambda_ssim=LAMBDA_SSIM,
                 null_weight=NULL_LOSS_WEIGHT):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.null_weight = null_weight
        # Per-pixel norm stats as buffers so .to(device) moves them automatically.
        self.register_buffer("mean", torch.from_numpy(mean_4x4.astype(np.float32))[None, None])
        self.register_buffer("std", torch.from_numpy(std_4x4.astype(np.float32))[None, None])

    def forward(self, pred_n, target_n, null_mask):
        # Denormalise to dB space.
        pred_db = pred_n * self.std + self.mean
        target_db = target_n * self.std + self.mean

        # Null-weighted L1 in dB.
        err = torch.abs(pred_db - target_db)
        weight = 1.0 + (self.null_weight - 1.0) * null_mask
        l1 = (err * weight).mean()

        # SSIM term in dB (single channel).
        ssim_val = ssim_dB(pred_db, target_db)
        ssim_loss = 1.0 - ssim_val

        total = self.lambda_l1 * l1 + self.lambda_ssim * ssim_loss
        return total, l1.detach(), ssim_val.detach()


# ─── Training loop ───────────────────────────────────────────────────────────
AMP_DEVICE_TYPE = "cuda" if DEVICE.type == "cuda" else "cpu"


def train_one_epoch(generator, discriminator, ema, loader,
                    opt_g, opt_d, scaler_g, scaler_d, bce, recon_loss):
    generator.train(); discriminator.train()
    g_losses, d_losses, l1s, ssims = [], [], [], []
    total_steps = len(loader)
    for step, (x, y, null_mask) in enumerate(loader, 1):
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)
        null_mask = null_mask.to(DEVICE, non_blocking=PIN_MEMORY)

        # D step
        with torch.no_grad():
            with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
                y_fake = generator(x)
        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
            pr = discriminator(x, y)
            pf = discriminator(x, y_fake)
            rl = torch.full_like(pr, LABEL_SMOOTH_REAL)
            fl = torch.full_like(pf, LABEL_SMOOTH_FAKE)
            d_loss = 0.5 * (bce(pr, rl) + bce(pf, fl))
        opt_d.zero_grad(set_to_none=True)
        scaler_d.scale(d_loss).backward()
        scaler_d.step(opt_d); scaler_d.update()

        # G step
        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
            y_fake = generator(x)
            pf_g = discriminator(x, y_fake)
            adv = bce(pf_g, torch.ones_like(pf_g))
            recon, l1_val, ssim_val = recon_loss(y_fake, y, null_mask)
            g_loss = adv + recon
        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(g_loss).backward()
        scaler_g.step(opt_g); scaler_g.update()

        ema.update(generator)

        g_losses.append(float(g_loss)); d_losses.append(float(d_loss))
        l1s.append(float(l1_val)); ssims.append(float(ssim_val))
        if step == 1 or step % STEP_PRINT_EVERY == 0 or step == total_steps:
            print(f"  step {step:03d}/{total_steps} | g={g_loss.item():.4f} d={d_loss.item():.4f}"
                  f" l1_dB={l1_val.item():.3f} ssim={ssim_val.item():.4f}", flush=True)
    return (np.mean(g_losses), np.mean(d_losses),
            np.mean(l1s), np.mean(ssims))


@torch.no_grad()
def evaluate_loader(generator, ema, loader):
    """Validation with EMA weights swapped in temporarily."""
    live = ema.swap_into(generator)
    try:
        generator.eval()
        yt, yp = [], []
        for x, y, _null in loader:
            x = x.to(DEVICE, non_blocking=PIN_MEMORY)
            y = y.to(DEVICE, non_blocking=PIN_MEMORY)
            with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
                y_hat = generator(x)
            yt.append(y.cpu().numpy()); yp.append(y_hat.cpu().numpy())
        yt = np.concatenate(yt).reshape(-1); yp = np.concatenate(yp).reshape(-1)
        return {"rmse": float(rmse(yt, yp)),
                "mae":  float(mae(yt, yp)),
                "pearson": float(pearson_correlation(yt, yp))}
    finally:
        ema.restore(generator, live)


def main():
    for p in (HDF5, NORM_2X2, NORM_4X4, SPLITS):
        if not p.exists():
            raise FileNotFoundError(p)

    s_2x2 = np.load(NORM_2X2); s_4x4 = np.load(NORM_4X4)
    mean_2x2 = s_2x2["mean"].astype(np.float32)
    std_2x2 = np.maximum(s_2x2["std"].astype(np.float32), 1e-6)
    mean_4x4 = s_4x4["mean"].astype(np.float32)
    std_4x4 = np.maximum(s_4x4["std"].astype(np.float32), 1e-6)

    sp = np.load(SPLITS)
    tr, va, te = sp["train"].astype(np.int64), sp["val"].astype(np.int64), sp["test"].astype(np.int64)
    print(f"Splits: {len(tr)} train / {len(va)} val / {len(te)} test", flush=True)

    tr_ds = FusionDatasetNoM4(HDF5, tr, mean_2x2, std_2x2, mean_4x4, std_4x4,
                              augment_noise=True, noise_std=NOISE_STD)
    va_ds = FusionDatasetNoM4(HDF5, va, mean_2x2, std_2x2, mean_4x4, std_4x4,
                              augment_noise=False)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    G = EnhancedResUNetGenerator(in_ch=5, out_ch=1, base=GEN_BASE,
                                 attn_heads=ATTN_HEADS).to(DEVICE)
    D = PatchDiscriminator(in_channels=6, base=64).to(DEVICE)
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"G params: {g_params/1e6:.2f}M (base={GEN_BASE}, attn_heads={ATTN_HEADS})   "
          f"D params: {d_params/1e6:.2f}M", flush=True)

    # Resume: if last_generator.pt / last_generator_ema.pt / last_discriminator.pt
    # exist, warm-start from them. Optimiser state is NOT preserved (Adam moments
    # rebuild within ~1-2 epochs); the per-pixel loss anchor is heavy enough that
    # this transient is harmless.
    last_g = CKPT_DIR / "last_generator.pt"
    last_g_ema = CKPT_DIR / "last_generator_ema.pt"
    last_d = CKPT_DIR / "last_discriminator.pt"
    if last_g.exists():
        G.load_state_dict(torch.load(last_g, map_location=DEVICE))
        print(f"Resumed G live weights from {last_g.name}", flush=True)
    if last_d.exists():
        D.load_state_dict(torch.load(last_d, map_location=DEVICE))
        print(f"Resumed D weights from {last_d.name}", flush=True)
    else:
        print("No D checkpoint found — starting D from random init.", flush=True)

    opt_g = optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    sched_g = optim.lr_scheduler.ReduceLROnPlateau(
        opt_g, mode="min", factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR,
    )
    sched_d = optim.lr_scheduler.ReduceLROnPlateau(
        opt_d, mode="min", factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR,
    )
    scaler_g = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
    scaler_d = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
    bce = nn.BCEWithLogitsLoss()
    recon_loss = ReconLoss(mean_4x4, std_4x4).to(DEVICE)
    ema = EMA(G, decay=EMA_DECAY)
    if last_g_ema.exists():
        ema.shadow = torch.load(last_g_ema, map_location=DEVICE)
        print(f"Resumed EMA shadow from {last_g_ema.name}", flush=True)

    # Seed best_rmse from the resumed EMA so the first post-resume validation
    # doesn't overwrite a better best_generator.pt with a slightly worse one.
    best_rmse = float("inf"); best_epoch = -1; epochs_without_improve = 0
    if last_g_ema.exists():
        seed_metrics = evaluate_loader(G, ema, va_loader)
        best_rmse = seed_metrics["rmse"]
        print(f"Seeded best_rmse from resumed EMA: {best_rmse:.5f} "
              f"(mae={seed_metrics['mae']:.5f}, r={seed_metrics['pearson']:.5f})",
              flush=True)
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        cur_lr_g = opt_g.param_groups[0]["lr"]
        print(f"\n=== Epoch {epoch}/{EPOCHS}  (G_lr={cur_lr_g:.2e}) ===", flush=True)
        g_avg, d_avg, l1_avg, ssim_avg = train_one_epoch(
            G, D, ema, tr_loader, opt_g, opt_d, scaler_g, scaler_d, bce, recon_loss
        )
        print(f"  train: g_loss={g_avg:.4f}  d_loss={d_avg:.4f}  "
              f"l1_dB={l1_avg:.3f}  ssim={ssim_avg:.4f}", flush=True)

        if epoch % VALIDATE_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            m = evaluate_loader(G, ema, va_loader)
            print(f"  val (EMA): rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  r={m['pearson']:.4f}",
                  flush=True)
            if m["rmse"] < best_rmse - 1e-5:
                best_rmse = m["rmse"]; best_epoch = epoch; epochs_without_improve = 0
                # Save EMA weights as the "best" checkpoint (what we evaluate against).
                torch.save(ema.state_dict(), CKPT_DIR / "best_generator.pt")
                print(f"  * new best (val rmse {best_rmse:.5f}) saved to best_generator.pt",
                      flush=True)
            else:
                epochs_without_improve += VALIDATE_EVERY
                print(f"  (no improvement for {epochs_without_improve} epochs)", flush=True)

            # Step ReduceLROnPlateau on the validation metric.
            sched_g.step(m["rmse"]); sched_d.step(m["rmse"])

        torch.save(G.state_dict(), CKPT_DIR / "last_generator.pt")
        torch.save(ema.state_dict(), CKPT_DIR / "last_generator_ema.pt")
        torch.save(D.state_dict(), CKPT_DIR / "last_discriminator.pt")
        if epoch % SAVE_EVERY == 0:
            torch.save(ema.state_dict(), CKPT_DIR / f"generator_ema_epoch_{epoch:03d}.pt")

        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}; best epoch was {best_epoch} "
                  f"(val rmse {best_rmse:.5f})", flush=True)
            break

    total = (time.time() - t0) / 60
    print(f"\nTraining done in {total:.1f} min. Best epoch {best_epoch} "
          f"val rmse {best_rmse:.5f}.  Checkpoints in {CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
