"""
Train the 4-to-8 sub-array synthesis cGAN.

This is the next-scale analogue of train_cgan_2to4_fusion_no_m4.py. The same
EnhancedResUNet generator + PatchDiscriminator + EMA + null-weighted L1 +
dB-SSIM + ReduceLROnPlateau setup, applied at the 4x4 -> 8x8 step.

Inputs (per sample, channel-wise):
  ch0: matlab_4x4_n          (normalised by norm_stats_4x4_from_8x8.npz)
  ch1: hfss_pred_4x4_n       (no-m4 cGAN prediction; same norm stats as ch0)
  ch2: residual_n            (matlab_4x4_n - hfss_pred_4x4_n)
  ch3: dphase_x / 180        (broadcast scalar)
  ch4: dphase_y / 180        (broadcast scalar)
Target:
  hfss_8x8_n                 (normalised by norm_stats_8x8.npz)

The matlab_8x8 channel is deliberately excluded (analogue of the 2-to-4
no-m4 ablation): we are testing whether the model can synthesise the 8x8
pattern from sub-array data and the analytical 4x4 prior alone.

Usage:
    python -m scripts.train_cgan_4to8_subarray
"""

import sys
import time
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, DEVICE,
    BATCH_SIZE, MAX_EPOCHS, RANDOM_SEED, NULL_THRESHOLD_DB,
)
from src.training.metrics import rmse, mae, pearson_correlation
from scripts.train_cgan import PatchDiscriminator
from scripts.train_cgan_2to4_fusion_no_m4 import (
    EnhancedResUNetGenerator, EMA, ReconLoss,
    train_one_epoch, evaluate_loader, AMP_DEVICE_TYPE,
    GEN_BASE, ATTN_HEADS, EMA_DECAY,
    LAMBDA_L1, LAMBDA_SSIM, NULL_LOSS_WEIGHT, NOISE_STD,
    LABEL_SMOOTH_REAL, LABEL_SMOOTH_FAKE, USE_AMP,
)

# ─── Paths ───────────────────────────────────────────────────────────────────
HDF5 = PROCESSED_DIR / "antenna_data_4to8_subarray.h5"
NORM_4X4 = PROCESSED_DIR / "norm_stats_4x4_from_8x8.npz"
NORM_8X8 = PROCESSED_DIR / "norm_stats_8x8.npz"
SPLITS = PROCESSED_DIR / "split_indices_4to8.npz"
CKPT_DIR = CHECKPOINTS_DIR / "cgan_resunet_patchgan_4to8_subarray"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hyperparameters (mirror the 2-to-4 no-m4 setup) ────────────────────────
EPOCHS = min(MAX_EPOCHS, 200)
BATCH = BATCH_SIZE
LR_G = 2e-4
LR_D = 2e-5
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 4
PLATEAU_MIN_LR = 1e-6
NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda"
VALIDATE_EVERY = 2
EARLY_STOP_PATIENCE = 40
SAVE_EVERY = 10
STEP_PRINT_EVERY = 10

# Reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# ─── Dataset ─────────────────────────────────────────────────────────────────
class Subarray4to8Dataset(Dataset):
    """Channel layout mirrors FusionDatasetNoM4 but at the 4x4 -> 8x8 scale.

    Returns (x, y, null_mask) where:
        x:         (5, H, W) float32
        y:         (1, H, W) float32 (normalised hfss_8x8)
        null_mask: (1, H, W) float32 (binary: target_dB < target_max_dB - 20)
    """

    def __init__(self, h5_path, indices,
                 mean_4x4, std_4x4, mean_8x8, std_8x8,
                 augment_noise=False, noise_std=NOISE_STD):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean_4x4 = mean_4x4.astype(np.float32)
        self.std_4x4 = std_4x4.astype(np.float32)
        self.mean_8x8 = mean_8x8.astype(np.float32)
        self.std_8x8 = std_8x8.astype(np.float32)
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

        m4 = f["matlab_4x4"][idx].astype(np.float32)
        hp4 = f["hfss_pred_4x4"][idx].astype(np.float32)
        h8 = f["hfss_8x8"][idx].astype(np.float32)
        meta = f["metadata"][idx].astype(np.float32)

        m4_n = (m4 - self.mean_4x4) / self.std_4x4
        hp4_n = (hp4 - self.mean_4x4) / self.std_4x4
        residual_n = m4_n - hp4_n
        h8_n = (h8 - self.mean_8x8) / self.std_8x8

        peak = float(h8.max())
        null_mask = (h8 < (peak + NULL_THRESHOLD_DB)).astype(np.float32)

        dphase_x = np.full_like(m4_n, fill_value=meta[0] / 180.0, dtype=np.float32)
        dphase_y = np.full_like(m4_n, fill_value=meta[1] / 180.0, dtype=np.float32)

        x = np.stack([m4_n, hp4_n, residual_n, dphase_x, dphase_y], axis=0)

        if self.augment_noise and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=(3,) + m4_n.shape).astype(np.float32)
            x[:3] += noise

        y = h8_n[None, ...]
        null_mask = null_mask[None, ...]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(null_mask)


def main():
    for p in (HDF5, NORM_4X4, NORM_8X8, SPLITS):
        if not p.exists():
            raise FileNotFoundError(p)

    s4 = np.load(NORM_4X4); s8 = np.load(NORM_8X8)
    mean_4x4 = s4["mean"].astype(np.float32)
    std_4x4 = np.maximum(s4["std"].astype(np.float32), 1e-6)
    mean_8x8 = s8["mean"].astype(np.float32)
    std_8x8 = np.maximum(s8["std"].astype(np.float32), 1e-6)

    sp = np.load(SPLITS)
    tr, va, te = sp["train"].astype(np.int64), sp["val"].astype(np.int64), sp["test"].astype(np.int64)
    print(f"Splits: {len(tr)} train / {len(va)} val / {len(te)} test", flush=True)

    tr_ds = Subarray4to8Dataset(HDF5, tr, mean_4x4, std_4x4, mean_8x8, std_8x8,
                                augment_noise=True, noise_std=NOISE_STD)
    va_ds = Subarray4to8Dataset(HDF5, va, mean_4x4, std_4x4, mean_8x8, std_8x8,
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

    # Resume support (warm-restart from last checkpoint if present)
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
        print("No D checkpoint found - starting D from random init.", flush=True)

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
    recon_loss = ReconLoss(mean_8x8, std_8x8).to(DEVICE)
    ema = EMA(G, decay=EMA_DECAY)
    if last_g_ema.exists():
        ema.shadow = torch.load(last_g_ema, map_location=DEVICE)
        print(f"Resumed EMA shadow from {last_g_ema.name}", flush=True)

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
                torch.save(ema.state_dict(), CKPT_DIR / "best_generator.pt")
                print(f"  * new best (val rmse {best_rmse:.5f}) saved to best_generator.pt",
                      flush=True)
            else:
                epochs_without_improve += VALIDATE_EVERY
                print(f"  (no improvement for {epochs_without_improve} epochs)", flush=True)

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
