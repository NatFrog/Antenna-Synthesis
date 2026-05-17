"""
Train the 2-to-4 fusion cGAN.

Generator : ResUNet (5-channel in, 1-channel out)
Discriminator : PatchGAN (6-channel: 5 condition + 1 pattern), reused from train_cgan.py

Inputs (per sample, channel-wise):
  ch0: matlab_2x2           (normalised with norm_stats_2x2.npz)
  ch1: hfss_pred_2x2        (normalised with norm_stats_2x2.npz)
  ch2: matlab_4x4           (normalised with norm_stats.npz)
  ch3: dphase_x / 180       (scalar broadcast)
  ch4: dphase_y / 180       (scalar broadcast)
Target:
  hfss_4x4                  (normalised with norm_stats.npz)

Usage:
    python -m scripts.train_cgan_2to4_fusion
"""

import sys
import time
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
    BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, RANDOM_SEED,
)
from src.training.metrics import rmse, mae, pearson_correlation
from scripts.train_cgan import PatchDiscriminator  # reuse unchanged

# ─── Paths ───────────────────────────────────────────────────────────────────
HDF5 = PROCESSED_DIR / "antenna_data_2to4.h5"
NORM_2X2 = PROCESSED_DIR / "norm_stats_2x2.npz"
NORM_4X4 = PROCESSED_DIR / "norm_stats.npz"
SPLITS = PROCESSED_DIR / "split_indices_2to4.npz"
CKPT_DIR = CHECKPOINTS_DIR / "cgan_resunet_patchgan_2to4"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hyperparameters (match train_cgan_2x2) ─────────────────────────────────
EPOCHS = min(MAX_EPOCHS, 120)
BATCH = BATCH_SIZE
LR_G = LEARNING_RATE
LR_D = LEARNING_RATE * 0.1
LAMBDA_L1 = 100.0
LABEL_SMOOTH_REAL = 0.9
LABEL_SMOOTH_FAKE = 0.1
NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda"
USE_AMP = False           # match the 2x2 training (safer; mixed-precision had issues historically)
VALIDATE_EVERY = 2
EARLY_STOP_PATIENCE = 20
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
class FusionDataset(Dataset):
    """HDF5-backed dataset bundling the 4 fused sources with per-channel norm."""

    def __init__(self, h5_path, indices,
                 mean_2x2, std_2x2, mean_4x4, std_4x4):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean_2x2 = mean_2x2.astype(np.float32)
        self.std_2x2 = std_2x2.astype(np.float32)
        self.mean_4x4 = mean_4x4.astype(np.float32)
        self.std_4x4 = std_4x4.astype(np.float32)
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
        m4 = f["matlab_4x4"][idx].astype(np.float32)
        h4 = f["hfss_4x4"][idx].astype(np.float32)
        meta = f["metadata"][idx].astype(np.float32)

        m2_n = (m2 - self.mean_2x2) / self.std_2x2
        hp2_n = (hp2 - self.mean_2x2) / self.std_2x2
        m4_n = (m4 - self.mean_4x4) / self.std_4x4
        h4_n = (h4 - self.mean_4x4) / self.std_4x4

        dphase_x = np.full_like(m4_n, fill_value=meta[0] / 180.0, dtype=np.float32)
        dphase_y = np.full_like(m4_n, fill_value=meta[1] / 180.0, dtype=np.float32)

        x = np.stack([m2_n, hp2_n, m4_n, dphase_x, dphase_y], axis=0)  # (5, 181, 360)
        y = h4_n[None, ...]  # (1, 181, 360)
        return torch.from_numpy(x), torch.from_numpy(y)


# ─── ResUNet (lifted from 05_ResUNet.ipynb) ─────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        s = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + s)


class DownRes(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f = self.res(x); p = self.pool(f)
        return f, p


class UpRes(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.res = ResidualBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:],
                                          mode="bilinear", align_corners=False)
        return self.res(torch.cat([x, skip], dim=1))


class ResUNetGenerator(nn.Module):
    def __init__(self, in_ch=5, out_ch=1, base=32):
        super().__init__()
        self.d1 = DownRes(in_ch, base)
        self.d2 = DownRes(base, base * 2)
        self.d3 = DownRes(base * 2, base * 4)
        self.d4 = DownRes(base * 4, base * 8)
        self.bottleneck = ResidualBlock(base * 8, base * 16)
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
        x = self.u1(b, s4)
        x = self.u2(x, s3)
        x = self.u3(x, s2)
        x = self.u4(x, s1)
        out = self.out(x)
        if out.shape[-2:] != x_in.shape[-2:]:
            out = nn.functional.interpolate(out, size=x_in.shape[-2:],
                                            mode="bilinear", align_corners=False)
        return out


# ─── Training loop ───────────────────────────────────────────────────────────
AMP_DEVICE_TYPE = "cuda" if DEVICE.type == "cuda" else "cpu"


def train_one_epoch(generator, discriminator, loader, opt_g, opt_d, scaler_g, scaler_d, bce, l1):
    generator.train(); discriminator.train()
    g_losses, d_losses = [], []
    total_steps = len(loader)
    for step, (x, y) in enumerate(loader, 1):
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)

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
            rec = l1(y_fake, y)
            g_loss = adv + LAMBDA_L1 * rec
        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(g_loss).backward()
        scaler_g.step(opt_g); scaler_g.update()

        g_losses.append(float(g_loss)); d_losses.append(float(d_loss))
        if step == 1 or step % STEP_PRINT_EVERY == 0 or step == total_steps:
            print(f"  step {step:03d}/{total_steps} | g={g_loss.item():.4f} d={d_loss.item():.4f}",
                  flush=True)
    return np.mean(g_losses), np.mean(d_losses)


@torch.no_grad()
def evaluate_loader(generator, loader):
    generator.eval()
    yt, yp = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)
        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
            y_hat = generator(x)
        yt.append(y.cpu().numpy()); yp.append(y_hat.cpu().numpy())
    yt = np.concatenate(yt).reshape(-1); yp = np.concatenate(yp).reshape(-1)
    return {"rmse": float(rmse(yt, yp)),
            "mae":  float(mae(yt, yp)),
            "pearson": float(pearson_correlation(yt, yp))}


def main():
    # Sanity checks
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

    tr_ds = FusionDataset(HDF5, tr, mean_2x2, std_2x2, mean_4x4, std_4x4)
    va_ds = FusionDataset(HDF5, va, mean_2x2, std_2x2, mean_4x4, std_4x4)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    G = ResUNetGenerator(in_ch=5, out_ch=1, base=32).to(DEVICE)
    D = PatchDiscriminator(in_channels=6, base=64).to(DEVICE)
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"G params: {g_params/1e6:.2f}M   D params: {d_params/1e6:.2f}M", flush=True)

    opt_g = optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    scaler_g = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
    scaler_d = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    best_rmse = float("inf"); best_epoch = -1; epochs_without_improve = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===", flush=True)
        g_avg, d_avg = train_one_epoch(G, D, tr_loader, opt_g, opt_d, scaler_g, scaler_d, bce, l1)
        print(f"  train: g_loss={g_avg:.4f}  d_loss={d_avg:.4f}", flush=True)

        if epoch % VALIDATE_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            m = evaluate_loader(G, va_loader)
            print(f"  val  : rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  r={m['pearson']:.4f}",
                  flush=True)
            if m["rmse"] < best_rmse - 1e-5:
                best_rmse = m["rmse"]; best_epoch = epoch; epochs_without_improve = 0
                torch.save(G.state_dict(), CKPT_DIR / "best_generator.pt")
                print(f"  * new best (val rmse {best_rmse:.5f}) saved to best_generator.pt",
                      flush=True)
            else:
                epochs_without_improve += VALIDATE_EVERY
                print(f"  (no improvement for {epochs_without_improve} epochs)", flush=True)

        torch.save(G.state_dict(), CKPT_DIR / "last_generator.pt")
        if epoch % SAVE_EVERY == 0:
            torch.save(G.state_dict(), CKPT_DIR / f"generator_epoch_{epoch:03d}.pt")

        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}; best epoch was {best_epoch} "
                  f"(val rmse {best_rmse:.5f})", flush=True)
            break

    total = (time.time() - t0) / 60
    print(f"\nTraining done in {total:.1f} min. Best epoch {best_epoch} "
          f"val rmse {best_rmse:.5f}.  Checkpoints in {CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
