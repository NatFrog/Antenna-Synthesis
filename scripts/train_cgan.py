"""
Train cGAN (U-Net Generator + PatchGAN Discriminator) for MATLAB->HFSS pattern correction.

Standalone script extracted from 04_unet_patchgen.ipynb.

Usage:
    python -m scripts.train_cgan
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
    HDF5_PATH,
    NORM_STATS_PATH,
    PROCESSED_DIR,
    CHECKPOINTS_DIR,
    N_THETA,
    N_PHI,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    RANDOM_SEED,
    DEVICE,
)
from src.training.metrics import rmse, mae, pearson_correlation

# ─── Config ──────────────────────────────────────────────────────────────────
SPLIT_INDICES_PATH = PROCESSED_DIR / "split_indices.npz"
CGAN_CKPT_DIR = CHECKPOINTS_DIR / "cgan_unet_patchgan"
CGAN_CKPT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = min(MAX_EPOCHS, 120)
BATCH_SIZE_CGAN = BATCH_SIZE
LR_G = LEARNING_RATE
LR_D = LEARNING_RATE * 0.1  # 10x slower than generator to prevent discriminator collapse
LAMBDA_L1 = 100.0
LABEL_SMOOTH_REAL = 0.9    # soft labels to keep discriminator from becoming overconfident
LABEL_SMOOTH_FAKE = 0.1

NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda"
USE_AMP = False
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
class AntennaPatternDataset(Dataset):
    """HDF5-backed dataset using precomputed split indices and normalization stats."""

    def __init__(self, h5_path, indices, mean_map, std_map):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean_map.astype(np.float32)
        self.std = std_map.astype(np.float32)
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

        matlab = f["matlab_patterns"][idx].astype(np.float32)
        hfss = f["hfss_patterns"][idx].astype(np.float32)
        meta = f["metadata"][idx].astype(np.float32)

        x_norm = (matlab - self.mean) / self.std
        y_norm = (hfss - self.mean) / self.std

        dphase_x = np.full_like(x_norm, fill_value=meta[0] / 180.0, dtype=np.float32)
        dphase_y = np.full_like(x_norm, fill_value=meta[1] / 180.0, dtype=np.float32)

        x = np.stack([x_norm, dphase_x, dphase_y], axis=0)  # (3, H, W)
        y = y_norm[None, ...]  # (1, H, W)
        return torch.from_numpy(x), torch.from_numpy(y)


# ─── Generator ───────────────────────────────────────────────────────────────
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=64):
        super().__init__()
        self.d1 = DownBlock(in_channels, base, use_bn=False)
        self.d2 = DownBlock(base, base * 2)
        self.d3 = DownBlock(base * 2, base * 4)
        self.d4 = DownBlock(base * 4, base * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.u1 = UpBlock(base * 8, base * 8, use_dropout=True)
        self.u2 = UpBlock(base * 16, base * 4)
        self.u3 = UpBlock(base * 8, base * 2)
        self.u4 = UpBlock(base * 4, base)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_channels, kernel_size=4, stride=2, padding=1),
        )

    @staticmethod
    def _match_hw(src, ref):
        if src.shape[-2:] != ref.shape[-2:]:
            src = nn.functional.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return src

    def forward(self, x):
        x_in = x
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        b = self.bottleneck(d4)

        u1 = self.u1(b)
        u1 = self._match_hw(u1, d4)
        u1 = torch.cat([u1, d4], dim=1)

        u2 = self.u2(u1)
        u2 = self._match_hw(u2, d3)
        u2 = torch.cat([u2, d3], dim=1)

        u3 = self.u3(u2)
        u3 = self._match_hw(u3, d2)
        u3 = torch.cat([u3, d2], dim=1)

        u4 = self.u4(u3)
        u4 = self._match_hw(u4, d1)
        u4 = torch.cat([u4, d1], dim=1)

        out = self.final(u4)
        out = self._match_hw(out, x_in[:, :1])
        return out


# ─── Discriminator ───────────────────────────────────────────────────────────
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base=64):
        super().__init__()

        def block(cin, cout, stride=2, use_bn=True):
            layers = [nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, base, use_bn=False),
            block(base, base * 2),
            block(base * 2, base * 4),
            block(base * 4, base * 8, stride=1),
            nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.model(inp)


# ─── Training functions ──────────────────────────────────────────────────────
AMP_DEVICE_TYPE = "cuda" if DEVICE.type == "cuda" else "cpu"


def train_one_epoch(generator, discriminator, train_loader, opt_g, opt_d,
                    scaler_g, scaler_d, bce, l1_loss):
    generator.train()
    discriminator.train()

    g_losses, d_losses = [], []
    total_steps = len(train_loader)

    for step, (x, y) in enumerate(train_loader, start=1):
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)

        # Train D
        with torch.no_grad():
            with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
                y_fake = generator(x)

        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
            pred_real = discriminator(x, y)
            pred_fake = discriminator(x, y_fake)
            real_labels = torch.full_like(pred_real, LABEL_SMOOTH_REAL)
            fake_labels = torch.full_like(pred_fake, LABEL_SMOOTH_FAKE)
            d_loss = 0.5 * (bce(pred_real, real_labels) + bce(pred_fake, fake_labels))

        opt_d.zero_grad(set_to_none=True)
        scaler_d.scale(d_loss).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        # Train G
        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
            y_fake = generator(x)
            pred_fake_for_g = discriminator(x, y_fake)
            adv_loss = bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            rec_loss = l1_loss(y_fake, y)
            g_loss = adv_loss + LAMBDA_L1 * rec_loss

        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(g_loss).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        g_losses.append(float(g_loss.item()))
        d_losses.append(float(d_loss.item()))

        if step == 1 or (step % STEP_PRINT_EVERY == 0) or (step == total_steps):
            print(f"  step {step:03d}/{total_steps} | g_loss={g_loss.item():.4f} d_loss={d_loss.item():.4f}")

    return np.mean(g_losses), np.mean(d_losses)


@torch.no_grad()
def evaluate_loader(generator, loader):
    generator.eval()
    y_true_all, y_pred_all = [], []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)
        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=USE_AMP):
            y_hat = generator(x)
        y_true_all.append(y.cpu().numpy())
        y_pred_all.append(y_hat.cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    metrics = {
        "rmse_norm": float(rmse(yp, yt)),
        "mae_norm": float(mae(yp, yt)),
        "pearson": float(pearson_correlation(yp, yt)),
    }
    return metrics, y_pred, y_true


def denormalize_map(arr, mean_map, std_map):
    return arr * std_map + mean_map


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"HDF5: {HDF5_PATH}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE_CGAN}, LR: {LR_G}")

    # Load splits and normalization
    splits = np.load(SPLIT_INDICES_PATH)
    train_idx = np.sort(splits["train"].astype(np.int64))
    val_idx = np.sort(splits["val"].astype(np.int64))
    test_idx = np.sort(splits["test"].astype(np.int64))

    stats = np.load(NORM_STATS_PATH)
    mean_map = stats["mean"].astype(np.float32)
    std_map = np.maximum(stats["std"].astype(np.float32), 1e-6)

    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    # Datasets and loaders
    train_ds = AntennaPatternDataset(HDF5_PATH, train_idx, mean_map, std_map)
    val_ds = AntennaPatternDataset(HDF5_PATH, val_idx, mean_map, std_map)
    test_ds = AntennaPatternDataset(HDF5_PATH, test_idx, mean_map, std_map)

    loader_kwargs = {"batch_size": BATCH_SIZE_CGAN, "num_workers": NUM_WORKERS, "pin_memory": PIN_MEMORY}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Models
    generator = UNetGenerator(in_channels=3, out_channels=1).to(DEVICE)
    discriminator = PatchDiscriminator(in_channels=4).to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    opt_g = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))

    scaler_g = torch.amp.GradScaler(device=AMP_DEVICE_TYPE, enabled=USE_AMP)
    scaler_d = torch.amp.GradScaler(device=AMP_DEVICE_TYPE, enabled=USE_AMP)

    # Training
    best_val_rmse = float("inf")
    no_improve_epochs = 0
    best_ckpt_path = CGAN_CKPT_DIR / "best_generator.pt"
    last_ckpt_path = CGAN_CKPT_DIR / "last_generator.pt"
    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        g_loss, d_loss = train_one_epoch(
            generator, discriminator, train_loader, opt_g, opt_d,
            scaler_g, scaler_d, bce, l1_loss,
        )

        run_val = (epoch == 1) or (epoch % VALIDATE_EVERY == 0) or (epoch == EPOCHS)
        if run_val:
            val_metrics, _, _ = evaluate_loader(generator, val_loader)

            if val_metrics["rmse_norm"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse_norm"]
                no_improve_epochs = 0
                torch.save(generator.state_dict(), best_ckpt_path)
            else:
                no_improve_epochs += VALIDATE_EVERY

            epoch_sec = time.perf_counter() - epoch_start
            elapsed_sec = time.perf_counter() - training_start
            eta_sec = (elapsed_sec / epoch) * (EPOCHS - epoch)
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"G: {g_loss:.4f} D: {d_loss:.4f} | "
                f"val_rmse: {val_metrics['rmse_norm']:.4f} "
                f"val_mae: {val_metrics['mae_norm']:.4f} "
                f"val_r: {val_metrics['pearson']:.4f} | "
                f"epoch: {epoch_sec:.1f}s eta: {eta_sec/60:.1f}m"
            )
        else:
            epoch_sec = time.perf_counter() - epoch_start
            elapsed_sec = time.perf_counter() - training_start
            eta_sec = (elapsed_sec / epoch) * (EPOCHS - epoch)
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | G: {g_loss:.4f} D: {d_loss:.4f} | val: skipped | "
                f"epoch: {epoch_sec:.1f}s eta: {eta_sec/60:.1f}m"
            )

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            torch.save(generator.state_dict(), last_ckpt_path)

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val RMSE improvement for {no_improve_epochs} epochs).")
            break

    print("Training complete.")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Best val RMSE (normalized): {best_val_rmse:.6f}")

    # ── Test evaluation ──
    generator.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    test_metrics, y_pred_norm, y_true_norm = evaluate_loader(generator, test_loader)
    print(f"\nTest metrics (normalized): {test_metrics}")

    y_pred_db = denormalize_map(y_pred_norm[:, 0], mean_map[None, ...], std_map[None, ...])
    y_true_db = denormalize_map(y_true_norm[:, 0], mean_map[None, ...], std_map[None, ...])

    delta = y_pred_db - y_true_db
    rmse_db = float(np.sqrt(np.mean(delta ** 2)))
    mae_db = float(np.mean(np.abs(delta)))
    print(f"Test RMSE (dB): {rmse_db:.4f}")
    print(f"Test MAE  (dB): {mae_db:.4f}")


if __name__ == "__main__":
    main()
