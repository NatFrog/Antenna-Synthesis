"""
Train cGAN (U-Net Generator + PatchGAN Discriminator) on the 2x2 antenna array dataset.

Uses the same model architecture and training loop as train_cgan.py, but points
to 2x2-specific data / norm stats / splits and saves checkpoints/results to
separate folders.

Usage:
    python -m scripts.train_cgan_2x2
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
from torch.utils.data import DataLoader

from src.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR,
    N_THETA, N_PHI,
    BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS,
    RANDOM_SEED, DEVICE,
)
from src.training.metrics import rmse, mae, pearson_correlation

# Reuse model classes and dataset from the 4x4 training script
from scripts.train_cgan import (
    UNetGenerator, PatchDiscriminator, AntennaPatternDataset,
)

# ── 2x2-specific paths ──
HDF5_PATH_2X2 = PROCESSED_DIR / "antenna_data_2x2.h5"
NORM_STATS_PATH_2X2 = PROCESSED_DIR / "norm_stats_2x2.npz"
SPLIT_INDICES_PATH_2X2 = PROCESSED_DIR / "split_indices_2x2.npz"
CGAN_CKPT_DIR_2X2 = CHECKPOINTS_DIR / "cgan_unet_patchgan_2x2"
CGAN_CKPT_DIR_2X2.mkdir(parents=True, exist_ok=True)

# ── Config (same as 4x4 to enable direct comparison) ──
EPOCHS = min(MAX_EPOCHS, 120)
BATCH_SIZE_CGAN = BATCH_SIZE
LR_G = LEARNING_RATE
LR_D = LEARNING_RATE * 0.1
LAMBDA_L1 = 100.0
LABEL_SMOOTH_REAL = 0.9
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

        if step == 1 or step % STEP_PRINT_EVERY == 0 or step == total_steps:
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
    return {
        "rmse_norm": float(rmse(yp, yt)),
        "mae_norm": float(mae(yp, yt)),
        "pearson": float(pearson_correlation(yp, yt)),
    }, y_pred, y_true


def denormalize_map(arr, mean_map, std_map):
    return arr * std_map + mean_map


def main():
    print(f"Device: {DEVICE}")
    print(f"HDF5: {HDF5_PATH_2X2}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE_CGAN}, LR_G: {LR_G}, LR_D: {LR_D}")

    splits = np.load(SPLIT_INDICES_PATH_2X2)
    train_idx = np.sort(splits["train"].astype(np.int64))
    val_idx = np.sort(splits["val"].astype(np.int64))
    test_idx = np.sort(splits["test"].astype(np.int64))

    stats = np.load(NORM_STATS_PATH_2X2)
    mean_map = stats["mean"].astype(np.float32)
    std_map = np.maximum(stats["std"].astype(np.float32), 1e-6)

    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    train_ds = AntennaPatternDataset(HDF5_PATH_2X2, train_idx, mean_map, std_map)
    val_ds = AntennaPatternDataset(HDF5_PATH_2X2, val_idx, mean_map, std_map)
    test_ds = AntennaPatternDataset(HDF5_PATH_2X2, test_idx, mean_map, std_map)

    loader_kwargs = {"batch_size": BATCH_SIZE_CGAN, "num_workers": NUM_WORKERS, "pin_memory": PIN_MEMORY}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    generator = UNetGenerator(in_channels=3, out_channels=1).to(DEVICE)
    discriminator = PatchDiscriminator(in_channels=4).to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    opt_g = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))
    scaler_g = torch.amp.GradScaler(device=AMP_DEVICE_TYPE, enabled=USE_AMP)
    scaler_d = torch.amp.GradScaler(device=AMP_DEVICE_TYPE, enabled=USE_AMP)

    best_val_rmse = float("inf")
    no_improve_epochs = 0
    best_ckpt_path = CGAN_CKPT_DIR_2X2 / "best_generator.pt"
    last_ckpt_path = CGAN_CKPT_DIR_2X2 / "last_generator.pt"
    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        g_loss, d_loss = train_one_epoch(
            generator, discriminator, train_loader, opt_g, opt_d,
            scaler_g, scaler_d, bce, l1_loss,
        )

        run_val = epoch == 1 or epoch % VALIDATE_EVERY == 0 or epoch == EPOCHS
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
