"""
Multi-scale residual cGAN with periodic hfss_16x16 monitoring.

This is the 5-channel "with anchor" extension of the multi-scale residual model.
A new ch4 carries the cascade-predicted HFSS pattern at one scale step below the
target (a real-coupling reference). The bet is that the model learns to use this
anchor to inform the residual at the target scale, in a way that transfers
across scales — so we can keep extrapolating to scales without HFSS supervision
in the loss.

Setup
-----
- Gradients see ONLY (matlab_NxN, hfss_NxN, hfss_pred_(N/2)x(N/2), beta) for
  N in {4, 8}.
- Output is the per-pixel residual delta = hfss - matlab in dB-space.
- Predicted pattern at any scale is composed analytically:
      pred_NxN_dB = matlab_NxN_dB + delta_dB
  then per-sample max-normalised to 0 dB to match the project convention.
- Every VALIDATE_EVERY epochs, we evaluate the model (with EMA weights, no
  gradients) on 100 hfss_16x16 samples (val_16x16) and save best_generator.pt
  whenever val_16x16 RMSE improves.
- Final honest evaluation runs on the other 100 hfss_16x16 samples (test_16x16)
  in scripts/evaluate_residual_multiscale.py — those samples are never touched
  during training.

Channels (5 in, 1 out):
  ch0 : matlab_NxN_n             (z-scored with norm_stats_matlab_combined.npz;
                                  same stats applied to N=4, N=8 and N=16)
  ch1 : dphase_x / 180
  ch2 : dphase_y / 180
  ch3 : scale_token = N/16       (0.25 for N=4, 0.50 for N=8, 1.00 for N=16)
  ch4 : hfss_pred_(N/2)x(N/2)_n  (anchor: cascade prediction at one scale step
                                  below the target, z-scored with the same
                                  combined matlab stats as ch0)
  out : delta_n                  (residual normalised by sigma_residual)

Anchor sources (ch4)
--------------------
  N=4 (training):  hfss_pred_2x2 from antenna_data_2to4.h5
  N=8 (training):  hfss_pred_4x4 from antenna_data_4to8.h5
  N=16 (inference): hfss_pred_8x8 from antenna_data_8to16_subarray.h5

Splits
------
- N=4 train: 2to4 train (3997)        -> gradients
  N=4 val:   2to4 val   (498)         -> in-distribution monitoring
  N=4 test:  2to4 test  (505)         -> not used in this experiment
- N=8 train: 4to8 train (1600)        -> gradients
  N=8 val:   4to8 val   (200)         -> in-distribution monitoring
  N=8 test:  4to8 test  (200)         -> reserved for the 16x16 split (below)
- val_16x16: first 100 of sorted 4to8 test indices  -> selection only (no grads)
- test_16x16: last  100 of sorted 4to8 test indices -> never touched during training

Usage:
    python -m scripts.train_residual_multiscale
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from src.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, DEVICE,
    BATCH_SIZE, MAX_EPOCHS, RANDOM_SEED, NULL_THRESHOLD_DB,
)
from src.training.metrics import rmse, mae, pearson_correlation
from scripts.train_cgan import PatchDiscriminator
from scripts.train_cgan_2to4_fusion_no_m4 import (
    EnhancedResUNetGenerator, EMA, ssim_dB,
    GEN_BASE, ATTN_HEADS, EMA_DECAY,
    LAMBDA_L1, LAMBDA_SSIM, NULL_LOSS_WEIGHT,
    LABEL_SMOOTH_REAL, LABEL_SMOOTH_FAKE,
    AMP_DEVICE_TYPE,
)

# Override the imported USE_AMP (the rest of the project disables AMP after
# historical numerical-stability issues with the 2x2 training). For *this*
# run we want the speedup, but fp16 was the original culprit — fp16 softmax
# in the bottleneck self-attention saturates and produces NaN on step 1.
#
# bfloat16 has the same dynamic range as fp32 (just lower mantissa precision),
# so attention/SSIM are stable, and the RTX 4070 (Ada, compute 8.9) supports
# bf16 natively. bf16 also does NOT need GradScaler — the dynamic range is
# already fp32-equivalent — so we gate the scaler accordingly below.
USE_AMP = True
AMP_DTYPE = torch.bfloat16  # set to torch.float16 only if you can guarantee no NaN
USE_GRAD_SCALER = USE_AMP and AMP_DTYPE == torch.float16

# ── Paths ───────────────────────────────────────────────────────────────────
H5_2TO4 = PROCESSED_DIR / "antenna_data_2to4.h5"
H5_4TO8 = PROCESSED_DIR / "antenna_data_4to8.h5"
H5_16X16 = PROCESSED_DIR / "antenna_data_8to16_subarray.h5"
NORM_COMBINED = PROCESSED_DIR / "norm_stats_matlab_combined.npz"
M16_TEST = PROCESSED_DIR / "matlab_16x16_test.npz"
SPLITS_2TO4 = PROCESSED_DIR / "split_indices_2to4.npz"
SPLITS_4TO8 = PROCESSED_DIR / "split_indices_4to8.npz"
CKPT_DIR = CHECKPOINTS_DIR / "residual_multiscale_with_anchor"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ─────────────────────────────────────────────────────────
EPOCHS = min(MAX_EPOCHS, 200)
BATCH = BATCH_SIZE                    # 16
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
STEP_PRINT_EVERY = 50
NOISE_STD = 0.05                      # input augmentation in normalised space
ANCHOR_DROPOUT_P = 0.25               # spec mitigation for ch4-overreliance:
                                       # zero out ch4 in this fraction of training
                                       # samples so the model can't just copy the
                                       # anchor at training scales. Spec range 20-30%.

# Reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# ── Datasets ────────────────────────────────────────────────────────────────
class ScaleDatasetWithAnchor(Dataset):
    """Yields (x, target_dB, matlab_dB, null_mask) for one fixed scale.

    x         : (5, H, W) float32 inputs (matlab_n, dpx, dpy, scale_tok, anchor_n)
    target_dB : (1, H, W) float32   hfss_NxN max-normalised to 0
    matlab_dB : (1, H, W) float32   matlab_NxN max-normalised to 0
    null_mask : (1, H, W) float32   1.0 where target_dB < target_max + NULL_THRESHOLD_DB

    The anchor (ch4) is the cascade-predicted HFSS pattern at one scale step
    below the target — e.g. hfss_pred_2x2 when training at N=4. It carries a
    real-coupling reference that the analytical matlab pattern (ch0) lacks.
    """

    def __init__(self, h5_path, indices, scale_N, matlab_key, hfss_key,
                 anchor_key, mean_mat, std_mat,
                 augment_noise=False, noise_std=NOISE_STD,
                 anchor_dropout_p=0.0):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scale_N = int(scale_N)
        self.matlab_key = matlab_key
        self.hfss_key = hfss_key
        self.anchor_key = anchor_key
        self.mean_mat = mean_mat.astype(np.float32)
        self.std_mat = np.maximum(std_mat.astype(np.float32), 1e-6)
        self.augment_noise = augment_noise
        self.noise_std = noise_std
        # Probability of zeroing ch4 (anchor) at sample time. Only used during
        # training; eval datasets keep the anchor intact.
        self.anchor_dropout_p = float(anchor_dropout_p)
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
        mat = f[self.matlab_key][idx].astype(np.float32)   # max-norm to 0 in source
        hfss = f[self.hfss_key][idx].astype(np.float32)
        anchor = f[self.anchor_key][idx].astype(np.float32)
        meta = f["metadata"][idx].astype(np.float32)

        mat_n = (mat - self.mean_mat) / self.std_mat
        anchor_n = (anchor - self.mean_mat) / self.std_mat
        dphase_x = np.full_like(mat_n, fill_value=meta[0] / 180.0, dtype=np.float32)
        dphase_y = np.full_like(mat_n, fill_value=meta[1] / 180.0, dtype=np.float32)
        scale_tok = np.full_like(mat_n, fill_value=self.scale_N / 16.0, dtype=np.float32)

        x = np.stack([mat_n, dphase_x, dphase_y, scale_tok, anchor_n], axis=0)

        if self.augment_noise and self.noise_std > 0:
            # Both ch0 (matlab) and ch4 (anchor) are pattern-shaped; the
            # constant scalar planes (dpx/dpy/scale_tok) get nothing.
            x[0] += np.random.normal(0, self.noise_std, size=mat_n.shape).astype(np.float32)
            x[4] += np.random.normal(0, self.noise_std, size=mat_n.shape).astype(np.float32)

        # Anchor-dropout: with probability anchor_dropout_p replace ch4 with
        # zeros (i.e. the *normalised* mean of the anchor distribution). This
        # forces the generator to remain useful when the anchor is missing or
        # uninformative — the failure mode at N=16 — and prevents the model
        # from reducing to a copy-the-anchor shortcut at training scales.
        if self.anchor_dropout_p > 0.0 and np.random.rand() < self.anchor_dropout_p:
            x[4] = 0.0

        target_max = float(hfss.max())
        null_mask = (hfss < (target_max + NULL_THRESHOLD_DB)).astype(np.float32)

        return (torch.from_numpy(x),
                torch.from_numpy(hfss[None]),
                torch.from_numpy(mat[None]),
                torch.from_numpy(null_mask[None]))


class Val16x16Dataset(Dataset):
    """100 hfss_16x16 samples — used for selection only, never for gradients.

    Loads precomputed analytical matlab_16x16 (from matlab_16x16_test.npz) and
    pairs with the corresponding hfss_16x16 + cascade-predicted hfss_pred_8x8
    (the ch4 anchor at N=16) from antenna_data_8to16_subarray.h5.
    """

    def __init__(self, indices, mat16_arr, mean_mat, std_mat):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mat16 = mat16_arr.astype(np.float32)        # (N, 181, 360)
        self.mean_mat = mean_mat.astype(np.float32)
        self.std_mat = np.maximum(std_mat.astype(np.float32), 1e-6)
        self._file = None
        # Pre-load all hfss_16x16, hfss_pred_8x8, metadata
        # (100 * 181 * 360 * 4 ≈ 26 MB per pattern array)
        with h5py.File(H5_16X16, "r") as f:
            self.hfss16 = f["hfss_16x16"][self.indices].astype(np.float32)
            self.hfss_pred_8x8 = f["hfss_pred_8x8"][self.indices].astype(np.float32)
            self.meta = f["metadata"][self.indices].astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        mat = self.mat16[i]
        hfss = self.hfss16[i]
        anchor = self.hfss_pred_8x8[i]
        meta = self.meta[i]

        mat_n = (mat - self.mean_mat) / self.std_mat
        anchor_n = (anchor - self.mean_mat) / self.std_mat
        dphase_x = np.full_like(mat_n, fill_value=meta[0] / 180.0, dtype=np.float32)
        dphase_y = np.full_like(mat_n, fill_value=meta[1] / 180.0, dtype=np.float32)
        scale_tok = np.full_like(mat_n, fill_value=16.0 / 16.0, dtype=np.float32)

        x = np.stack([mat_n, dphase_x, dphase_y, scale_tok, anchor_n], axis=0)
        return (torch.from_numpy(x),
                torch.from_numpy(hfss[None]),
                torch.from_numpy(mat[None]))


# ── Loss: composes pred = matlab_dB + delta_dB, max-norms, compares ─────────
class ResidualReconLoss(nn.Module):
    """Reconstruction loss for residual-output models.

    Model output is delta_n (normalised residual). Inside the loss:
        delta_dB = delta_n * sigma_res
        pred_dB  = matlab_dB + delta_dB
        pred_dB  = pred_dB - pred_dB.amax(per-sample)   # max-norm to 0
        L1   = ||pred_dB - target_dB||_1, weighted (3x at nulls)
        SSIM = ssim_dB(pred_dB, target_dB)
        total = LAMBDA_L1 * L1 + LAMBDA_SSIM * (1 - SSIM)
    """

    def __init__(self, sigma_res, lambda_l1=LAMBDA_L1, lambda_ssim=LAMBDA_SSIM,
                 null_weight=NULL_LOSS_WEIGHT):
        super().__init__()
        self.sigma_res = float(sigma_res)
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.null_weight = null_weight

    def compose(self, delta_n, matlab_dB):
        """Return per-sample max-normed predicted pattern in dB."""
        delta_dB = delta_n * self.sigma_res
        pred_dB = matlab_dB + delta_dB
        # Per-sample max-norm to 0 (subtract max over (H, W) per (batch, channel))
        pred_max = pred_dB.amax(dim=(-1, -2), keepdim=True)
        return pred_dB - pred_max

    def forward(self, delta_n, matlab_dB, target_dB, null_mask):
        pred_dB = self.compose(delta_n, matlab_dB)
        err = torch.abs(pred_dB - target_dB)
        weight = 1.0 + (self.null_weight - 1.0) * null_mask
        l1 = (err * weight).mean()
        ssim_val = ssim_dB(pred_dB, target_dB)
        total = self.lambda_l1 * l1 + self.lambda_ssim * (1.0 - ssim_val)
        return total, l1.detach(), ssim_val.detach(), pred_dB.detach()


# ── Train / val loops ───────────────────────────────────────────────────────
def train_one_epoch(G, D, ema, loader, opt_g, opt_d, scaler_g, scaler_d,
                    bce, recon_loss):
    G.train(); D.train()
    g_losses = []; d_losses = []; l1s = []; ssims = []
    total_steps = len(loader)
    for step, (x, target_dB, matlab_dB, null_mask) in enumerate(loader, 1):
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        target_dB = target_dB.to(DEVICE, non_blocking=PIN_MEMORY)
        matlab_dB = matlab_dB.to(DEVICE, non_blocking=PIN_MEMORY)
        null_mask = null_mask.to(DEVICE, non_blocking=PIN_MEMORY)

        # ── D step ──
        with torch.no_grad():
            with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP):
                delta_n = G(x)
                pred_dB_for_d = recon_loss.compose(delta_n, matlab_dB)
        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP):
            pr = D(x, target_dB / 30.0)            # rough scaling for D inputs
            pf = D(x, pred_dB_for_d / 30.0)
            rl = torch.full_like(pr, LABEL_SMOOTH_REAL)
            fl = torch.full_like(pf, LABEL_SMOOTH_FAKE)
            d_loss = 0.5 * (bce(pr, rl) + bce(pf, fl))
        opt_d.zero_grad(set_to_none=True)
        scaler_d.scale(d_loss).backward()
        scaler_d.step(opt_d); scaler_d.update()

        # ── G step ──
        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP):
            delta_n = G(x)
            recon, l1_val, ssim_val, pred_dB = recon_loss(delta_n, matlab_dB, target_dB, null_mask)
            pf_g = D(x, pred_dB / 30.0)
            adv = bce(pf_g, torch.ones_like(pf_g))
            g_loss = adv + recon
        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(g_loss).backward()
        scaler_g.step(opt_g); scaler_g.update()

        ema.update(G)

        g_losses.append(float(g_loss)); d_losses.append(float(d_loss))
        l1s.append(float(l1_val)); ssims.append(float(ssim_val))
        if step == 1 or step % STEP_PRINT_EVERY == 0 or step == total_steps:
            print(f"  step {step:03d}/{total_steps} | g={g_loss.item():.4f} d={d_loss.item():.4f}"
                  f" l1_dB={l1_val.item():.3f} ssim={ssim_val.item():.4f}", flush=True)
    return np.mean(g_losses), np.mean(d_losses), np.mean(l1s), np.mean(ssims)


@torch.no_grad()
def evaluate_indist(G, ema, loader, recon_loss):
    """In-distribution validation across N=4 + N=8 (no gradients)."""
    live = ema.swap_into(G)
    try:
        G.eval()
        yt, yp = [], []
        for x, target_dB, matlab_dB, _null in loader:
            x = x.to(DEVICE); target_dB = target_dB.to(DEVICE); matlab_dB = matlab_dB.to(DEVICE)
            delta_n = G(x)
            pred_dB = recon_loss.compose(delta_n, matlab_dB)
            yt.append(target_dB.cpu().numpy()); yp.append(pred_dB.cpu().numpy())
        yt = np.concatenate(yt).reshape(-1); yp = np.concatenate(yp).reshape(-1)
        return {"rmse": float(rmse(yt, yp)),
                "mae":  float(mae(yt, yp)),
                "pearson": float(pearson_correlation(yt, yp))}
    finally:
        ema.restore(G, live)


@torch.no_grad()
def evaluate_16x16(G, ema, loader, recon_loss):
    """SELECTION-ONLY hfss_16x16 evaluation (no gradients ever).

    EXPLICITLY MARKED IN LOG so it's clear when 16x16 is touched. Wrapped in
    no_grad so autograd can't build a graph; uses EMA weights so live training
    weights are never seen during this evaluation.
    """
    live = ema.swap_into(G)
    try:
        G.eval()
        yt, yp = [], []
        for x, target_dB, matlab_dB in loader:
            x = x.to(DEVICE); target_dB = target_dB.to(DEVICE); matlab_dB = matlab_dB.to(DEVICE)
            delta_n = G(x)
            pred_dB = recon_loss.compose(delta_n, matlab_dB)
            yt.append(target_dB.cpu().numpy()); yp.append(pred_dB.cpu().numpy())
        yt = np.concatenate(yt).reshape(-1); yp = np.concatenate(yp).reshape(-1)
        return {"rmse": float(rmse(yt, yp)),
                "mae":  float(mae(yt, yp)),
                "pearson": float(pearson_correlation(yt, yp))}
    finally:
        ema.restore(G, live)


def main():
    # Hard-fail if CUDA isn't actually being used — this run is too long to
    # silently fall back to CPU. The spec assumes GPU + AMP.
    if DEVICE.type != "cuda":
        raise RuntimeError(
            f"DEVICE is {DEVICE} but this training expects CUDA. "
            f"torch.cuda.is_available()={torch.cuda.is_available()}. "
            f"Install a CUDA-enabled torch build before re-launching."
        )
    amp_dtype_str = str(AMP_DTYPE).replace("torch.", "") if USE_AMP else "off"
    print(
        f"[device] torch={torch.__version__}  cuda={torch.version.cuda}  "
        f"device={DEVICE} ({torch.cuda.get_device_name(0)})  "
        f"AMP={USE_AMP} (dtype={amp_dtype_str}, grad_scaler={USE_GRAD_SCALER})  "
        f"anchor_dropout_p={ANCHOR_DROPOUT_P}",
        flush=True,
    )

    for p in (H5_2TO4, H5_4TO8, H5_16X16, NORM_COMBINED, M16_TEST,
              SPLITS_2TO4, SPLITS_4TO8):
        if not p.exists():
            raise FileNotFoundError(p)

    # Norm stats and sigma_residual
    s = np.load(NORM_COMBINED)
    mean_mat = s["mean"]; std_mat = s["std"]
    sigma_res = float(s["residual_std"])
    print(f"Loaded combined norm stats: mean range [{mean_mat.min():.2f}, {mean_mat.max():.2f}], "
          f"std range [{std_mat.min():.2f}, {std_mat.max():.2f}], sigma_residual={sigma_res:.3f} dB",
          flush=True)

    # Splits
    sp24 = np.load(SPLITS_2TO4)
    sp48 = np.load(SPLITS_4TO8)
    tr4 = np.sort(sp24["train"].astype(np.int64))
    va4 = np.sort(sp24["val"].astype(np.int64))
    tr8 = np.sort(sp48["train"].astype(np.int64))
    va8 = np.sort(sp48["val"].astype(np.int64))
    test48 = np.sort(sp48["test"].astype(np.int64))
    val16_idx = test48[:100]
    test16_idx = test48[100:]
    print(f"Train: N=4 {len(tr4)} + N=8 {len(tr8)} = {len(tr4)+len(tr8)} samples",
          flush=True)
    print(f"In-dist val: N=4 {len(va4)} + N=8 {len(va8)} = {len(va4)+len(va8)} samples",
          flush=True)
    print(f"val_16x16 (selection): {len(val16_idx)} samples", flush=True)
    print(f"test_16x16 (held-out, never touched here): {len(test16_idx)} samples",
          flush=True)

    # ── Datasets ──
    tr4_ds = ScaleDatasetWithAnchor(H5_2TO4, tr4, 4, "matlab_4x4", "hfss_4x4",
                                    anchor_key="hfss_pred_2x2",
                                    mean_mat=mean_mat, std_mat=std_mat,
                                    augment_noise=True,
                                    anchor_dropout_p=ANCHOR_DROPOUT_P)
    tr8_ds = ScaleDatasetWithAnchor(H5_4TO8, tr8, 8, "matlab_8x8", "hfss_8x8",
                                    anchor_key="hfss_pred_4x4",
                                    mean_mat=mean_mat, std_mat=std_mat,
                                    augment_noise=True,
                                    anchor_dropout_p=ANCHOR_DROPOUT_P)
    train_ds = ConcatDataset([tr4_ds, tr8_ds])
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              drop_last=True)

    va4_ds = ScaleDatasetWithAnchor(H5_2TO4, va4, 4, "matlab_4x4", "hfss_4x4",
                                    anchor_key="hfss_pred_2x2",
                                    mean_mat=mean_mat, std_mat=std_mat,
                                    augment_noise=False)
    va8_ds = ScaleDatasetWithAnchor(H5_4TO8, va8, 8, "matlab_8x8", "hfss_8x8",
                                    anchor_key="hfss_pred_4x4",
                                    mean_mat=mean_mat, std_mat=std_mat,
                                    augment_noise=False)
    va_loader = DataLoader(ConcatDataset([va4_ds, va8_ds]), batch_size=BATCH,
                           shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # 16x16 monitor — uses precomputed analytical matlab_16x16 (200 betas total;
    # we take the first 100 of the 4to8 sorted test split as val_16x16).
    m16_pack = np.load(M16_TEST)
    m16_test_idx = m16_pack["test_idx"]
    m16_arr = m16_pack["arr"].astype(np.float32)         # (200, 181, 360)
    if not np.array_equal(m16_test_idx, test48):
        raise RuntimeError("matlab_16x16_test.npz test_idx mismatch with split_indices_4to8 test")
    val16_ds = Val16x16Dataset(val16_idx, m16_arr[:100], mean_mat, std_mat)
    val16_loader = DataLoader(val16_ds, batch_size=BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # ── Model ──
    # 5 input channels (matlab_n, dpx, dpy, scale_tok, anchor_n);
    # D sees x (5) + y (1) = 6 channels.
    G = EnhancedResUNetGenerator(in_ch=5, out_ch=1, base=GEN_BASE,
                                 attn_heads=ATTN_HEADS).to(DEVICE)
    D = PatchDiscriminator(in_channels=6, base=64).to(DEVICE)
    print(f"G params: {sum(p.numel() for p in G.parameters())/1e6:.2f}M  "
          f"D params: {sum(p.numel() for p in D.parameters())/1e6:.2f}M", flush=True)

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
    sched_g = optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode="min",
        factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR)
    sched_d = optim.lr_scheduler.ReduceLROnPlateau(opt_d, mode="min",
        factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR)
    scaler_g = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_GRAD_SCALER)
    scaler_d = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_GRAD_SCALER)
    bce = nn.BCEWithLogitsLoss()
    recon_loss = ResidualReconLoss(sigma_res=sigma_res).to(DEVICE)
    ema = EMA(G, decay=EMA_DECAY)
    if last_g_ema.exists():
        ema.shadow = torch.load(last_g_ema, map_location=DEVICE)
        print(f"Resumed EMA shadow from {last_g_ema.name}", flush=True)

    best_rmse_16 = float("inf"); best_epoch = -1; epochs_without_improve = 0
    if last_g_ema.exists():
        # Seed best_rmse_16 from current EMA so we don't overwrite a better best.
        seed = evaluate_16x16(G, ema, val16_loader, recon_loss)
        best_rmse_16 = seed["rmse"]
        print(f"Seeded best_rmse_16 from resumed EMA: {best_rmse_16:.5f}", flush=True)

    t0 = time.time()
    epoch_times = []   # rolling per-epoch wall-clock (train + eval + save)
    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.time()
        cur_lr_g = opt_g.param_groups[0]["lr"]
        print(f"\n=== Epoch {epoch}/{EPOCHS}  (G_lr={cur_lr_g:.2e}) ===", flush=True)

        train_t0 = time.time()
        g_avg, d_avg, l1_avg, ssim_avg = train_one_epoch(
            G, D, ema, train_loader, opt_g, opt_d, scaler_g, scaler_d, bce, recon_loss
        )
        train_dt = time.time() - train_t0
        print(f"  train (N=4 + N=8 only): g_loss={g_avg:.4f}  d_loss={d_avg:.4f}  "
              f"l1_dB={l1_avg:.3f}  ssim={ssim_avg:.4f}  "
              f"[{train_dt/60:.2f} min]", flush=True)

        eval_dt = 0.0
        if epoch % VALIDATE_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            eval_t0 = time.time()
            m_indist = evaluate_indist(G, ema, va_loader, recon_loss)
            print(f"  val_indist (N=4+N=8, EMA): rmse={m_indist['rmse']:.4f}  "
                  f"mae={m_indist['mae']:.4f}  r={m_indist['pearson']:.4f}", flush=True)

            m_16 = evaluate_16x16(G, ema, val16_loader, recon_loss)
            print(f"  val_16x16  (EMA, NO GRADIENTS): rmse={m_16['rmse']:.4f}  "
                  f"mae={m_16['mae']:.4f}  r={m_16['pearson']:.4f}", flush=True)
            eval_dt = time.time() - eval_t0

            if m_16["rmse"] < best_rmse_16 - 1e-5:
                best_rmse_16 = m_16["rmse"]; best_epoch = epoch
                epochs_without_improve = 0
                torch.save(ema.state_dict(), CKPT_DIR / "best_generator.pt")
                print(f"  * new best (val_16x16 rmse {best_rmse_16:.5f}) saved to "
                      f"best_generator.pt", flush=True)
            else:
                epochs_without_improve += VALIDATE_EVERY
                print(f"  (no val_16x16 improvement for {epochs_without_improve} epochs)",
                      flush=True)

            # ReduceLROnPlateau driven by val_16x16 (the metric we actually care about).
            sched_g.step(m_16["rmse"]); sched_d.step(m_16["rmse"])

        torch.save(G.state_dict(), CKPT_DIR / "last_generator.pt")
        torch.save(ema.state_dict(), CKPT_DIR / "last_generator_ema.pt")
        torch.save(D.state_dict(), CKPT_DIR / "last_discriminator.pt")
        if epoch % SAVE_EVERY == 0:
            torch.save(ema.state_dict(), CKPT_DIR / f"generator_ema_epoch_{epoch:03d}.pt")

        epoch_dt = time.time() - epoch_t0
        epoch_times.append(epoch_dt)
        elapsed_min = (time.time() - t0) / 60
        # Rolling average over the last ≤10 epochs gives a stable ETA after the
        # initial cuda warmup epoch (which is always a bit slower).
        recent = epoch_times[-10:]
        avg_recent = sum(recent) / len(recent)
        remaining = max(0, EPOCHS - epoch)
        eta_min = remaining * avg_recent / 60
        print(f"  [time] epoch={epoch_dt/60:.2f} min  "
              f"(train={train_dt/60:.2f}, eval={eval_dt/60:.2f})  "
              f"elapsed={elapsed_min:.1f} min  "
              f"avg10={avg_recent/60:.2f} min/ep  "
              f"eta={eta_min:.0f} min ({eta_min/60:.1f} h)", flush=True)

        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}; best epoch was {best_epoch} "
                  f"(val_16x16 rmse {best_rmse_16:.5f})", flush=True)
            break

    total = (time.time() - t0) / 60
    print(f"\nTraining done in {total:.1f} min. Best epoch {best_epoch} "
          f"val_16x16 rmse {best_rmse_16:.5f}.  Checkpoints in {CKPT_DIR}",
          flush=True)


if __name__ == "__main__":
    main()