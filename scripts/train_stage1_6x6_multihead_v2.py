"""
Stage 1 at 6x6 — regional gated FiLM multi-head ResUNet (v2).

Self-contained script: model, loss, dataset, and training loop in one file.

Improvements over v1 (stage1_multihead_resunet.py):
  - Sharper amplitude gate (no floor in main beam; slope/margin tuned for HPBW protection)
  - Dual alpha: alpha_main ~ 0, alpha_null ~ 1 (region-specific residual scale)
  - FiLM modulation of decoder features from bottleneck (aux heads share representation)
  - Aux-driven gate bias from bottleneck (null/PSL context steers spatial gate)
  - Composed-pattern null loss + sidelobe-band loss (spatial, not just scalar heads)
  - Validation on composed dB RMSE (main+SL and null regions), not normalised residual only

Usage:
    python -m scripts.train_stage1_6x6_multihead_v2
    python -m scripts.train_stage1_6x6_multihead_v2 --resume

Checkpoints: checkpoints/stage1_6x6_multihead_v2/
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import minimum_filter
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset

from scripts.train_cgan_2to4_fusion import DownRes, ResidualBlock, UpRes
from scripts.train_cgan_2to4_fusion_no_m4 import (
    AMP_DEVICE_TYPE,
    ATTN_HEADS,
    EMA_DECAY,
    GEN_BASE,
    LAMBDA_L1,
    LAMBDA_SSIM,
    NOISE_STD,
    SelfAttention2D,
    USE_AMP,
    ssim_dB,
)
from src.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DEVICE,
    MAX_EPOCHS,
    NULL_THRESHOLD_DB,
    PROCESSED_DIR,
    RANDOM_SEED,
)
from src.training.metrics import mae, pearson_correlation, rmse

# ── Paths ───────────────────────────────────────────────────────────────────
NPZ_EXTRAS = PROCESSED_DIR / "stage1_6x6_extras.npz"
NORM_6X6 = PROCESSED_DIR / "norm_stats_stage1_6x6.npz"
SPLIT_6X6 = PROCESSED_DIR / "split_indices_stage1_6x6.npz"
CKPT_DIR = CHECKPOINTS_DIR / "stage1_6x6_multihead_v2"
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
AUX_RAMP_EPOCHS = 20

LAMBDA_PSL = 3.0
LAMBDA_NULL = 8.0
LAMBDA_MAINBEAM = 35.0
LAMBDA_COMPOSED_NULL = 20.0
LAMBDA_SIDELOBE = 8.0
NULL_LOSS_WEIGHT = 3.0
SIDELOBE_WEIGHT = 2.0
MAINBEAM_DB = 3.0

GATE_SLOPE = 0.65
GATE_MARGIN = 5.0
GATE_FLOOR = 0.0
GATE_FLOOR_RAMP_EPOCHS = 15
GATE_FLOOR_START = 0.08

VAL_WEIGHT_MAIN_SL = 0.40
VAL_WEIGHT_NULL = 0.60

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# ── Target extraction (numpy) ─────────────────────────────────────────────────

def compute_psl_db(pattern_db: np.ndarray) -> float:
    peak_idx = np.unravel_index(int(np.argmax(pattern_db)), pattern_db.shape)
    pt, pp = peak_idx
    peak_val = float(pattern_db[pt, pp])
    cut = pattern_db[:, pp]
    peaks, _ = find_peaks(cut, distance=3)
    if len(peaks) < 2:
        return -13.0
    main_loc = int(np.argmin(np.abs(peaks - pt)))
    sl_peaks = np.delete(peaks, main_loc)
    if len(sl_peaks) == 0:
        return -13.0
    return float(cut[sl_peaks].max()) - peak_val


def compute_topk_null_depths(
    pattern_db: np.ndarray,
    k: int = 10,
    null_threshold_db: float = -20.0,
) -> np.ndarray:
    peak = float(pattern_db.max())
    null_mask = pattern_db < (peak + null_threshold_db)
    lm = minimum_filter(pattern_db, size=5)
    is_min = (pattern_db == lm) & null_mask
    if not is_min.any():
        return np.full(k, null_threshold_db, dtype=np.float32)
    depths = pattern_db[is_min].astype(np.float32)
    order = np.argsort(depths)[:k]
    top = depths[order] - peak
    if len(top) < k:
        pad = top[-1] if len(top) > 0 else null_threshold_db
        top = np.pad(top, (0, k - len(top)), constant_values=pad)
    return top.astype(np.float32)


# ── Model ───────────────────────────────────────────────────────────────────

class RegionalGatedFiLMMultiHeadResUNet(nn.Module):
    """
    Gated residual ResUNet with regional alpha and FiLM decoder modulation.

    pred_residual_n = gate * (alpha_main*(1-g) + alpha_null*g) * residual_raw
    gate = sigmoid(spatial + aux_bias) * amplitude_gate(sb), clamped to gate_floor
    """

    def __init__(
        self,
        in_ch: int = 5,
        base: int = GEN_BASE,
        attn_heads: int = ATTN_HEADS,
        top_k: int = 10,
        sb_mean: np.ndarray | None = None,
        sb_std: np.ndarray | None = None,
        res_mean: np.ndarray | None = None,
        res_std: np.ndarray | None = None,
        gate_slope: float = GATE_SLOPE,
        gate_margin: float = GATE_MARGIN,
        gate_floor: float = GATE_FLOOR,
    ):
        super().__init__()
        self.top_k = top_k
        self.gate_floor = gate_floor

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

        b_dim = base * 16
        self.film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(b_dim, base * 2),
        )
        self.gate_aux = nn.Linear(b_dim, 1)
        self.residual_head = nn.Conv2d(base, 1, 1)
        self.gate_head = nn.Sequential(
            nn.Conv2d(base + 1, base // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, base // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, 1, 1),
        )
        self.alpha_main = nn.Parameter(torch.tensor(0.05))
        self.alpha_null = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("gate_slope", torch.tensor(gate_slope))
        self.register_buffer("gate_margin", torch.tensor(gate_margin))
        nn.init.zeros_(self.gate_head[-1].bias)
        nn.init.zeros_(self.gate_aux.bias)

        self.psl_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(b_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.null_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(b_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, top_k),
        )

        if sb_mean is not None:
            self.register_buffer(
                "sb_mean", torch.from_numpy(sb_mean.astype(np.float32))[None, None])
            self.register_buffer(
                "sb_std", torch.from_numpy(sb_std.astype(np.float32))[None, None])
            self.register_buffer(
                "res_mean", torch.from_numpy(res_mean.astype(np.float32))[None, None])
            self.register_buffer(
                "res_std", torch.from_numpy(res_std.astype(np.float32))[None, None])

    def _encode_decode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s1, p1 = self.d1(x)
        s2, p2 = self.d2(p1)
        s3, p3 = self.d3(p2)
        s4, p4 = self.d4(p3)
        b = self.bottleneck(p4)
        b = self.attn(b)
        dec = self.u1(b, s4)
        dec = self.u2(dec, s3)
        dec = self.u3(dec, s2)
        dec = self.u4(dec, s1)
        if dec.shape[-2:] != x.shape[-2:]:
            dec = F.interpolate(dec, size=x.shape[-2:], mode="bilinear", align_corners=False)
        film = self.film(b)
        gamma, beta = film.chunk(2, dim=1)
        dec = dec * (1.0 + 0.1 * gamma[..., None, None]) + 0.1 * beta[..., None, None]
        return b, dec

    def _amplitude_gate(self, sb_n: torch.Tensor) -> torch.Tensor:
        sb_db = sb_n * self.sb_std + self.sb_mean
        rel = sb_db - sb_db.amax(dim=(-2, -1), keepdim=True)
        return torch.sigmoid(self.gate_slope * (-rel - self.gate_margin))

    def _monotonic_null_depths(self, raw: torch.Tensor) -> torch.Tensor:
        d0 = -F.softplus(raw[:, :1])
        if raw.shape[1] == 1:
            return d0
        steps = F.softplus(raw[:, 1:])
        return torch.cat([d0, d0 + torch.cumsum(steps, dim=1)], dim=1)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        sb_n = x[:, 1:2]
        b, dec = self._encode_decode(x)

        residual_raw = self.residual_head(dec)
        gate_logit = self.gate_head(torch.cat([dec, sb_n], dim=1))
        gate_logit = gate_logit + self.gate_aux(b)[..., None, None]
        gate_spatial = torch.sigmoid(gate_logit)
        amp_gate = self._amplitude_gate(sb_n)
        gate = (gate_spatial * amp_gate).clamp(min=self.gate_floor)

        alpha_eff = self.alpha_main * (1.0 - gate) + self.alpha_null * gate
        residual_gated = alpha_eff * gate * residual_raw

        if not return_aux:
            return residual_gated

        return {
            "residual": residual_gated,
            "residual_raw": residual_raw,
            "gate": gate,
            "alpha_eff": alpha_eff,
            "psl": self.psl_head(b).squeeze(-1),
            "null_depths": self._monotonic_null_depths(self.null_head(b)),
        }


# ── Loss ────────────────────────────────────────────────────────────────────

class RegionalMultiHeadLoss(nn.Module):
    """Reconstruction + composed null/sidelobe + scalar aux + main-beam preservation."""

    def __init__(
        self,
        res_mean: np.ndarray,
        res_std: np.ndarray,
        sb_mean: np.ndarray,
        sb_std: np.ndarray,
        lambda_l1: float = LAMBDA_L1,
        lambda_ssim: float = LAMBDA_SSIM,
        lambda_psl: float = LAMBDA_PSL,
        lambda_null: float = LAMBDA_NULL,
        lambda_mainbeam: float = LAMBDA_MAINBEAM,
        lambda_composed_null: float = LAMBDA_COMPOSED_NULL,
        lambda_sidelobe: float = LAMBDA_SIDELOBE,
        null_weight: float = NULL_LOSS_WEIGHT,
        sidelobe_weight: float = SIDELOBE_WEIGHT,
        mainbeam_db: float = MAINBEAM_DB,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_psl = lambda_psl
        self.lambda_null = lambda_null
        self.lambda_mainbeam = lambda_mainbeam
        self.lambda_composed_null = lambda_composed_null
        self.lambda_sidelobe = lambda_sidelobe
        self.null_weight = null_weight
        self.sidelobe_weight = sidelobe_weight
        self.mainbeam_db = mainbeam_db

        self.register_buffer("res_mean", torch.from_numpy(res_mean.astype(np.float32))[None, None])
        self.register_buffer("res_std", torch.from_numpy(res_std.astype(np.float32))[None, None])
        self.register_buffer("sb_mean", torch.from_numpy(sb_mean.astype(np.float32))[None, None])
        self.register_buffer("sb_std", torch.from_numpy(sb_std.astype(np.float32))[None, None])

    def _tiered_weights(self, sb_db: torch.Tensor, null_mask: torch.Tensor) -> torch.Tensor:
        rel = sb_db - sb_db.amax(dim=(-2, -1), keepdim=True)
        sidelobe = ((rel < -self.mainbeam_db) & (rel >= -20.0)).float()
        weight = 1.0 + (self.null_weight - 1.0) * null_mask
        weight = weight + (self.sidelobe_weight - 1.0) * sidelobe
        return weight

    def forward(
        self,
        out: dict,
        target_res_n: torch.Tensor,
        null_mask: torch.Tensor,
        target_psl: torch.Tensor,
        target_null_depths: torch.Tensor,
        sb_n: torch.Tensor,
        aux_scale: float = 1.0,
    ):
        pred_n = out["residual"]
        pred_db = pred_n * self.res_std + self.res_mean
        target_db = target_res_n * self.res_std + self.res_mean
        sb_db = sb_n * self.sb_std + self.sb_mean

        weight = self._tiered_weights(sb_db, null_mask)
        l1 = (torch.abs(pred_db - target_db) * weight).mean()
        composed_pred = sb_db + pred_db
        composed_true = sb_db + target_db
        ssim_val = ssim_dB(composed_pred, composed_true)
        recon = self.lambda_l1 * l1 + self.lambda_ssim * (1.0 - ssim_val)

        psl_loss = F.l1_loss(out["psl"], target_psl)
        null_loss = F.l1_loss(out["null_depths"], target_null_depths)

        sb_peak = sb_db.amax(dim=(-2, -1), keepdim=True)
        main_mask = (sb_db >= sb_peak - self.mainbeam_db).float()
        mainbeam_loss = (
            (torch.abs(composed_pred - sb_db) * main_mask).sum()
            / main_mask.sum().clamp(min=1.0)
        )

        rel = sb_db - sb_peak
        sidelobe_mask = ((rel < -self.mainbeam_db) & (rel >= -20.0)).float()
        sidelobe_loss = (
            (torch.abs(composed_pred - composed_true) * sidelobe_mask).sum()
            / sidelobe_mask.sum().clamp(min=1.0)
        )

        composed_null_loss = (
            (torch.abs(composed_pred - composed_true) * null_mask).sum()
            / null_mask.sum().clamp(min=1.0)
        )

        total = recon + aux_scale * (
            self.lambda_psl * psl_loss
            + self.lambda_null * null_loss
            + self.lambda_mainbeam * mainbeam_loss
            + self.lambda_composed_null * composed_null_loss
            + self.lambda_sidelobe * sidelobe_loss
        )

        stats = {
            "loss": total.detach(),
            "l1_db": l1.detach(),
            "ssim": ssim_val.detach(),
            "psl": psl_loss.detach(),
            "null": null_loss.detach(),
            "mainbeam": mainbeam_loss.detach(),
            "composed_null": composed_null_loss.detach(),
            "sidelobe": sidelobe_loss.detach(),
            "gate_mean": out["gate"].mean().detach(),
        }
        return total, stats


# ── EMA ─────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, module: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(module.state_dict())
        for v in self.shadow.values():
            if torch.is_tensor(v) and v.is_floating_point():
                v.requires_grad_(False)

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for k, v in module.state_dict().items():
            if not torch.is_tensor(v):
                continue
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return self.shadow

    def swap_into(self, module: nn.Module):
        live = copy.deepcopy(module.state_dict())
        module.load_state_dict(self.shadow, strict=True)
        return live

    def restore(self, module: nn.Module, live) -> None:
        module.load_state_dict(live, strict=True)


# ── Dataset ─────────────────────────────────────────────────────────────────

class Stage1_6x6_MultiHeadV2Dataset(Dataset):
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
        res = hf6 - sb6

        m6_n = (m6 - self.s["matlab_6x6_mean"]) / self.s["matlab_6x6_std"]
        sb6_n = (sb6 - self.s["sub_block_6x6_mean"]) / self.s["sub_block_6x6_std"]
        fp_n = (fp - self.s["fingerprint_mean"]) / self.s["fingerprint_std"]
        res_n = (res - self.s["residual_mean"]) / self.s["residual_std"]

        dpx_n = np.full_like(m6_n, self.dpx[idx] / 180.0, dtype=np.float32)
        dpy_n = np.full_like(m6_n, self.dpy[idx] / 180.0, dtype=np.float32)

        peak = float(hf6.max())
        null_mask = (hf6 < (peak + NULL_THRESHOLD_DB)).astype(np.float32)
        psl = np.float32(compute_psl_db(hf6))
        null_depths = compute_topk_null_depths(
            hf6, k=self.top_k, null_threshold_db=NULL_THRESHOLD_DB)

        x = np.stack([m6_n, sb6_n, fp_n, dpx_n, dpy_n], axis=0)
        if self.augment_noise and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=(3,) + m6_n.shape).astype(np.float32)
            x[:3] += noise

        return (
            torch.from_numpy(x),
            torch.from_numpy(res_n[None, ...]),
            torch.from_numpy(null_mask[None, ...]),
            torch.tensor(psl),
            torch.from_numpy(null_depths),
        )


# ── Training helpers ──────────────────────────────────────────────────────────

def aux_scale(epoch: int) -> float:
    return min(1.0, epoch / AUX_RAMP_EPOCHS)


def gate_floor_for_epoch(epoch: int) -> float:
    if GATE_FLOOR_RAMP_EPOCHS <= 0:
        return GATE_FLOOR
    t = min(1.0, epoch / GATE_FLOOR_RAMP_EPOCHS)
    return GATE_FLOOR_START + t * (GATE_FLOOR - GATE_FLOOR_START)


def train_one_epoch(model, ema, loader, opt, scaler, criterion, epoch):
    model.train()
    model.gate_floor = gate_floor_for_epoch(epoch)
    scale = aux_scale(epoch)
    totals = {k: [] for k in (
        "loss", "l1", "ssim", "psl", "null", "mainbeam", "composed_null", "sidelobe", "gate",
    )}
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
        totals["l1"].append(float(stats["l1_db"]))
        totals["ssim"].append(float(stats["ssim"]))
        totals["psl"].append(float(stats["psl"]))
        totals["null"].append(float(stats["null"]))
        totals["mainbeam"].append(float(stats["mainbeam"]))
        totals["composed_null"].append(float(stats["composed_null"]))
        totals["sidelobe"].append(float(stats["sidelobe"]))
        totals["gate"].append(float(stats["gate_mean"]))
        if step == 1 or step % STEP_PRINT_EVERY == 0 or step == total_steps:
            print(
                f"  step {step:03d}/{total_steps} | loss={stats['loss'].item():.4f}"
                f" l1={stats['l1_db'].item():.3f} null_c={stats['composed_null'].item():.3f}"
                f" mb={stats['mainbeam'].item():.4f} gate={stats['gate_mean'].item():.3f}",
                flush=True,
            )
    return {k: float(np.mean(v)) for k, v in totals.items()}


@torch.no_grad()
def evaluate_composed(model, ema, loader, res_mean, res_std, sb_mean, sb_std):
    """Composed-pattern dB RMSE on main+SL (>= -20 dB) and null regions."""
    live = ema.swap_into(model)
    model.gate_floor = GATE_FLOOR
    try:
        model.eval()
        main_sl_err, null_err, all_err = [], [], []
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
            main_sl = composed_true >= (sb_peak - 20.0)
            for i in range(err.shape[0]):
                e = err[i, 0].cpu().numpy()
                t = composed_true[i, 0].cpu().numpy()
                p = t + e
                nm = null_mask[i, 0].cpu().numpy() > 0.5
                ms = main_sl[i, 0].cpu().numpy()
                if ms.any():
                    main_sl_err.append(rmse(p[ms], t[ms]))
                if nm.any():
                    null_err.append(rmse(p[nm], t[nm]))
                all_err.append(rmse(p.ravel(), t.ravel()))

        main_sl_rmse = float(np.mean(main_sl_err)) if main_sl_err else float("nan")
        null_rmse = float(np.mean(null_err)) if null_err else float("nan")
        full_rmse = float(np.mean(all_err)) if all_err else float("nan")
        score = VAL_WEIGHT_MAIN_SL * main_sl_rmse + VAL_WEIGHT_NULL * null_rmse
        return {
            "score": score,
            "main_sl_rmse": main_sl_rmse,
            "null_rmse": null_rmse,
            "full_rmse": full_rmse,
        }
    finally:
        ema.restore(model, live)


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


def main():
    parser = argparse.ArgumentParser(description="Train regional gated FiLM multi-head v2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for p in (NPZ_EXTRAS, NORM_6X6, SPLIT_6X6):
        if not p.exists():
            raise FileNotFoundError(p)

    print("Loading data ...", flush=True)
    e = np.load(NPZ_EXTRAS)
    matlab_6x6 = e["matlab_6x6"]
    sub_block_6x6 = e["sub_block_6x6"]
    hfss_6x6 = e["hfss_6x6"]
    fingerprint = e["fingerprint"]
    dpx = e["dpx"]
    dpy = e["dpy"]

    stats = dict(np.load(NORM_6X6))
    sp = np.load(SPLIT_6X6)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"Splits: {len(tr)} train / {len(va)} val / {len(te)} test (test reserved)", flush=True)

    tr_ds = Stage1_6x6_MultiHeadV2Dataset(
        tr, stats, matlab_6x6, sub_block_6x6, fingerprint, hfss_6x6, dpx, dpy,
        augment_noise=True)
    va_ds = Stage1_6x6_MultiHeadV2Dataset(
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

    G = RegionalGatedFiLMMultiHeadResUNet(
        in_ch=5, base=GEN_BASE, attn_heads=ATTN_HEADS, top_k=TOP_K_NULLS,
        sb_mean=stats["sub_block_6x6_mean"], sb_std=stats["sub_block_6x6_std"],
        res_mean=stats["residual_mean"], res_std=stats["residual_std"],
        gate_slope=GATE_SLOPE, gate_margin=GATE_MARGIN, gate_floor=GATE_FLOOR_START,
    ).to(DEVICE)
    print(f"G params: {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M", flush=True)
    if DEVICE.type == "cuda":
        print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print(f"Device: {DEVICE}  *** WARNING: training on CPU ***", flush=True)

    last_g = CKPT_DIR / "last_generator.pt"
    last_g_ema = CKPT_DIR / "last_generator_ema.pt"
    if args.resume and last_g.exists():
        G.load_state_dict(torch.load(last_g, map_location=DEVICE, weights_only=True))
        print(f"Resumed from {last_g.name}", flush=True)
    elif last_g.exists() and not args.resume:
        print("Fresh start (use --resume to continue). Ignoring old checkpoints.", flush=True)

    opt = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR)
    scaler = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_AMP)
    criterion = RegionalMultiHeadLoss(
        stats["residual_mean"], stats["residual_std"],
        stats["sub_block_6x6_mean"], stats["sub_block_6x6_std"],
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
        gf = gate_floor_for_epoch(epoch)
        print(
            f"\n=== Epoch {epoch}/{EPOCHS}  (lr={cur_lr:.2e}, aux={aux_scale(epoch):.2f}, "
            f"gate_floor={gf:.3f}) ===",
            flush=True,
        )
        tr_stats = train_one_epoch(G, ema, tr_loader, opt, scaler, criterion, epoch)
        print(
            f"  train: loss={tr_stats['loss']:.4f}  l1={tr_stats['l1']:.3f}  "
            f"null_c={tr_stats['composed_null']:.3f}  sidelobe={tr_stats['sidelobe']:.3f}  "
            f"mainbeam={tr_stats['mainbeam']:.4f}  gate={tr_stats['gate']:.3f}  "
            f"a_main={float(G.alpha_main):.3f}  a_null={float(G.alpha_null):.3f}",
            flush=True,
        )

        if epoch % VALIDATE_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            comp = evaluate_composed(G, ema, va_loader, res_mean_t, res_std_t, sb_mean_t, sb_std_t)
            res_m = evaluate_residual(G, ema, va_loader)
            print(
                f"  val (EMA): composed score={comp['score']:.4f}  "
                f"main+SL={comp['main_sl_rmse']:.4f}  null={comp['null_rmse']:.4f}  "
                f"full={comp['full_rmse']:.4f}  |  res_n rmse={res_m['rmse']:.4f}",
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
