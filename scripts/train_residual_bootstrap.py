"""
Three-phase bootstrap for the multi-scale residual generator (5-ch with anchor).

Same G / D / losses as ``train_residual_multiscale`` throughout.

Phase A — 2x2 + 4x4 only (2to4 HDF5, no 8x8 in the loss)
    Full pass over train betas at N=4 and, unless ``--skip-n2``, at N=2
    (concatenated datasets). Supervision prefers ``hfss_pred_4x4`` /
    ``hfss_pred_2x2`` when present. Optional ``--og-sized-epochs`` uses a
    smaller rotating N=2 block so samples/epoch matches ``len(tr4)+len(tr8)``.

Phase B — Pseudo 8x8 from the Phase A residual generator
    Load Phase A ``best_generator.pt`` (EMA), forward at N=8 on 4->8 rows
    (``matlab_8x8`` + ``hfss_pred_4x4`` anchor), save max-normed patterns
    to a ``.npz`` (default: train+val indices for Phase C).

Phase C — Same layout as the original multiscale trainer: N=4 + N=8
    ``ConcatDataset``: real 4x4 supervision (2to4) + 8x8 with **pseudo** targets
    from Phase B (no ``hfss_8x8`` in the loss for the 8x8 branch).

Examples
--------
    python -m scripts.train_residual_bootstrap phase_a
    python -m scripts.train_residual_bootstrap export_pseudo
    python -m scripts.train_residual_bootstrap phase_c --pseudo \\
        processed/hfss_8x8_pseudo_bootstrap.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.config import CHECKPOINTS_DIR, DEVICE, NULL_THRESHOLD_DB, PROCESSED_DIR, RANDOM_SEED
from scripts.train_cgan import PatchDiscriminator
from scripts.train_cgan_2to4_fusion_no_m4 import (
    ATTN_HEADS,
    EMA,
    EMA_DECAY,
    GEN_BASE,
    EnhancedResUNetGenerator,
)
from scripts.train_residual_multiscale import (
    AMP_DEVICE_TYPE,
    AMP_DTYPE,
    ANCHOR_DROPOUT_P,
    BATCH,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    H5_16X16,
    H5_2TO4,
    H5_4TO8,
    LR_D,
    LR_G,
    M16_TEST,
    NOISE_STD,
    NUM_WORKERS,
    PIN_MEMORY,
    PLATEAU_FACTOR,
    PLATEAU_MIN_LR,
    PLATEAU_PATIENCE,
    SAVE_EVERY,
    SPLITS_2TO4,
    SPLITS_4TO8,
    STEP_PRINT_EVERY,
    USE_AMP,
    USE_GRAD_SCALER,
    NORM_COMBINED,
    ResidualReconLoss,
    ScaleDatasetWithAnchor,
    Val16x16Dataset,
    evaluate_16x16,
    evaluate_indist,
    train_one_epoch,
)

CKPT_PHASE_A = CHECKPOINTS_DIR / "residual_bootstrap_phase_a"
CKPT_PHASE_C = CHECKPOINTS_DIR / "residual_bootstrap_phase_c"
DEFAULT_PSEUDO_NPZ = PROCESSED_DIR / "hfss_8x8_pseudo_bootstrap.npz"

# Validation cadence for phase A/C only (do not couple to train_residual_multiscale).
BOOTSTRAP_VALIDATE_EVERY = 1


def _require_cuda() -> None:
    if DEVICE.type != "cuda":
        raise RuntimeError(
            f"DEVICE is {DEVICE} but bootstrap training expects CUDA "
            f"(torch.cuda.is_available()={torch.cuda.is_available()})."
        )


def _h5_has_dataset(h5_path: Path, key: str) -> bool:
    with h5py.File(str(h5_path), "r") as f:
        return key in f


def pick_hfss_supervision_key(h5_path: Path, preferred: str, fallback: str) -> str:
    """Prefer cGAN / cascade HFSS predictions as supervision targets when present."""
    with h5py.File(str(h5_path), "r") as f:
        if preferred in f:
            if preferred != fallback:
                print(
                    f"[bootstrap supervision] {h5_path.name}: using {preferred!r}",
                    flush=True,
                )
            return preferred
    print(
        f"[bootstrap supervision] {h5_path.name}: {preferred!r} missing — "
        f"using {fallback!r}",
        flush=True,
    )
    return fallback


def _resolve_n4_supervision_key(args: argparse.Namespace) -> str:
    """N=4 reconstruction target: optional forced HDF5 key, else pred-preferring lookup."""
    forced = getattr(args, "hfss_supervision_n4", None)
    if forced:
        if not _h5_has_dataset(H5_2TO4, forced):
            raise KeyError(f"N=4 supervision key {forced!r} not in {H5_2TO4}")
        print(f"[bootstrap] N=4 supervision key: {forced!r} (forced)", flush=True)
        return str(forced)
    return pick_hfss_supervision_key(H5_2TO4, "hfss_pred_4x4", "hfss_4x4")


def _loader_kwargs(num_workers: int) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return kw


def _maybe_compile_gan(G: nn.Module, D: nn.Module, enabled: bool, tag: str) -> tuple:
    """``torch.compile`` defaults to Inductor, which needs Triton (often missing on Windows)."""
    if not enabled:
        return G, D
    if sys.platform.startswith("win"):
        print(
            f"[{tag}] torch.compile skipped: torch.inductor needs Triton, which is not "
            f"available on typical Windows PyTorch builds (train eager or use Linux/WSL).",
            flush=True,
        )
        return G, D

    G_orig, D_orig = G, D
    try:
        Gc = torch.compile(G, mode="default", fullgraph=False)  # type: ignore[assignment]
        Dc = torch.compile(D, mode="default", fullgraph=False)  # type: ignore[assignment]
        B, H, W = 2, 181, 360
        x = torch.zeros(B, 5, H, W, device=DEVICE, dtype=torch.float32)
        y = torch.zeros(B, 1, H, W, device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            with torch.amp.autocast(
                device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP
            ):
                _ = Gc(x)
                _ = Dc(x, y)
        print(f"[{tag}] torch.compile(G, D) mode=default (inductor warmup ok)", flush=True)
        return Gc, Dc
    except Exception as e:
        print(f"[{tag}] torch.compile skipped (eager fallback): {e}", flush=True)
        return G_orig, D_orig


def _adam_maybe_fused(params, lr: float, fused: bool) -> optim.Optimizer:
    kwargs = dict(lr=lr, betas=(0.5, 0.999))
    if fused and DEVICE.type == "cuda":
        try:
            return optim.Adam(params, **kwargs, fused=True)  # type: ignore[call-arg]
        except TypeError:
            pass
    return optim.Adam(params, **kwargs)


class BootstrapScaleDataset(Dataset):
    """Same contract as ``ScaleDatasetWithAnchor``; ``anchor_key=None`` zeros ch4 (N=2)."""

    def __init__(
        self,
        h5_path: Path,
        indices,
        scale_N: int,
        matlab_key: str,
        hfss_key: str,
        anchor_key: Optional[str],
        mean_mat: np.ndarray,
        std_mat: np.ndarray,
        augment_noise: bool = False,
        noise_std: float = NOISE_STD,
        anchor_dropout_p: float = 0.0,
    ):
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
        mat = f[self.matlab_key][idx].astype(np.float32)
        hfss = f[self.hfss_key][idx].astype(np.float32)
        if self.anchor_key is None:
            anchor = np.zeros_like(mat, dtype=np.float32)
        else:
            anchor = f[self.anchor_key][idx].astype(np.float32)
        meta = f["metadata"][idx].astype(np.float32)

        mat_n = (mat - self.mean_mat) / self.std_mat
        anchor_n = (anchor - self.mean_mat) / self.std_mat
        dphase_x = np.full_like(mat_n, fill_value=meta[0] / 180.0, dtype=np.float32)
        dphase_y = np.full_like(mat_n, fill_value=meta[1] / 180.0, dtype=np.float32)
        scale_tok = np.full_like(mat_n, fill_value=self.scale_N / 16.0, dtype=np.float32)

        x = np.stack([mat_n, dphase_x, dphase_y, scale_tok, anchor_n], axis=0)

        if self.augment_noise and self.noise_std > 0:
            x[0] += np.random.normal(0, self.noise_std, size=mat_n.shape).astype(np.float32)
            x[4] += np.random.normal(0, self.noise_std, size=mat_n.shape).astype(np.float32)

        if self.anchor_dropout_p > 0.0 and np.random.rand() < self.anchor_dropout_p:
            x[4] = 0.0

        target_max = float(hfss.max())
        null_mask = (hfss < (target_max + NULL_THRESHOLD_DB)).astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(hfss[None]),
            torch.from_numpy(mat[None]),
            torch.from_numpy(null_mask[None]),
        )


class PhaseATrainOGMatchedEpochDataset(Dataset):
    """Phase A train set with the same **sample count per epoch** as OG multiscale.

    OG uses ``len(tr4) + len(tr8)`` train rows (N=4 on 2to4 + N=8 on 4to8). Here:
    - indices ``[0, len(tr4))``: N=4, one step per 2to4 train beta;
    - indices ``[len(tr4), len(tr4)+len(tr8))``: N=2; each epoch uses a contiguous
      block of ``len(tr8)`` betas from ``tr4``, rotated by
      ``(epoch-1)*len(tr8) mod len(tr4)`` so all betas receive N=2 updates over
      many epochs (requires ``len(tr8) <= len(tr4)``, which holds for project splits).

    Call ``set_epoch(epoch)`` (1-based) before each training epoch when using a
    shuffled DataLoader so the N=2 block aligns with the training epoch number.
    """

    def __init__(
        self,
        tr4_indices,
        tr8_len: int,
        hfss_key_n4: str,
        n2_key: str,
        mean_mat: np.ndarray,
        std_mat: np.ndarray,
        augment_noise: bool = True,
        noise_std: float = NOISE_STD,
        anchor_dropout_p: float = 0.0,
        anchor_key_n4: str = "hfss_pred_2x2",
    ):
        self.tr4 = np.asarray(tr4_indices, dtype=np.int64)
        self.tr8_len = int(tr8_len)
        if self.tr8_len > len(self.tr4):
            raise ValueError(
                f"Phase A OG-matched epoch needs len(tr8)={self.tr8_len} <= len(tr4)={len(self.tr4)}"
            )
        self.hfss_key_n4 = hfss_key_n4
        self.anchor_key_n4 = str(anchor_key_n4)
        self.n2_key = n2_key
        self.mean_mat = mean_mat.astype(np.float32)
        self.std_mat = np.maximum(std_mat.astype(np.float32), 1e-6)
        self.augment_noise = augment_noise
        self.noise_std = noise_std
        self.anchor_dropout_p = float(anchor_dropout_p)
        self.h5_path = str(H5_2TO4)
        self._file = None
        self._epoch = 1

    def set_epoch(self, epoch_1based: int) -> None:
        self._epoch = max(1, int(epoch_1based))

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.tr4) + self.tr8_len

    def _pack(
        self,
        mat: np.ndarray,
        hfss: np.ndarray,
        anchor: np.ndarray,
        meta: np.ndarray,
        scale_N: int,
    ):
        mat_n = (mat - self.mean_mat) / self.std_mat
        anchor_n = (anchor - self.mean_mat) / self.std_mat
        dphase_x = np.full_like(mat_n, fill_value=meta[0] / 180.0, dtype=np.float32)
        dphase_y = np.full_like(mat_n, fill_value=meta[1] / 180.0, dtype=np.float32)
        scale_tok = np.full_like(mat_n, fill_value=scale_N / 16.0, dtype=np.float32)
        x = np.stack([mat_n, dphase_x, dphase_y, scale_tok, anchor_n], axis=0)

        if self.augment_noise and self.noise_std > 0:
            x[0] += np.random.normal(0, self.noise_std, size=mat_n.shape).astype(np.float32)
            x[4] += np.random.normal(0, self.noise_std, size=mat_n.shape).astype(np.float32)

        if self.anchor_dropout_p > 0.0 and np.random.rand() < self.anchor_dropout_p:
            x[4] = 0.0

        target_max = float(hfss.max())
        null_mask = (hfss < (target_max + NULL_THRESHOLD_DB)).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(hfss[None]),
            torch.from_numpy(mat[None]),
            torch.from_numpy(null_mask[None]),
        )

    def __getitem__(self, i: int):
        f = self._get_file()
        n4 = len(self.tr4)
        if i < n4:
            row = int(self.tr4[i])
            mat = f["matlab_4x4"][row].astype(np.float32)
            hfss = f[self.hfss_key_n4][row].astype(np.float32)
            anchor = f[self.anchor_key_n4][row].astype(np.float32)
            meta = f["metadata"][row].astype(np.float32)
            return self._pack(mat, hfss, anchor, meta, 4)

        j = i - n4
        off = ((self._epoch - 1) * self.tr8_len) % n4
        row = int(self.tr4[(off + j) % n4])
        mat = f["matlab_2x2"][row].astype(np.float32)
        hfss = f[self.n2_key][row].astype(np.float32)
        anchor = np.zeros_like(mat, dtype=np.float32)
        meta = f["metadata"][row].astype(np.float32)
        return self._pack(mat, hfss, anchor, meta, 2)


class Scale8PseudoTargetDataset(Dataset):
    """N=8 sample with external pseudo HFSS target (Phase C)."""

    def __init__(
        self,
        h5_path: Path,
        indices,
        pseudo_index: np.ndarray,
        pseudo_hfss: np.ndarray,
        mean_mat: np.ndarray,
        std_mat: np.ndarray,
        augment_noise: bool = False,
        noise_std: float = NOISE_STD,
        anchor_dropout_p: float = 0.0,
    ):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean_mat = mean_mat.astype(np.float32)
        self.std_mat = np.maximum(std_mat.astype(np.float32), 1e-6)
        self.augment_noise = augment_noise
        self.noise_std = noise_std
        self.anchor_dropout_p = float(anchor_dropout_p)
        self._file = None

        pidx = np.asarray(pseudo_index, dtype=np.int64)
        ph = pseudo_hfss.astype(np.float32)
        if pidx.shape[0] != ph.shape[0]:
            raise ValueError("pseudo_index and pseudo_hfss length mismatch")
        self._lookup = {int(pi): ph[i] for i, pi in enumerate(pidx)}
        for idx in self.indices:
            if int(idx) not in self._lookup:
                raise KeyError(f"pseudo 8x8 missing for 4to8 index {int(idx)}")

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        f = self._get_file()
        idx = int(self.indices[i])
        mat = f["matlab_8x8"][idx].astype(np.float32)
        anchor = f["hfss_pred_4x4"][idx].astype(np.float32)
        hfss = self._lookup[idx]
        meta = f["metadata"][idx].astype(np.float32)

        mat_n = (mat - self.mean_mat) / self.std_mat
        anchor_n = (anchor - self.mean_mat) / self.std_mat
        dphase_x = np.full_like(mat_n, fill_value=meta[0] / 180.0, dtype=np.float32)
        dphase_y = np.full_like(mat_n, fill_value=meta[1] / 180.0, dtype=np.float32)
        scale_tok = np.full_like(mat_n, fill_value=8.0 / 16.0, dtype=np.float32)

        x = np.stack([mat_n, dphase_x, dphase_y, scale_tok, anchor_n], axis=0)

        if self.augment_noise and self.noise_std > 0:
            x[0] += np.random.normal(0, self.noise_std, size=mat_n.shape).astype(np.float32)
            x[4] += np.random.normal(0, self.noise_std, size=mat_n.shape).astype(np.float32)

        if self.anchor_dropout_p > 0.0 and np.random.rand() < self.anchor_dropout_p:
            x[4] = 0.0

        target_max = float(hfss.max())
        null_mask = (hfss < (target_max + NULL_THRESHOLD_DB)).astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(hfss[None]),
            torch.from_numpy(mat[None]),
            torch.from_numpy(null_mask[None]),
        )


def _load_norm_and_splits():
    for p in (H5_2TO4, H5_4TO8, H5_16X16, NORM_COMBINED, M16_TEST, SPLITS_2TO4, SPLITS_4TO8):
        if not p.exists():
            raise FileNotFoundError(p)
    s = np.load(NORM_COMBINED)
    mean_mat = s["mean"]
    std_mat = s["std"]
    sigma_res = float(s["residual_std"])

    sp24 = np.load(SPLITS_2TO4)
    sp48 = np.load(SPLITS_4TO8)
    tr4 = np.sort(sp24["train"].astype(np.int64))
    va4 = np.sort(sp24["val"].astype(np.int64))
    tr8 = np.sort(sp48["train"].astype(np.int64))
    va8 = np.sort(sp48["val"].astype(np.int64))
    test48 = np.sort(sp48["test"].astype(np.int64))
    val16_idx = test48
    return mean_mat, std_mat, sigma_res, tr4, va4, tr8, va8, test48, val16_idx


def cmd_phase_a(args: argparse.Namespace) -> None:
    _require_cuda()
    ckpt_a = Path(args.ckpt_dir_phase_a) if getattr(args, "ckpt_dir_phase_a", None) else CKPT_PHASE_A
    ckpt_a.mkdir(parents=True, exist_ok=True)

    batch = getattr(args, "batch_size", None) or BATCH
    num_workers = NUM_WORKERS if getattr(args, "num_workers", None) is None else int(args.num_workers)
    validate_every = int(getattr(args, "validate_every", BOOTSTRAP_VALIDATE_EVERY))
    ld_kw = _loader_kwargs(num_workers)

    mean_mat, std_mat, sigma_res, tr4, va4, tr8, va8, test48, val16_idx = _load_norm_and_splits()
    print(
        f"[phase_a] sigma_residual={sigma_res:.3f} dB  ckpt_dir={ckpt_a}",
        flush=True,
    )

    hfss_key_n4 = _resolve_n4_supervision_key(args)
    anchor_key_n4 = str(getattr(args, "anchor_key_n4", "hfss_pred_2x2"))
    if not _h5_has_dataset(H5_2TO4, anchor_key_n4):
        raise KeyError(f"{H5_2TO4} missing anchor dataset {anchor_key_n4!r} for N=4 ch4")

    use_n2 = not args.skip_n2
    n2_key = args.n2_hfss_key
    if use_n2 and not _h5_has_dataset(H5_2TO4, n2_key):
        raise KeyError(
            f"{H5_2TO4} has no dataset {n2_key!r}. Expected keys include matlab_2x2, "
            f"hfss_pred_2x2. Use --skip-n2 to omit N=2."
        )

    parts_va = [
        ScaleDatasetWithAnchor(
            H5_2TO4,
            va4,
            4,
            "matlab_4x4",
            hfss_key_n4,
            anchor_key=anchor_key_n4,
            mean_mat=mean_mat,
            std_mat=std_mat,
            augment_noise=False,
        )
    ]
    if use_n2:
        parts_va.append(
            BootstrapScaleDataset(
                H5_2TO4,
                va4,
                2,
                "matlab_2x2",
                n2_key,
                anchor_key=None,
                mean_mat=mean_mat,
                std_mat=std_mat,
                augment_noise=False,
            )
        )

    if use_n2 and args.og_sized_epochs:
        train_ds: Dataset = PhaseATrainOGMatchedEpochDataset(
            tr4,
            len(tr8),
            hfss_key_n4,
            n2_key,
            mean_mat=mean_mat,
            std_mat=std_mat,
            augment_noise=True,
            anchor_dropout_p=ANCHOR_DROPOUT_P,
            anchor_key_n4=anchor_key_n4,
        )
        train_layout = "compact: len(tr4)+len(tr8) per epoch, rotating N=2 (--og-sized-epochs)"
    else:
        parts_tr: list[Dataset] = [
            ScaleDatasetWithAnchor(
                H5_2TO4,
                tr4,
                4,
                "matlab_4x4",
                hfss_key_n4,
                anchor_key=anchor_key_n4,
                mean_mat=mean_mat,
                std_mat=std_mat,
                augment_noise=True,
                anchor_dropout_p=ANCHOR_DROPOUT_P,
            )
        ]
        if use_n2:
            parts_tr.append(
                BootstrapScaleDataset(
                    H5_2TO4,
                    tr4,
                    2,
                    "matlab_2x2",
                    n2_key,
                    anchor_key=None,
                    mean_mat=mean_mat,
                    std_mat=std_mat,
                    augment_noise=True,
                    anchor_dropout_p=ANCHOR_DROPOUT_P,
                )
            )
        train_ds = ConcatDataset(parts_tr) if len(parts_tr) > 1 else parts_tr[0]
        train_layout = (
            "2x2 + 4x4: full N=4 + full N=2 on tr4 each epoch"
            if use_n2
            else "4x4 on tr4 only"
        )

    n_train = len(train_ds)
    n_steps = n_train // batch
    print(
        f"[phase_a] train: {n_train} samples/ep, {n_steps} steps "
        f"(batch={batch}, drop_last) | {train_layout} | num_workers={num_workers}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        **ld_kw,
    )
    va_loader = DataLoader(
        ConcatDataset(parts_va),
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        **ld_kw,
    )

    m16_pack = np.load(M16_TEST)
    m16_test_idx = m16_pack["test_idx"]
    m16_arr = m16_pack["arr"].astype(np.float32)
    if not np.array_equal(m16_test_idx, test48):
        raise RuntimeError("matlab_16x16_test.npz test_idx mismatch with split_indices_4to8 test")
    if len(val16_idx) != len(m16_arr):
        raise RuntimeError(
            f"val_16x16 len {len(val16_idx)} != matlab_16x16 arr len {len(m16_arr)}"
        )
    val16_ds = Val16x16Dataset(val16_idx, m16_arr, mean_mat, std_mat)
    val16_loader = DataLoader(
        val16_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        **ld_kw,
    )

    G = EnhancedResUNetGenerator(in_ch=5, out_ch=1, base=GEN_BASE, attn_heads=ATTN_HEADS).to(DEVICE)
    D = PatchDiscriminator(in_channels=6, base=64).to(DEVICE)

    last_g = ckpt_a / "last_generator.pt"
    last_g_ema = ckpt_a / "last_generator_ema.pt"
    last_d = ckpt_a / "last_discriminator.pt"
    if last_g.exists():
        G.load_state_dict(torch.load(last_g, map_location=DEVICE))
        print(f"[phase_a] resumed G from {last_g.name}", flush=True)
    if last_d.exists():
        D.load_state_dict(torch.load(last_d, map_location=DEVICE))
        print(f"[phase_a] resumed D from {last_d.name}", flush=True)

    G, D = _maybe_compile_gan(G, D, bool(getattr(args, "compile", False)), "phase_a")

    fused = bool(getattr(args, "fused_adam", False))
    opt_g = _adam_maybe_fused(G.parameters(), LR_G, fused)
    opt_d = _adam_maybe_fused(D.parameters(), LR_D, fused)
    sched_g = optim.lr_scheduler.ReduceLROnPlateau(
        opt_g, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR
    )
    sched_d = optim.lr_scheduler.ReduceLROnPlateau(
        opt_d, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR
    )
    scaler_g = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_GRAD_SCALER)
    scaler_d = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_GRAD_SCALER)
    bce = nn.BCEWithLogitsLoss()
    recon_loss = ResidualReconLoss(sigma_res=sigma_res).to(DEVICE)
    ema_decay = float(args.ema_decay) if getattr(args, "ema_decay", None) is not None else EMA_DECAY
    ema = EMA(G, decay=ema_decay)
    if last_g_ema.exists():
        ema.shadow = torch.load(last_g_ema, map_location=DEVICE)
        print(f"[phase_a] resumed EMA from {last_g_ema.name}", flush=True)

    epochs = args.epochs if args.epochs is not None else EPOCHS
    n_tag = "N=2+4" if use_n2 else "N=4"
    print(
        f"[phase_a] train {n_tag} only (no 8x8 supervision); "
        f"val_indist {n_tag}; val_16x16 selection",
        flush=True,
    )

    best_rmse_16 = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    if last_g_ema.exists():
        seed = evaluate_16x16(G, ema, val16_loader, recon_loss)
        best_rmse_16 = seed["rmse"]
        print(f"[phase_a] seeded best_rmse_16 from EMA: {best_rmse_16:.5f}", flush=True)

    t0 = time.time()
    epoch_times = []
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        cur_lr_g = opt_g.param_groups[0]["lr"]
        print(f"\n=== [phase_a] Epoch {epoch}/{epochs}  (G_lr={cur_lr_g:.2e}) ===", flush=True)

        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)

        train_t0 = time.time()
        g_avg, d_avg, l1_avg, ssim_avg = train_one_epoch(
            G, D, ema, train_loader, opt_g, opt_d, scaler_g, scaler_d, bce, recon_loss
        )
        train_dt = time.time() - train_t0
        print(
            f"  train: g={g_avg:.4f} d={d_avg:.4f} l1_dB={l1_avg:.3f} ssim={ssim_avg:.4f} "
            f"[{train_dt/60:.2f} min]",
            flush=True,
        )

        eval_dt = 0.0
        if epoch % validate_every == 0 or epoch == 1 or epoch == epochs:
            eval_t0 = time.time()
            m_in = evaluate_indist(G, ema, va_loader, recon_loss)
            print(
                f"  val_indist ({n_tag}, EMA): rmse={m_in['rmse']:.4f} mae={m_in['mae']:.4f} "
                f"r={m_in['pearson']:.4f}",
                flush=True,
            )
            m_16 = evaluate_16x16(G, ema, val16_loader, recon_loss)
            print(
                f"  val_16x16 (EMA, no grad): rmse={m_16['rmse']:.4f} mae={m_16['mae']:.4f} "
                f"r={m_16['pearson']:.4f}",
                flush=True,
            )
            eval_dt = time.time() - eval_t0

            if m_16["rmse"] < best_rmse_16 - 1e-5:
                best_rmse_16 = m_16["rmse"]
                best_epoch = epoch
                epochs_without_improve = 0
                torch.save(ema.state_dict(), ckpt_a / "best_generator.pt")
                print(f"  * new best val_16x16 rmse {best_rmse_16:.5f} -> best_generator.pt", flush=True)
            else:
                epochs_without_improve += 1
                print(
                    f"  (no val_16x16 improvement for {epochs_without_improve} val rounds)",
                    flush=True,
                )
            sched_g.step(m_16["rmse"])
            sched_d.step(m_16["rmse"])

        torch.save(G.state_dict(), ckpt_a / "last_generator.pt")
        torch.save(ema.state_dict(), ckpt_a / "last_generator_ema.pt")
        torch.save(D.state_dict(), ckpt_a / "last_discriminator.pt")
        if epoch % SAVE_EVERY == 0:
            torch.save(ema.state_dict(), ckpt_a / f"generator_ema_epoch_{epoch:03d}.pt")

        epoch_dt = time.time() - epoch_t0
        epoch_times.append(epoch_dt)
        recent = epoch_times[-10:]
        avg_recent = sum(recent) / len(recent)
        remaining = max(0, epochs - epoch)
        eta_min = remaining * avg_recent / 60
        print(
            f"  [time] epoch={epoch_dt/60:.2f} min elapsed={(time.time()-t0)/60:.1f} min "
            f"eta={eta_min:.0f} min",
            flush=True,
        )

        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(
                f"\n[phase_a] early stop epoch {epoch}; best {best_epoch} "
                f"(val_16x16 rmse {best_rmse_16:.5f})",
                flush=True,
            )
            break

    print(
        f"\n[phase_a] done in {(time.time()-t0)/60:.1f} min. best epoch {best_epoch} "
        f"val_16x16 rmse {best_rmse_16:.5f}. Next: export_pseudo",
        flush=True,
    )


@torch.no_grad()
def cmd_export_pseudo(args: argparse.Namespace) -> None:
    _require_cuda()
    gen_path = Path(args.generator)
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)

    mean_mat, std_mat, sigma_res, tr4, va4, tr8, va8, test48, _val16 = _load_norm_and_splits()
    recon_loss = ResidualReconLoss(sigma_res=sigma_res).to(DEVICE)

    if args.split == "train":
        id_list = tr8
    elif args.split == "val":
        id_list = va8
    elif args.split == "trainval":
        id_list = np.sort(np.concatenate([tr8, va8]))
    elif args.split == "all":
        id_list = np.sort(np.concatenate([tr8, va8, test48]))
    else:
        raise ValueError(args.split)

    G = EnhancedResUNetGenerator(in_ch=5, out_ch=1, base=GEN_BASE, attn_heads=ATTN_HEADS).to(DEVICE)
    state = torch.load(gen_path, map_location=DEVICE)
    G.load_state_dict(state)
    G.eval()
    print(f"[export_pseudo] loaded {gen_path}  samples={len(id_list)}  split={args.split}", flush=True)

    out_idx = []
    out_y = []
    bs = args.batch_size

    with h5py.File(H5_4TO8, "r") as f:
        for start in range(0, len(id_list), bs):
            chunk = id_list[start : start + bs]
            xs = []
            mats = []
            for idx in chunk:
                idx = int(idx)
                mat = f["matlab_8x8"][idx].astype(np.float32)
                anchor = f["hfss_pred_4x4"][idx].astype(np.float32)
                meta = f["metadata"][idx].astype(np.float32)
                mat_n = (mat - mean_mat.astype(np.float32)) / np.maximum(std_mat.astype(np.float32), 1e-6)
                anchor_n = (anchor - mean_mat.astype(np.float32)) / np.maximum(
                    std_mat.astype(np.float32), 1e-6
                )
                dpx = np.full_like(mat_n, fill_value=meta[0] / 180.0, dtype=np.float32)
                dpy = np.full_like(mat_n, fill_value=meta[1] / 180.0, dtype=np.float32)
                st = np.full_like(mat_n, fill_value=8.0 / 16.0, dtype=np.float32)
                x = np.stack([mat_n, dpx, dpy, st, anchor_n], axis=0)
                xs.append(x)
                mats.append(mat[None])

            xb = torch.from_numpy(np.stack(xs, axis=0)).to(DEVICE)
            mb = torch.from_numpy(np.stack(mats, axis=0)).to(DEVICE)
            with torch.amp.autocast(device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP):
                delta_n = G(xb)
                pred = recon_loss.compose(delta_n, mb)
            pred_np = pred.cpu().float().numpy()
            for i, idx in enumerate(chunk):
                out_idx.append(int(idx))
                out_y.append(pred_np[i, 0].astype(np.float32))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        index=np.asarray(out_idx, dtype=np.int64),
        hfss_pseudo_8x8=np.stack(out_y, axis=0),
        source_generator=str(gen_path),
        split=args.split,
    )
    print(f"[export_pseudo] wrote {out_path}  shape={np.stack(out_y, axis=0).shape}", flush=True)


def cmd_phase_c(args: argparse.Namespace) -> None:
    _require_cuda()
    ckpt_c = Path(args.ckpt_dir_phase_c) if getattr(args, "ckpt_dir_phase_c", None) else CKPT_PHASE_C
    ckpt_c.mkdir(parents=True, exist_ok=True)

    batch = getattr(args, "batch_size", None) or BATCH
    num_workers = NUM_WORKERS if getattr(args, "num_workers", None) is None else int(args.num_workers)
    validate_every = int(getattr(args, "validate_every", BOOTSTRAP_VALIDATE_EVERY))
    ld_kw = _loader_kwargs(num_workers)

    pseudo_path = Path(args.pseudo)
    if not pseudo_path.exists():
        raise FileNotFoundError(pseudo_path)
    pack = np.load(pseudo_path)
    pidx = pack["index"]
    phf = pack["hfss_pseudo_8x8"]

    mean_mat, std_mat, sigma_res, tr4, va4, tr8, va8, test48, val16_idx = _load_norm_and_splits()
    hfss_key_n4 = _resolve_n4_supervision_key(args)
    anchor_key_n4 = str(getattr(args, "anchor_key_n4", "hfss_pred_2x2"))
    if not _h5_has_dataset(H5_2TO4, anchor_key_n4):
        raise KeyError(f"{H5_2TO4} missing anchor dataset {anchor_key_n4!r} for N=4 ch4")

    tr4_ds = ScaleDatasetWithAnchor(
        H5_2TO4,
        tr4,
        4,
        "matlab_4x4",
        hfss_key_n4,
        anchor_key=anchor_key_n4,
        mean_mat=mean_mat,
        std_mat=std_mat,
        augment_noise=True,
        anchor_dropout_p=ANCHOR_DROPOUT_P,
    )
    tr8_ds = Scale8PseudoTargetDataset(
        H5_4TO8,
        tr8,
        pidx,
        phf,
        mean_mat=mean_mat,
        std_mat=std_mat,
        augment_noise=True,
        anchor_dropout_p=ANCHOR_DROPOUT_P,
    )
    train_ds = ConcatDataset([tr4_ds, tr8_ds])

    va4_ds = ScaleDatasetWithAnchor(
        H5_2TO4,
        va4,
        4,
        "matlab_4x4",
        hfss_key_n4,
        anchor_key=anchor_key_n4,
        mean_mat=mean_mat,
        std_mat=std_mat,
        augment_noise=False,
    )
    va8_ds = Scale8PseudoTargetDataset(
        H5_4TO8,
        va8,
        pidx,
        phf,
        mean_mat=mean_mat,
        std_mat=std_mat,
        augment_noise=False,
    )
    va_ds = ConcatDataset([va4_ds, va8_ds])

    n_train = len(train_ds)
    n_steps = n_train // batch
    print(
        f"[phase_c] train: {n_train} samples/ep ({len(tr4)} N=4 + {len(tr8)} N=8 pseudo), "
        f"{n_steps} steps (batch={batch}, drop_last) | ckpt_dir={ckpt_c} | num_workers={num_workers}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        **ld_kw,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        **ld_kw,
    )

    m16_pack = np.load(M16_TEST)
    if not np.array_equal(m16_pack["test_idx"], test48):
        raise RuntimeError("matlab_16x16_test.npz test_idx mismatch")
    m16_arr = m16_pack["arr"].astype(np.float32)
    if len(val16_idx) != len(m16_arr):
        raise RuntimeError(
            f"val_16x16 len {len(val16_idx)} != matlab_16x16 arr len {len(m16_arr)}"
        )
    val16_ds = Val16x16Dataset(val16_idx, m16_arr, mean_mat, std_mat)
    val16_loader = DataLoader(
        val16_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        **ld_kw,
    )

    G = EnhancedResUNetGenerator(in_ch=5, out_ch=1, base=GEN_BASE, attn_heads=ATTN_HEADS).to(DEVICE)
    D = PatchDiscriminator(in_channels=6, base=64).to(DEVICE)

    init_g = Path(args.init_generator)
    if not init_g.exists():
        raise FileNotFoundError(init_g)

    last_g = ckpt_c / "last_generator.pt"
    last_g_ema = ckpt_c / "last_generator_ema.pt"
    last_d = ckpt_c / "last_discriminator.pt"
    if last_g.exists():
        G.load_state_dict(torch.load(last_g, map_location=DEVICE))
        print(f"[phase_c] resumed G from {last_g.name}", flush=True)
    else:
        G.load_state_dict(torch.load(init_g, map_location=DEVICE))
        print(
            f"[phase_c] init G from {init_g} (8x8 targets from {pseudo_path.name}; "
            f"ckpt_dir={ckpt_c})",
            flush=True,
        )
    if last_d.exists():
        D.load_state_dict(torch.load(last_d, map_location=DEVICE))
        print(f"[phase_c] resumed D from {last_d.name}", flush=True)

    G, D = _maybe_compile_gan(G, D, bool(getattr(args, "compile", False)), "phase_c")

    fused = bool(getattr(args, "fused_adam", False))
    opt_g = _adam_maybe_fused(G.parameters(), LR_G, fused)
    opt_d = _adam_maybe_fused(D.parameters(), LR_D, fused)
    sched_g = optim.lr_scheduler.ReduceLROnPlateau(
        opt_g, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR
    )
    sched_d = optim.lr_scheduler.ReduceLROnPlateau(
        opt_d, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR
    )
    scaler_g = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_GRAD_SCALER)
    scaler_d = torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=USE_GRAD_SCALER)
    bce = nn.BCEWithLogitsLoss()
    recon_loss = ResidualReconLoss(sigma_res=sigma_res).to(DEVICE)
    ema_decay = float(args.ema_decay) if getattr(args, "ema_decay", None) is not None else EMA_DECAY
    ema = EMA(G, decay=ema_decay)
    if last_g_ema.exists():
        ema.shadow = torch.load(last_g_ema, map_location=DEVICE)
        print(f"[phase_c] resumed EMA from {last_g_ema.name}", flush=True)

    epochs = args.epochs if args.epochs is not None else EPOCHS

    best_rmse_16 = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    if last_g_ema.exists():
        seed = evaluate_16x16(G, ema, val16_loader, recon_loss)
        best_rmse_16 = seed["rmse"]
        print(f"[phase_c] seeded best_rmse_16: {best_rmse_16:.5f}", flush=True)

    t0 = time.time()
    epoch_times = []
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        cur_lr_g = opt_g.param_groups[0]["lr"]
        print(f"\n=== [phase_c] Epoch {epoch}/{epochs}  (G_lr={cur_lr_g:.2e}) ===", flush=True)

        train_t0 = time.time()
        g_avg, d_avg, l1_avg, ssim_avg = train_one_epoch(
            G, D, ema, train_loader, opt_g, opt_d, scaler_g, scaler_d, bce, recon_loss
        )
        train_dt = time.time() - train_t0
        print(
            f"  train (N=4 + N=8 pseudo): g={g_avg:.4f} d={d_avg:.4f} l1_dB={l1_avg:.3f} "
            f"ssim={ssim_avg:.4f} [{train_dt/60:.2f} min]",
            flush=True,
        )

        if epoch % validate_every == 0 or epoch == 1 or epoch == epochs:
            m_in = evaluate_indist(G, ema, va_loader, recon_loss)
            print(
                f"  val_indist (N=4+8; 8x8 vs pseudo, EMA): rmse={m_in['rmse']:.4f} "
                f"mae={m_in['mae']:.4f} r={m_in['pearson']:.4f}",
                flush=True,
            )
            m_16 = evaluate_16x16(G, ema, val16_loader, recon_loss)
            print(
                f"  val_16x16 (EMA): rmse={m_16['rmse']:.4f} mae={m_16['mae']:.4f} "
                f"r={m_16['pearson']:.4f}",
                flush=True,
            )

            if m_16["rmse"] < best_rmse_16 - 1e-5:
                best_rmse_16 = m_16["rmse"]
                best_epoch = epoch
                epochs_without_improve = 0
                torch.save(ema.state_dict(), ckpt_c / "best_generator.pt")
                print(f"  * new best val_16x16 rmse {best_rmse_16:.5f} -> best_generator.pt", flush=True)
            else:
                epochs_without_improve += 1
            sched_g.step(m_16["rmse"])
            sched_d.step(m_16["rmse"])

        torch.save(G.state_dict(), ckpt_c / "last_generator.pt")
        torch.save(ema.state_dict(), ckpt_c / "last_generator_ema.pt")
        torch.save(D.state_dict(), ckpt_c / "last_discriminator.pt")
        if epoch % SAVE_EVERY == 0:
            torch.save(ema.state_dict(), ckpt_c / f"generator_ema_epoch_{epoch:03d}.pt")

        epoch_dt = time.time() - epoch_t0
        epoch_times.append(epoch_dt)
        recent = epoch_times[-10:]
        avg_recent = sum(recent) / len(recent)
        remaining = max(0, epochs - epoch)
        print(
            f"  [time] epoch={epoch_dt/60:.2f} min eta={remaining * avg_recent / 60:.0f} min",
            flush=True,
        )

        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(
                f"\n[phase_c] early stop epoch {epoch}; best {best_epoch} "
                f"(val_16x16 rmse {best_rmse_16:.5f})",
                flush=True,
            )
            break

    print(
        f"\n[phase_c] done in {(time.time()-t0)/60:.1f} min. best {best_epoch} "
        f"val_16x16 rmse {best_rmse_16:.5f}",
        flush=True,
    )


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ap = argparse.ArgumentParser(description="Bootstrap residual multiscale trainer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("phase_a", help="Phase A: N=2,N=4 only (optional pred- or HFSS-ground-truth targets)")
    pa.add_argument("--n2-hfss-key", default="hfss_pred_2x2", help="HDF5 key for N=2 supervision target")
    pa.add_argument("--skip-n2", action="store_true", help="Only N=4 if 2x2 HFSS unavailable")
    pa.add_argument(
        "--og-sized-epochs",
        action="store_true",
        help="Phase A: len(tr4)+len(tr8) samples/ep with rotating N=2 (fewer steps than full 2x2+4x4)",
    )
    pa.add_argument("--epochs", type=int, default=None, help=f"default {EPOCHS}")
    pa.add_argument(
        "--hfss-supervision-n4",
        type=str,
        default=None,
        help="Force N=4 reconstruction target key (e.g. hfss_4x4). Default: prefer hfss_pred_4x4 when present.",
    )
    pa.add_argument(
        "--anchor-key-n4",
        type=str,
        default="hfss_pred_2x2",
        help="ch4 anchor at N=4 (HDF5 key on 2to4 rows)",
    )
    pa.add_argument(
        "--ckpt-dir-phase-a",
        type=str,
        default=None,
        help=f"Checkpoint directory for phase A (default {CKPT_PHASE_A})",
    )
    pa.add_argument("--num-workers", type=int, default=None, help=f"DataLoader workers (default {NUM_WORKERS})")
    pa.add_argument("--batch-size", type=int, default=None, help=f"Batch size (default {BATCH})")
    pa.add_argument(
        "--validate-every",
        type=int,
        default=BOOTSTRAP_VALIDATE_EVERY,
        help="Run in-dist + val_16x16 every N epochs",
    )
    pa.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(G,D) after load (PyTorch 2+; first epoch may be slow compiling)",
    )
    pa.add_argument("--fused-adam", action="store_true", help="Use fused CUDA Adam when supported")
    pa.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help=f"EMA decay (default {EMA_DECAY})",
    )
    pa.set_defaults(func=cmd_phase_a)

    ex = sub.add_parser("export_pseudo", help="Phase A EMA -> pseudo hfss_8x8 labels")
    ex.add_argument("--generator", type=str, default=str(CKPT_PHASE_A / "best_generator.pt"))
    ex.add_argument("--output", type=str, default=str(DEFAULT_PSEUDO_NPZ))
    ex.add_argument(
        "--split",
        choices=["train", "val", "trainval", "all"],
        default="trainval",
        help="Which 4to8 indices get pseudo labels (phase_c needs train+val)",
    )
    ex.add_argument("--batch-size", type=int, default=BATCH)
    ex.set_defaults(func=cmd_export_pseudo)

    pc = sub.add_parser(
        "phase_c",
        help="N=4+N=8 like residual multiscale; 8x8 uses pseudo targets from export_pseudo",
    )
    pc.add_argument("--pseudo", type=str, default=str(DEFAULT_PSEUDO_NPZ))
    pc.add_argument(
        "--init-generator",
        type=str,
        default=str(CKPT_PHASE_A / "best_generator.pt"),
        help="Weights before optional resume from phase_c last_*",
    )
    pc.add_argument("--epochs", type=int, default=None)
    pc.add_argument(
        "--hfss-supervision-n4",
        type=str,
        default=None,
        help="Force N=4 reconstruction target key (e.g. hfss_4x4). Default: prefer hfss_pred_4x4 when present.",
    )
    pc.add_argument(
        "--anchor-key-n4",
        type=str,
        default="hfss_pred_2x2",
        help="ch4 anchor at N=4 (HDF5 key on 2to4 rows)",
    )
    pc.add_argument(
        "--ckpt-dir-phase-c",
        type=str,
        default=None,
        help=f"Checkpoint directory for phase C (default {CKPT_PHASE_C})",
    )
    pc.add_argument("--num-workers", type=int, default=None, help=f"DataLoader workers (default {NUM_WORKERS})")
    pc.add_argument("--batch-size", type=int, default=None, help=f"Batch size (default {BATCH})")
    pc.add_argument(
        "--validate-every",
        type=int,
        default=BOOTSTRAP_VALIDATE_EVERY,
        help="Run in-dist + val_16x16 every N epochs",
    )
    pc.add_argument("--compile", action="store_true", help="torch.compile(G,D) after load")
    pc.add_argument("--fused-adam", action="store_true", help="Use fused CUDA Adam when supported")
    pc.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help=f"EMA decay (default {EMA_DECAY})",
    )
    pc.set_defaults(func=cmd_phase_c)

    args = ap.parse_args()
    amp_dtype_str = str(AMP_DTYPE).replace("torch.", "") if USE_AMP else "off"
    print(
        f"[bootstrap] torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={DEVICE} AMP={USE_AMP} dtype={amp_dtype_str}",
        flush=True,
    )
    args.func(args)


if __name__ == "__main__":
    main()
