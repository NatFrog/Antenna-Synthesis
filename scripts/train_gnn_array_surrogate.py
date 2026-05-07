"""
Train a lattice Graph Convolution Network surrogate: MATLAB far-field + array graph → HFSS.

Default: 4×4 `processed/antenna_data.h5`, random 80/10/10 split unless `--split-npz` is set.

Multi-scale (example 4×4 + 2×2):

    python -m scripts.train_gnn_array_surrogate \\
      --hdf5-extra processed/antenna_data_2x2.h5 --nx-extra 2 --ny-extra 2
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE,
    DEVICE,
    HDF5_PATH,
    LEARNING_RATE,
    N_PHI,
    N_THETA,
    RANDOM_SEED,
    CHECKPOINTS_DIR,
)
from src.models.gnn_array_surrogate import ArrayCouplingGNN


@dataclass(frozen=True)
class ScaleSpec:
    nx: int
    ny: int
    h5_path: Path
    split_npz: Path | None


class LatticeArrayHDF5Dataset(Dataset):
    """HDF5 MATLAB/HFSS pairs; HFSS normalized with global pooled stats."""

    def __init__(
        self,
        spec: ScaleSpec,
        split_indices: np.ndarray,
        hfss_mean: float,
        hfss_std: float,
        pixels_per_sample: int,
        sample_rng_base: int,
    ):
        self.spec = spec
        self.indices = np.sort(split_indices.astype(np.int64))
        self.hfss_mean = float(hfss_mean)
        self.hfss_std = float(max(hfss_std, 1e-6))
        self.pixels_per_sample = int(pixels_per_sample)
        self._sample_rng_base = int(sample_rng_base)
        self._h5: h5py.File | None = None
        self._theta: np.ndarray | None = None
        self._phi: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(str(self.spec.h5_path), "r")
            self._theta = np.asarray(self._h5["theta_grid"][:], dtype=np.float32)
            self._phi = np.asarray(self._h5["phi_grid"][:], dtype=np.float32)

    def __getitem__(self, i: int):
        self._open()
        assert self._h5 is not None and self._theta is not None and self._phi is not None
        gidx = int(self.indices[i])
        ml = np.asarray(self._h5["matlab_patterns"][gidx], dtype=np.float32)
        hfss = np.asarray(self._h5["hfss_patterns"][gidx], dtype=np.float32)
        meta = self._h5["metadata"][gidx][:2]

        rnd = random.Random(self._sample_rng_base + gidx * 9176 + i * 31337)
        total = N_THETA * N_PHI
        coords = rnd.sample(range(total), k=self.pixels_per_sample)

        ti = np.array([c // N_PHI for c in coords], dtype=np.int64)
        pi = np.array([c % N_PHI for c in coords], dtype=np.int64)
        tgt = hfss[ti, pi].astype(np.float32)
        tgt_norm = (tgt - self.hfss_mean) / self.hfss_std

        return {
            "matlab": torch.from_numpy(ml),
            "tgt_norm": torch.from_numpy(tgt_norm),
            "theta": torch.from_numpy(self._theta[ti]),
            "phi": torch.from_numpy(self._phi[pi]),
            "dphase_x": float(meta[0]),
            "dphase_y": float(meta[1]),
        }


def _collate_same_scale(batch):
    mats = torch.stack([b["matlab"] for b in batch]).float()
    tgt = torch.stack([b["tgt_norm"] for b in batch]).float()
    theta = torch.stack([b["theta"] for b in batch]).float()
    phi = torch.stack([b["phi"] for b in batch]).float()
    dpx = torch.tensor([b["dphase_x"] for b in batch], dtype=torch.float32)
    dpy = torch.tensor([b["dphase_y"] for b in batch], dtype=torch.float32)
    return dict(matlab=mats, tgt=tgt, theta=theta, phi=phi, dphase_x=dpx, dphase_y=dpy)


def _count_h5_patterns(path: Path) -> int:
    with h5py.File(str(path), "r") as fh:
        return int(fh["matlab_patterns"].shape[0])


def _random_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ix = np.arange(n)
    rng.shuffle(ix)
    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    tr = ix[:n_train]
    va = ix[n_train : n_train + n_val]
    te = ix[n_train + n_val :]
    if len(te) == 0:
        te = va[-max(1, len(va) // 10) :].copy()
    return tr, va, te


def _load_splits(spec: ScaleSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spec.split_npz and spec.split_npz.exists():
        z = np.load(spec.split_npz)
        return z["train"], z["val"], z["test"]
    n = _count_h5_patterns(spec.h5_path)
    return _random_split(n, RANDOM_SEED + spec.nx * 97 + spec.ny)


def pooled_hfss_stats(specs: list[ScaleSpec], train_indices_list: list[np.ndarray], sample_cap: int) -> tuple[float, float]:
    acc: list[np.ndarray] = []
    per = max(1, sample_cap // max(1, len(specs)))
    for spec, tr in zip(specs, train_indices_list, strict=True):
        subset = np.sort(tr[:per]) if len(tr) > per else tr
        with h5py.File(str(spec.h5_path), "r") as fh:
            for j in subset:
                acc.append(np.asarray(fh["hfss_patterns"][j], dtype=np.float32).reshape(-1))
    allv = np.concatenate(acc, axis=0)
    return float(np.mean(allv)), float(np.std(allv))


def train_step(model, batch, nx, ny, opt, criterion, device):
    mats = batch["matlab"].to(device)
    dpx = batch["dphase_x"].to(device)
    dpy = batch["dphase_y"].to(device)
    tg = batch["tgt"].reshape(-1).to(device)
    bsz, kpix = mats.shape[0], batch["tgt"].shape[1]
    theta_flat = batch["theta"].reshape(-1).to(device)
    phi_flat = batch["phi"].reshape(-1).to(device)

    opt.zero_grad(set_to_none=True)
    preds = model(mats, dpx, dpy, nx, ny, theta_flat, phi_flat)
    loss = criterion(preds.view(-1), tg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    return float(loss.item())


@torch.no_grad()
def val_epoch(model, loader, nx, ny, device):
    criterion = nn.L1Loss()
    tot = 0.0
    n_batches = 0
    model.eval()
    for batch in tqdm(loader, desc="Val", leave=False):
        mats = batch["matlab"].to(device)
        dpx = batch["dphase_x"].to(device)
        dpy = batch["dphase_y"].to(device)
        tg = batch["tgt"].reshape(-1).to(device)
        theta_flat = batch["theta"].reshape(-1).to(device)
        phi_flat = batch["phi"].reshape(-1).to(device)
        preds = model(mats, dpx, dpy, nx, ny, theta_flat, phi_flat)
        tot += criterion(preds.view(-1), tg).item()
        n_batches += 1
    return tot / max(1, n_batches)


def parse_args():
    ap = argparse.ArgumentParser(description="Train GNN planar-array surrogate (MATLAB → HFSS).")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--pixels-per-sample", type=int, default=4096, help="RF directions per map per batch item.")
    ap.add_argument("--stats-samples-cap", type=int, default=400, help="Max train maps used for pooled HFSS mu/sigma.")

    # Single-scale defaults
    ap.add_argument("--hdf5", type=Path, default=HDF5_PATH)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--split-npz", type=Path, default=None)

    # Optional second scale (minimal multi-scale MVP)
    ap.add_argument("--hdf5-extra", type=Path, default=None, help="Second HDF5 (e.g. 2×2).")
    ap.add_argument("--nx-extra", type=int, default=2)
    ap.add_argument("--ny-extra", type=int, default=2)
    ap.add_argument("--split-extra", type=Path, default=None)

    ap.add_argument("--out-dir", type=Path, default=CHECKPOINTS_DIR / "gnn_array_surrogate")
    ap.add_argument("--device", default=str(DEVICE))
    return ap.parse_args()


def main():
    args = parse_args()

    specs: list[ScaleSpec] = [
        ScaleSpec(args.nx, args.ny, args.hdf5, args.split_npz),
    ]
    if args.hdf5_extra is not None:
        specs.append(
            ScaleSpec(args.nx_extra, args.ny_extra, args.hdf5_extra, args.split_extra),
        )

    for s in specs:
        if not s.h5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {s.h5_path}")

    splits = [_load_splits(s) for s in specs]
    train_ix = [s[0] for s in splits]
    val_ix = [s[1] for s in splits]

    mu, sig = pooled_hfss_stats(specs, train_ix, args.stats_samples_cap)
    print(f"Pooled HFSS train stats: mean={mu:.4f} dB std={sig:.4f} dB")

    train_sets = []
    val_sets = []
    for spec, (tr_i, va_i, _) in zip(specs, splits, strict=True):
        train_sets.append(
            LatticeArrayHDF5Dataset(
                spec,
                tr_i,
                mu,
                sig,
                pixels_per_sample=args.pixels_per_sample,
                sample_rng_base=RANDOM_SEED + len(train_sets),
            ),
        )
        val_sets.append(
            LatticeArrayHDF5Dataset(
                spec,
                va_i,
                mu,
                sig,
                pixels_per_sample=args.pixels_per_sample // 2,
                sample_rng_base=RANDOM_SEED + 7919 + len(val_sets),
            ),
        )

    train_loaders = [
        DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            collate_fn=_collate_same_scale,
        )
        for ds in train_sets
    ]
    val_loaders = [
        DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate_same_scale)
        for ds in val_sets
    ]

    device = torch.device(args.device)

    model = ArrayCouplingGNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.L1Loss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_steps = 0
        for loader, spec in zip(train_loaders, specs, strict=True):
            for batch in tqdm(loader, desc=f"Train ep{epoch} {spec.nx}x{spec.ny}", leave=False):
                loss = train_step(model, batch, spec.nx, spec.ny, opt, criterion, device)
                running += loss
                n_steps += 1
        avg_tr = running / max(1, n_steps)
        print(f"Epoch {epoch}  train L1 (norm HFSS): {avg_tr:.6f}")

        vals = []
        for vld, spec in zip(val_loaders, specs, strict=True):
            vals.append(val_epoch(model, vld, spec.nx, spec.ny, device))
        mval = float(np.mean(vals))
        print(f"Epoch {epoch}  mean val L1 (norm HFSS space): {mval:.6f}")
        if mval < best_val:
            best_val = mval
            ck = args.out_dir / "best_gnn_surrogate.pt"
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    hfss_mean=np.float32(mu),
                    hfss_std=np.float32(sig),
                    scales=[dict(nx=s.nx, ny=s.ny, h5=str(s.h5_path)) for s in specs],
                ),
                ck,
            )
            print(f"  saved → {ck}")


if __name__ == "__main__":
    main()
