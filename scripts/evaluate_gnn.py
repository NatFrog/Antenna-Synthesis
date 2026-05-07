"""
Roll out trained `ArrayCouplingGNN` over full (theta, phi) grids for HDF5 MATLAB inputs.

Example:

    python -m scripts.evaluate_gnn_array_surrogate \\
      --checkpoint checkpoints/gnn_array_surrogate/best_gnn_surrogate.pt \\
      --hdf5 processed/antenna_data.h5 \\
      --nx 4 --ny 4 --subset val
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEVICE, HDF5_PATH, N_PHI, N_THETA, RANDOM_SEED, RESULTS_DIR
from src.evaluation.visualization import (
    plot_error_distribution,
    plot_pattern_comparison,
    plot_plane_cuts,
    plot_scatter_pred_vs_true,
)
from src.models.gnn_array_surrogate import ArrayCouplingGNN


def _split_indices(split_npz: Path | None, subset: str, n_tot: int) -> np.ndarray:
    if split_npz and split_npz.exists():
        z = np.load(split_npz)
        if subset == "all":
            return np.arange(n_tot, dtype=np.int64)
        return np.sort(z[subset].astype(np.int64))
    rng = np.random.default_rng(RANDOM_SEED)
    ix = np.arange(n_tot)
    rng.shuffle(ix)
    n_train = max(1, int(0.8 * n_tot))
    n_val = max(1, int(0.1 * n_tot))
    if subset == "all":
        return np.sort(ix)
    if subset == "train":
        return np.sort(ix[:n_train])
    if subset == "val":
        return np.sort(ix[n_train : n_train + n_val])
    return np.sort(ix[n_train + n_val :])


def build_direction_flat(theta_grid_deg: np.ndarray, phi_grid_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ti = np.arange(N_THETA, dtype=np.int64).repeat(N_PHI)
    pj = np.tile(np.arange(N_PHI, dtype=np.int64), N_THETA)
    return theta_grid_deg[ti].astype(np.float32), phi_grid_deg[pj].astype(np.float32)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--hdf5", type=Path, default=HDF5_PATH)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--split-npz", type=Path, default=None)
    ap.add_argument("--subset", choices=("train", "val", "test", "all"), default="test")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--dir-chunk", type=int, default=8192)
    ap.add_argument("--device", default=str(DEVICE))
    ap.add_argument(
        "--experiment-name",
        type=str,
        default="gnn_array_surrogate",
        help="Subfolder under results/ for PNG outputs and metrics.",
    )
    ap.add_argument(
        "--n-vis-samples",
        type=int,
        default=10,
        help="Number of maps to plot (comparison + plane cuts); spaced across the evaluated subset.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    results_dir = RESULTS_DIR / args.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    mu = float(ck["hfss_mean"])
    sig = float(max(ck["hfss_std"], 1e-6))

    model = ArrayCouplingGNN().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    with h5py.File(str(args.hdf5), "r") as fh:
        n_tot = fh["matlab_patterns"].shape[0]
        idx = _split_indices(args.split_npz, args.subset, n_tot)
        if args.samples is not None:
            idx = idx[: args.samples]

        tg_deg = fh["theta_grid"][:].astype(np.float64)
        pg_deg = fh["phi_grid"][:].astype(np.float64)
        theta_flat, phi_flat = build_direction_flat(tg_deg, pg_deg)

        errs = []
        mlab_list: list[np.ndarray] = []
        pred_list: list[np.ndarray] = []
        tgt_list: list[np.ndarray] = []
        for gidx in tqdm(idx, desc="Maps"):
            ml_np = np.asarray(fh["matlab_patterns"][gidx], dtype=np.float32)
            ml = torch.from_numpy(ml_np).unsqueeze(0)
            meta = fh["metadata"][gidx][:2]
            dpx = torch.tensor([meta[0]], dtype=torch.float32)
            dpy = torch.tensor([meta[1]], dtype=torch.float32)
            tgt = np.asarray(fh["hfss_patterns"][gidx], dtype=np.float32)

            preds: list[np.ndarray] = []
            ml_d = ml.to(device)
            dpx_d = dpx.to(device)
            dpy_d = dpy.to(device)

            total = theta_flat.shape[0]
            for start in range(0, total, args.dir_chunk):
                end = min(start + args.dir_chunk, total)
                th = torch.from_numpy(theta_flat[start:end]).to(device)
                ph = torch.from_numpy(phi_flat[start:end]).to(device)
                p = model(ml_d, dpx_d, dpy_d, args.nx, args.ny, th, ph)
                preds.append(p.detach().cpu().numpy().astype(np.float64) * sig + mu)

            phf = np.concatenate(preds).reshape(N_THETA, N_PHI)
            rmse = math.sqrt(np.mean((phf - tgt) ** 2))
            errs.append(rmse)
            mlab_list.append(ml_np)
            pred_list.append(phf.astype(np.float32))
            tgt_list.append(tgt)

    mean_rmse = float(np.mean(errs))
    print(f"Mean RMSE dB ({len(errs)} maps): {mean_rmse:.4f}")

    all_ml = np.stack(mlab_list, axis=0)
    all_pred = np.stack(pred_list, axis=0)
    all_tgt = np.stack(tgt_list, axis=0)
    n_maps = all_pred.shape[0]

    with open(results_dir / "metrics.txt", "w", encoding="utf-8") as mf:
        mf.write("evaluate_gnn_array_surrogate\n")
        mf.write(f"maps: {n_maps}\n")
        mf.write(f"mean_rmse_db: {mean_rmse:.6f}\n")

    vis_n = min(args.n_vis_samples, n_maps)
    vis_ix = np.linspace(0, n_maps - 1, vis_n, dtype=int)
    for k, si in enumerate(vis_ix):
        plot_pattern_comparison(
            all_ml[si],
            all_pred[si],
            all_tgt[si],
            title=f"GNN surrogate map {k} (h5 idx {int(idx[si])})",
            save_path=str(results_dir / f"comparison_{k:04d}.png"),
        )
        plot_plane_cuts(
            all_ml[si],
            all_pred[si],
            all_tgt[si],
            theta_grid=tg_deg,
            phi_grid=pg_deg,
            title=f"GNN surrogate map {k} (h5 idx {int(idx[si])})",
            save_path=str(results_dir / f"cuts_{k:04d}.png"),
        )

    errors = all_pred - all_tgt
    plot_error_distribution(
        errors,
        title=f"{args.experiment_name}: pointwise error (pred - HFSS)",
        save_path=str(results_dir / "error_distribution.png"),
    )

    np.random.seed(RANDOM_SEED)
    plot_scatter_pred_vs_true(
        all_pred,
        all_tgt,
        title=f"{args.experiment_name}: predicted vs true HFSS",
        save_path=str(results_dir / "scatter_pred_vs_true.png"),
    )

    print(f"Figures and metrics saved to: {results_dir}")


if __name__ == "__main__":
    main()
