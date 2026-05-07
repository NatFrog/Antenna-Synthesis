"""
Cross-scale evaluation: load a GNN surrogate checkpoint (e.g. trained on 4×4) and
roll it out on **another array size** using that scale's MATLAB inputs and HFSS labels.

Default target is 8×8: ``processed/antenna_data_8x8.h5`` with ``--nx 8 --ny 8``.
Denormalization uses **checkpoint** ``hfss_mean`` / ``hfss_std`` (training stats), as the
decoder was trained in that normalized space; errors are vs raw 8×8 HFSS dB in the HDF5.

Example:

    python -m scripts.evaluate_gnn_cross_scale \\
      --checkpoint checkpoints/gnn_array_surrogate/best_gnn_surrogate.pt \\
      --hdf5 processed/antenna_data_8x8.h5 --nx 8 --ny 8 --subset test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_gnn_array_surrogate as _roll

from src.config import DEVICE, N_PHI, N_THETA, RANDOM_SEED, RESULTS_DIR
from src.evaluation.visualization import (
    plot_error_distribution,
    plot_pattern_comparison,
    plot_plane_cuts,
    plot_scatter_pred_vs_true,
)
from src.models.gnn_array_surrogate import ArrayCouplingGNN

_DEFAULT_H5_8X8 = PROJECT_ROOT / "processed" / "antenna_data_8x8.h5"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate GNN surrogate on a different array scale than training (e.g. 8×8 data)."
    )
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--hdf5",
        type=Path,
        default=_DEFAULT_H5_8X8,
        help="HDF5 with matlab_patterns / hfss_patterns / metadata for the eval scale.",
    )
    ap.add_argument("--nx", type=int, default=8, help="Lattice size at eval (element count x).")
    ap.add_argument("--ny", type=int, default=8, help="Lattice size at eval (element count y).")
    ap.add_argument("--split-npz", type=Path, default=None)
    ap.add_argument("--subset", choices=("train", "val", "test", "all"), default="test")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--dir-chunk", type=int, default=8192)
    ap.add_argument("--device", default=str(DEVICE))
    ap.add_argument(
        "--experiment-name",
        type=str,
        default="gnn_cross_scale_8x8",
        help="Subfolder under results/ for PNG outputs.",
    )
    ap.add_argument("--n-vis-samples", type=int, default=10)
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
    train_scales = ck.get("scales")

    model = ArrayCouplingGNN().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    eval_tag = f"{args.nx}x{args.ny}"

    with h5py.File(str(args.hdf5), "r") as fh:
        n_tot = fh["matlab_patterns"].shape[0]
        idx = _roll._split_indices(args.split_npz, args.subset, n_tot)
        if args.samples is not None:
            idx = idx[: args.samples]

        tg_deg = fh["theta_grid"][:].astype(np.float64)
        pg_deg = fh["phi_grid"][:].astype(np.float64)
        theta_flat, phi_flat = _roll.build_direction_flat(tg_deg, pg_deg)

        errs: list[float] = []
        mlab_list: list[np.ndarray] = []
        pred_list: list[np.ndarray] = []
        tgt_list: list[np.ndarray] = []

        for gidx in tqdm(idx, desc=f"Maps ({eval_tag} eval)"):
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
    print(
        f"Cross-scale ({eval_tag} data, checkpoint denorm): "
        f"mean RMSE dB ({len(errs)} maps) = {mean_rmse:.4f}"
    )

    all_ml = np.stack(mlab_list, axis=0)
    all_pred = np.stack(pred_list, axis=0)
    all_tgt = np.stack(tgt_list, axis=0)
    n_maps = all_pred.shape[0]

    with open(results_dir / "metrics.txt", "w", encoding="utf-8") as mf:
        mf.write("evaluate_gnn_cross_scale\n")
        mf.write(f"checkpoint: {args.checkpoint}\n")
        mf.write(f"eval_hdf5: {args.hdf5}\n")
        mf.write(f"eval_lattice: nx={args.nx} ny={args.ny}\n")
        mf.write(f"checkpoint_hfss_mean: {mu:.6f}\n")
        mf.write(f"checkpoint_hfss_std: {sig:.6f}\n")
        if train_scales is not None:
            mf.write(f"checkpoint_train_scales: {json.dumps(train_scales)}\n")
        mf.write(f"maps: {n_maps}\n")
        mf.write(f"mean_rmse_db_vs_hfss: {mean_rmse:.6f}\n")

    vis_n = min(args.n_vis_samples, n_maps)
    vis_ix = np.linspace(0, n_maps - 1, vis_n, dtype=int)

    for k, si in enumerate(vis_ix):
        plot_pattern_comparison(
            all_ml[si],
            all_pred[si],
            all_tgt[si],
            title=f"Cross-scale {eval_tag} | sample {k} (HDF5 idx {int(idx[si])})",
            save_path=str(results_dir / f"comparison_{k:04d}.png"),
        )
        plot_plane_cuts(
            all_ml[si],
            all_pred[si],
            all_tgt[si],
            theta_grid=tg_deg,
            phi_grid=pg_deg,
            title=f"Cross-scale {eval_tag} | sample {k} (HDF5 idx {int(idx[si])})",
            save_path=str(results_dir / f"cuts_{k:04d}.png"),
        )

    errors = all_pred - all_tgt
    plot_error_distribution(
        errors,
        title=f"{args.experiment_name}: pred − HFSS ({eval_tag} GT)",
        save_path=str(results_dir / "error_distribution.png"),
    )

    np.random.seed(RANDOM_SEED)
    plot_scatter_pred_vs_true(
        all_pred,
        all_tgt,
        title=f"{args.experiment_name}: pred vs HFSS ({eval_tag})",
        save_path=str(results_dir / "scatter_pred_vs_true.png"),
    )

    print(f"Figures and metrics saved to: {results_dir}")


if __name__ == "__main__":
    main()
