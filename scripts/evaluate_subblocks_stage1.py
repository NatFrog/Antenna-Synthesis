"""
Evaluate stage-1 ResUNet (5-ch composition input) from train_subblocks_stage1.

Metrics:
  - Coupling residual (normalised and dB)
  - Recomposed HFSS 4x4: composition + coupling
  - Baselines: composition only; MATLAB 4x4 vs recomposed HFSS truth
  - Antenna-specific metrics, null-region metrics, HFSS-above-floor regions
  - Per-sample CSV

Figures (under results/resunet_4x4_subblock_stage1_comp5ch/):
  - comparison_XX.png   MATLAB | composition | pred HFSS | true HFSS | error
  - coupling_XX.png, cuts_XX.png
  - error_distribution, scatter plots (HFSS and coupling)

Usage:
    python -m scripts.evaluate_subblocks_stage1
    python -m scripts.evaluate_subblocks_stage1 --split test --n-vis 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
from scipy.ndimage import minimum_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import BATCH_SIZE, DEVICE, N_THETA, NULL_THRESHOLD_DB, PROCESSED_DIR, RESULTS_DIR
from src.evaluation.antenna_metrics import compute_antenna_metric_errors
from src.evaluation.visualization import (
    plot_error_distribution,
    plot_plane_cuts,
    plot_scatter_pred_vs_true,
)
from src.training.metrics import (
    compute_batch_hfss_region_metrics,
    compute_batch_metrics,
    compute_pattern_metrics,
    mae,
    pearson_correlation,
    rmse,
)
from scripts.train_cgan_2to4_fusion_no_m4 import ATTN_HEADS, GEN_BASE, EnhancedResUNetGenerator
from scripts.train_subblocks_stage1 import (
    CKPT_DIR,
    COMP_NPZ,
    HDF5,
    IN_CH,
    NORM_2X2_HFSS,
    NORM_2X2_MATLAB,
    NORM_4X4,
    NORM_CPL,
    NORM_SUB,
    OUT_CH,
    SPLITS,
    SubblockCouplingDataset,
    _denorm_composition,
    _indices_with_composition,
    _load_compositions,
)

DEFAULT_CKPT = CKPT_DIR / "best_generator.pt"
DEFAULT_OUT = RESULTS_DIR / "resunet_4x4_subblock_stage1_comp5ch"


def global_idx_to_label(gi: int) -> str:
    return f"s{gi + 1:05d}"


def compute_null_metrics(
    pred: np.ndarray,
    tgt: np.ndarray,
    ref_for_mask: np.ndarray,
) -> dict[str, float]:
    """Null metrics using ref peak for -20 dB mask."""
    null_rmses, non_null_rmses, depth_errs = [], [], []
    nfa, nft = 0, 0
    for i in range(pred.shape[0]):
        p, t, ref = pred[i], tgt[i], ref_for_mask[i]
        peak = ref.max()
        mask_null = ref < (peak + NULL_THRESHOLD_DB)
        if mask_null.sum() > 0:
            null_rmses.append(float(np.sqrt(np.mean((p[mask_null] - t[mask_null]) ** 2))))
            pf = p[mask_null] - ref[mask_null]
            tf2 = t[mask_null] - ref[mask_null]
            nfa += int((np.abs(pf - tf2) < 2.0).sum())
            nft += int(mask_null.sum())
        mask_nn = ~mask_null
        if mask_nn.sum() > 0:
            non_null_rmses.append(float(np.sqrt(np.mean((p[mask_nn] - t[mask_nn]) ** 2))))
        lm = minimum_filter(ref, size=5)
        ilm = (ref == lm) & mask_null
        if ilm.sum() > 0:
            mp = np.argwhere(ilm)
            mv = ref[ilm]
            for idx in np.argsort(mv)[:10]:
                ti, pi = mp[idx]
                depth_errs.append(abs(p[ti, pi] - t[ti, pi]))
    return {
        "rmse_at_nulls_db": float(np.mean(null_rmses)) if null_rmses else float("nan"),
        "rmse_at_non_nulls_db": float(np.mean(non_null_rmses)) if non_null_rmses else float("nan"),
        "null_depth_error_db": float(np.mean(depth_errs)) if depth_errs else float("nan"),
        "null_fill_accuracy_pct": float(nfa / max(nft, 1) * 100),
    }


def plot_composition_comparison(
    matlab: np.ndarray,
    composition: np.ndarray,
    pred_hfss: np.ndarray,
    true_hfss: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
    vmin: float = -40,
    vmax: float = 0,
) -> None:
    import matplotlib.pyplot as plt

    error = pred_hfss - true_hfss
    extent = [-179.5, 179.5, 180, 0]
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    panels = [
        (matlab, "MATLAB 4x4"),
        (composition, "4x4 composition"),
        (pred_hfss, "Pred HFSS"),
        (true_hfss, "True HFSS"),
    ]
    for ax, (data, label) in zip(axes[:4], panels):
        im = ax.imshow(data, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="jet")
        ax.set_xlabel("Phi (deg)")
        ax.set_ylabel("Theta (deg)")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, label="dB")
    err_max = max(abs(error.min()), abs(error.max()), 1.0)
    im = axes[4].imshow(error, aspect="auto", extent=extent, vmin=-err_max, vmax=err_max, cmap="RdBu_r")
    axes[4].set_title(f"Error (pred-true)\nRMSE={rmse(pred_hfss, true_hfss):.3f} dB")
    axes[4].set_xlabel("Phi (deg)")
    axes[4].set_ylabel("Theta (deg)")
    plt.colorbar(im, ax=axes[4], label="dB")
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_coupling_maps(
    true_cpl: np.ndarray,
    pred_cpl: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt

    err = pred_cpl - true_cpl
    extent = [-179.5, 179.5, 180, 0]
    vmax = max(
        abs(true_cpl.min()), abs(true_cpl.max()),
        abs(pred_cpl.min()), abs(pred_cpl.max()), 5,
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, label in zip(axes[:2], [true_cpl, pred_cpl], ["True coupling", "Pred coupling"]):
        im = ax.imshow(data, aspect="auto", extent=extent, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_title(label)
        ax.set_xlabel("Phi (deg)")
        ax.set_ylabel("Theta (deg)")
        plt.colorbar(im, ax=ax, label="dB")
    err_max = max(abs(err.min()), abs(err.max()), 0.5)
    im = axes[2].imshow(err, aspect="auto", extent=extent, vmin=-err_max, vmax=err_max, cmap="RdBu_r")
    axes[2].set_title(f"Coupling error\nRMSE={rmse(pred_cpl, true_cpl):.3f} dB")
    axes[2].set_xlabel("Phi (deg)")
    axes[2].set_ylabel("Theta (deg)")
    plt.colorbar(im, ax=ax, label="dB")
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def _load_grids(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        if "theta_grid" in f and "phi_grid" in f:
            return f["theta_grid"][:].astype(np.float64), f["phi_grid"][:].astype(np.float64)
    return np.arange(0, N_THETA, dtype=np.float64), np.arange(-179.5, 180.0, 1.0, dtype=np.float64)


def _write_metrics_txt(path: Path, header: str, metrics: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--n-vis", type=int, default=10, help="number of comparison/cut figures")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    for p in (HDF5, COMP_NPZ, SPLITS, NORM_4X4, NORM_2X2_MATLAB, NORM_2X2_HFSS, NORM_SUB, NORM_CPL, args.ckpt):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    comp_4x4 = _load_compositions(COMP_NPZ, HDF5)
    n_comp = len(comp_4x4)

    s4 = np.load(NORM_4X4)
    sm2 = np.load(NORM_2X2_MATLAB)
    sh2 = np.load(NORM_2X2_HFSS)
    ss = np.load(NORM_SUB)
    sc = np.load(NORM_CPL)
    mean_4x4 = s4["mean"].astype(np.float32)
    std_4x4 = np.maximum(s4["std"].astype(np.float32), 1e-6)
    mean_m2 = sm2["mean"].astype(np.float32)
    std_m2 = np.maximum(sm2["std"].astype(np.float32), 1e-6)
    mean_h2 = sh2["mean"].astype(np.float32)
    std_h2 = np.maximum(sh2["std"].astype(np.float32), 1e-6)
    mean_sub = ss["mean"].astype(np.float32)
    std_sub = np.maximum(ss["std"].astype(np.float32), 1e-6)
    mean_cpl = sc["mean"].astype(np.float32)
    std_cpl = np.maximum(sc["std"].astype(np.float32), 1e-6)

    sp = np.load(SPLITS)
    indices = np.sort(_indices_with_composition(sp[args.split], n_comp))
    n = len(indices)

    ds = SubblockCouplingDataset(
        HDF5, comp_4x4, indices, mean_4x4, std_4x4,
        mean_m2, std_m2, mean_h2, std_h2,
        mean_sub, std_sub, mean_cpl, std_cpl,
        augment_noise=False,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    G = EnhancedResUNetGenerator(
        in_ch=IN_CH, out_ch=OUT_CH, base=GEN_BASE, attn_heads=ATTN_HEADS,
    ).to(DEVICE)
    G.load_state_dict(torch.load(args.ckpt, map_location=DEVICE, weights_only=True))
    print(
        f"Loaded {args.ckpt.name} | in_ch={IN_CH} | {args.split} n={n} | device={DEVICE}",
        flush=True,
    )

    G.eval()
    pred_cpl_list, true_cpl_list = [], []
    pred_hfss_list, true_hfss_list = [], []
    comp_list, matlab_list = [], []
    pred_cpl_n_list, true_cpl_n_list = [], []

    for x, y, _ in tqdm(loader, desc="Inference"):
        x = x.to(DEVICE)
        with torch.no_grad():
            pred_n = G(x).cpu().numpy()[:, 0]
        true_n = y.numpy()[:, 0]
        comp_n = x[:, 1].cpu().numpy()
        comp_db = _denorm_composition(comp_n, mean_sub, std_sub)
        pred_cpl = pred_n * std_cpl[None] + mean_cpl[None]
        true_cpl = true_n * std_cpl[None] + mean_cpl[None]
        pred_hfss = comp_db + pred_cpl
        true_hfss = comp_db + true_cpl
        m4_n = x[:, 0].cpu().numpy()
        matlab = m4_n * std_4x4[None] + mean_4x4[None]

        pred_cpl_list.append(pred_cpl)
        true_cpl_list.append(true_cpl)
        pred_hfss_list.append(pred_hfss)
        true_hfss_list.append(true_hfss)
        comp_list.append(comp_db)
        matlab_list.append(matlab)
        pred_cpl_n_list.append(pred_n)
        true_cpl_n_list.append(true_n)

    pred_cpl = np.concatenate(pred_cpl_list)
    true_cpl = np.concatenate(true_cpl_list)
    pred_hfss = np.concatenate(pred_hfss_list)
    true_hfss = np.concatenate(true_hfss_list)
    composition = np.concatenate(comp_list)
    matlab_4x4 = np.concatenate(matlab_list)
    pred_cpl_n = np.concatenate(pred_cpl_n_list)
    true_cpl_n = np.concatenate(true_cpl_n_list)

    hfss_pm = compute_batch_metrics(pred_hfss, true_hfss)
    print(f"\n=== Recomposed HFSS 4x4 ({args.split}, n={n}) ===")
    for k, v in hfss_pm.items():
        print(f"  {k}: {v:.6f}")

    cpl_pm = compute_batch_metrics(pred_cpl, true_cpl)
    print("\n=== Coupling residual (dB) ===")
    for k, v in cpl_pm.items():
        print(f"  {k}: {v:.6f}")

    cpl_norm_rmse = rmse(pred_cpl_n, true_cpl_n)
    cpl_norm_mae = mae(pred_cpl_n, true_cpl_n)
    cpl_norm_r = pearson_correlation(pred_cpl_n, true_cpl_n)
    print("\n=== Coupling residual (normalised) ===")
    print(f"  rmse_norm: {cpl_norm_rmse:.6f}")
    print(f"  mae_norm:  {cpl_norm_mae:.6f}")
    print(f"  pearson_r: {cpl_norm_r:.6f}")

    baseline_comp = compute_batch_metrics(composition, true_hfss)
    baseline_mat = compute_batch_metrics(matlab_4x4, true_hfss)
    print("\n=== Baseline: 4x4 composition (no coupling) vs true HFSS ===")
    for k, v in baseline_comp.items():
        print(f"  {k}: {v:.6f}")
    print("\n=== Baseline: MATLAB 4x4 vs true HFSS ===")
    for k, v in baseline_mat.items():
        print(f"  {k}: {v:.6f}")

    print("\n=== Improvement vs baselines (RMSE dB, lower is better) ===")
    print(f"  vs composition only: {baseline_comp['rmse_db'] - hfss_pm['rmse_db']:.6f} dB")
    print(f"  vs MATLAB 4x4:       {baseline_mat['rmse_db'] - hfss_pm['rmse_db']:.6f} dB")

    region_pm = compute_batch_hfss_region_metrics(pred_hfss, true_hfss)
    print("\n=== HFSS-above-floor regions (pred vs true HFSS) ===")
    for k, v in region_pm.items():
        print(f"  {k}: {v:.6f}")

    ae_list = []
    for i in range(n):
        try:
            ae_list.append(compute_antenna_metric_errors(pred_hfss[i], true_hfss[i]))
        except Exception:
            pass
    avg_ant = {}
    if ae_list:
        for key in ae_list[0]:
            vals = [e[key] for e in ae_list if not np.isnan(e[key])]
            avg_ant[key] = float(np.mean(vals)) if vals else float("nan")
    print("\n=== Antenna metrics (recomposed HFSS) ===")
    for k, v in avg_ant.items():
        print(f"  {k}: {v:.6f}")

    null_mat = compute_null_metrics(pred_hfss, true_hfss, matlab_4x4)
    null_comp = compute_null_metrics(pred_hfss, true_hfss, composition)
    null_hfss = compute_null_metrics(pred_hfss, true_hfss, true_hfss)
    print("\n=== Null metrics (mask from MATLAB peak) ===")
    for k, v in null_mat.items():
        print(f"  {k}: {v:.6f}")
    print("\n=== Null metrics (mask from composition peak) ===")
    for k, v in null_comp.items():
        print(f"  {k}: {v:.6f}")

    per_sample_path = args.out_dir / f"per_sample_{args.split}.csv"
    rows = [
        "global_idx,label,hfss_rmse_db,coupling_rmse_db,comp_baseline_rmse_db,"
        "matlab_baseline_rmse_db,hfss_mae_db,coupling_mae_db,hfss_pearson_r",
    ]
    for i, gi in enumerate(indices):
        label = global_idx_to_label(int(gi))
        hpm = compute_pattern_metrics(pred_hfss[i], true_hfss[i])
        cpm = compute_pattern_metrics(pred_cpl[i], true_cpl[i])
        rows.append(
            f"{gi},{label},"
            f"{hpm['rmse_db']:.6f},{cpm['rmse_db']:.6f},"
            f"{compute_pattern_metrics(composition[i], true_hfss[i])['rmse_db']:.6f},"
            f"{compute_pattern_metrics(matlab_4x4[i], true_hfss[i])['rmse_db']:.6f},"
            f"{hpm['mae_db']:.6f},{cpm['mae_db']:.6f},{hpm['pearson_r']:.6f}"
        )
    per_sample_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nPer-sample metrics: {per_sample_path}", flush=True)

    metrics_out = {
        "model": "train_subblocks_stage1 (5-ch composition)",
        "split": args.split,
        "n_samples": n,
        "in_ch": IN_CH,
        "checkpoint": str(args.ckpt),
        "compositions_npz": str(COMP_NPZ),
        **{f"hfss_{k}": v for k, v in hfss_pm.items()},
        **{f"coupling_{k}": v for k, v in cpl_pm.items()},
        "coupling_rmse_norm": cpl_norm_rmse,
        "coupling_mae_norm": cpl_norm_mae,
        "coupling_pearson_norm": cpl_norm_r,
        **{f"baseline_comp_{k}": v for k, v in baseline_comp.items()},
        **{f"baseline_matlab_{k}": v for k, v in baseline_mat.items()},
        "improvement_rmse_vs_comp_db": baseline_comp["rmse_db"] - hfss_pm["rmse_db"],
        "improvement_rmse_vs_matlab_db": baseline_mat["rmse_db"] - hfss_pm["rmse_db"],
        **region_pm,
        **{f"antenna_{k}": v for k, v in avg_ant.items()},
        **{f"null_matlab_{k}": v for k, v in null_mat.items()},
        **{f"null_comp_{k}": v for k, v in null_comp.items()},
        **{f"null_hfsspeak_{k}": v for k, v in null_hfss.items()},
    }
    metrics_path = args.out_dir / f"metrics_{args.split}.txt"
    _write_metrics_txt(
        metrics_path,
        "Evaluation: resunet_4x4_subblock_stage1_comp5ch",
        metrics_out,
    )
    print(f"Metrics saved: {metrics_path}", flush=True)

    vis_idx = np.linspace(0, n - 1, min(args.n_vis, n), dtype=int)
    theta_grid, phi_grid = _load_grids(HDF5)

    for plot_i, i in enumerate(tqdm(vis_idx, desc="Saving figures")):
        gi = int(indices[i])
        label = global_idx_to_label(gi)
        title = f"{args.split} {label} (idx={gi})"
        plot_composition_comparison(
            matlab_4x4[i], composition[i], pred_hfss[i], true_hfss[i],
            title=title,
            save_path=args.out_dir / f"comparison_{plot_i:02d}.png",
        )
        plot_coupling_maps(
            true_cpl[i], pred_cpl[i],
            title=f"Coupling — {title}",
            save_path=args.out_dir / f"coupling_{plot_i:02d}.png",
        )
        plot_plane_cuts(
            composition[i], pred_hfss[i], true_hfss[i],
            theta_grid=theta_grid,
            phi_grid=phi_grid,
            title=title,
            save_path=args.out_dir / f"cuts_{plot_i:02d}.png",
        )

    err_hfss = pred_hfss - true_hfss
    plot_error_distribution(
        err_hfss,
        title=f"Recomposed HFSS error ({args.split})",
        save_path=str(args.out_dir / f"error_distribution_{args.split}.png"),
    )
    plot_scatter_pred_vs_true(
        pred_hfss, true_hfss,
        title=f"Recomposed HFSS: pred vs true ({args.split})",
        save_path=str(args.out_dir / f"scatter_hfss_{args.split}.png"),
    )
    plot_scatter_pred_vs_true(
        pred_cpl, true_cpl,
        title=f"Coupling residual: pred vs true ({args.split})",
        save_path=str(args.out_dir / f"scatter_coupling_{args.split}.png"),
    )

    print(f"\nFigures and metrics written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
