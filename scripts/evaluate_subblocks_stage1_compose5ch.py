"""
Evaluate stage-1 compose5ch model (ch1 = position-wise b0..b3 composition).

Usage:
    python -m scripts.evaluate_subblocks_stage1_compose5ch
    python -m scripts.evaluate_subblocks_stage1_compose5ch --split test --n-vis 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import BATCH_SIZE, DEVICE, RESULTS_DIR
from scripts.compose_subblocks_4x4 import build_composed_4x4_from_h5, build_grids
from scripts.evaluate_subblocks_stage1 import (
    _load_grids,
    _write_metrics_txt,
    global_idx_to_label,
    plot_composition_comparison,
    plot_coupling_maps,
)
from scripts.train_cgan_2to4_fusion_no_m4 import ATTN_HEADS, GEN_BASE, EnhancedResUNetGenerator
from scripts.train_subblocks_stage1 import (
    IN_CH,
    OUT_CH,
    SubblockCouplingDataset,
    _denorm_composition,
)
from scripts.train_subblocks_stage1_compose5ch import (
    CKPT_DIR,
    HDF5,
    NORM_2X2_HFSS,
    NORM_2X2_MATLAB,
    NORM_4X4,
    NORM_COMPOSED,
    NORM_CPL,
    SPLITS,
)
from src.training.metrics import (
    compute_batch_metrics,
    compute_pattern_metrics,
    mae,
    pearson_correlation,
    rmse,
)
from src.evaluation.visualization import plot_error_distribution, plot_plane_cuts, plot_scatter_pred_vs_true

DEFAULT_CKPT = CKPT_DIR / "best_generator.pt"
DEFAULT_OUT = RESULTS_DIR / "resunet_4x4_subblock_stage1_compose5ch"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("test", "val"), default="test")
    ap.add_argument("--n-vis", type=int, default=10)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    for p in (HDF5, SPLITS, NORM_4X4, NORM_2X2_MATLAB, NORM_2X2_HFSS, NORM_CPL, args.ckpt):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    th, ph = build_grids()
    comp_4x4 = build_composed_4x4_from_h5(HDF5, th, ph, elem_combine="median")

    s4 = np.load(NORM_4X4)
    sm2 = np.load(NORM_2X2_MATLAB)
    sh2 = np.load(NORM_2X2_HFSS)
    sc = np.load(NORM_CPL)
    mean_4x4 = s4["mean"].astype(np.float32)
    std_4x4 = np.maximum(s4["std"].astype(np.float32), 1e-6)
    mean_m2 = sm2["mean"].astype(np.float32)
    std_m2 = np.maximum(sm2["std"].astype(np.float32), 1e-6)
    mean_h2 = sh2["mean"].astype(np.float32)
    std_h2 = np.maximum(sh2["std"].astype(np.float32), 1e-6)
    mean_cpl = sc["mean"].astype(np.float32)
    std_cpl = np.maximum(sc["std"].astype(np.float32), 1e-6)

    if NORM_COMPOSED.exists():
        nc = np.load(NORM_COMPOSED)
        mean_sub = nc["mean"].astype(np.float32)
        std_sub = np.maximum(nc["std"].astype(np.float32), 1e-6)
    else:
        sp = np.load(SPLITS)
        train_comp = comp_4x4[sp["train"]]
        mean_sub = train_comp.mean(axis=0).astype(np.float32)
        std_sub = np.maximum(train_comp.std(axis=0).astype(np.float32), 1e-6)

    sp = np.load(SPLITS)
    indices = np.sort(sp[args.split].astype(np.int64))
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
        f"Loaded {args.ckpt.name} | in_ch={IN_CH} ch1=compose | {args.split} n={n} | device={DEVICE}",
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
    print("\n=== Coupling residual (normalised) ===")
    print(f"  rmse_norm: {cpl_norm_rmse:.6f}")
    print(f"  mae_norm:  {mae(pred_cpl_n, true_cpl_n):.6f}")
    print(f"  pearson_r: {pearson_correlation(pred_cpl_n, true_cpl_n):.6f}")

    baseline_comp = compute_batch_metrics(composition, true_hfss)
    baseline_mat = compute_batch_metrics(matlab_4x4, true_hfss)
    print("\n=== Baseline: composed 4x4 (no coupling) vs true HFSS ===")
    for k, v in baseline_comp.items():
        print(f"  {k}: {v:.6f}")
    print("\n=== Baseline: MATLAB 4x4 vs true HFSS ===")
    for k, v in baseline_mat.items():
        print(f"  {k}: {v:.6f}")

    metrics_out = {
        "model": "train_subblocks_stage1_compose5ch",
        "split": args.split,
        "n_samples": n,
        "in_ch": IN_CH,
        "checkpoint": str(args.ckpt),
        "ch1_source": "compose_4x4_from_blocks(b0..b3)",
        **{f"hfss_{k}": v for k, v in hfss_pm.items()},
        **{f"coupling_{k}": v for k, v in cpl_pm.items()},
        "coupling_rmse_norm": cpl_norm_rmse,
        **{f"baseline_comp_{k}": v for k, v in baseline_comp.items()},
        **{f"baseline_matlab_{k}": v for k, v in baseline_mat.items()},
        "improvement_rmse_vs_comp_db": baseline_comp["rmse_db"] - hfss_pm["rmse_db"],
        "improvement_rmse_vs_matlab_db": baseline_mat["rmse_db"] - hfss_pm["rmse_db"],
    }
    metrics_path = args.out_dir / f"metrics_{args.split}.txt"
    _write_metrics_txt(metrics_path, "Evaluation: resunet_4x4_subblock_stage1_compose5ch", metrics_out)
    print(f"\nMetrics saved: {metrics_path}", flush=True)

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

    plot_error_distribution(
        pred_hfss - true_hfss,
        title=f"Recomposed HFSS error ({args.split})",
        save_path=str(args.out_dir / f"error_distribution_{args.split}.png"),
    )
    plot_scatter_pred_vs_true(
        pred_hfss, true_hfss,
        title=f"Recomposed HFSS: pred vs true ({args.split})",
        save_path=str(args.out_dir / f"scatter_hfss_{args.split}.png"),
    )
    print(f"\nFigures and metrics written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
