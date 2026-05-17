"""
Final honest evaluation of the multi-scale residual cGAN (5-channel "with anchor"
variant) on test_16x16.

test_16x16 = last 100 of sorted 4to8 test indices, NEVER touched during training
(not even for checkpoint selection — that used the FIRST 100). This script loads
the best_generator.pt checkpoint (selected on val_16x16) and runs it once on
test_16x16, plus a sanity pass over val_16x16 for comparison.

The model takes 5 input channels: ch0 matlab_16x16_n, ch1 dphase_x/180,
ch2 dphase_y/180, ch3 scale_token=1.0, ch4 hfss_pred_8x8_n (cascade prediction
at one scale step below the target — the "anchor").

Usage:
    python -m scripts.evaluate_residual_multiscale

    Bootstrap Phase C checkpoint (16x16 honest test, same protocol):
    python -m scripts.evaluate_residual_multiscale \\
        --checkpoint checkpoints/residual_bootstrap_phase_c/best_generator.pt \\
        --results results/residual_bootstrap_phase_c
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
import torch
from tqdm import tqdm
from scipy.ndimage import minimum_filter
from scipy.signal import find_peaks

from src.config import (
    PROCESSED_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
    BATCH_SIZE, DEVICE, NULL_THRESHOLD_DB,
)
from src.training.metrics import compute_batch_metrics
from src.evaluation.antenna_metrics import compute_antenna_metric_errors
from src.evaluation.visualization import (
    plot_pattern_comparison, plot_plane_cuts,
    plot_error_distribution, plot_scatter_pred_vs_true,
)
from scripts.train_cgan_2to4_fusion_no_m4 import (
    EnhancedResUNetGenerator, GEN_BASE, ATTN_HEADS,
)
from scripts.train_residual_multiscale import ResidualReconLoss

H5_4TO8 = PROCESSED_DIR / "antenna_data_4to8.h5"
H5_16X16 = PROCESSED_DIR / "antenna_data_8to16_subarray.h5"
M16_TEST = PROCESSED_DIR / "matlab_16x16_test.npz"
NORM_COMBINED = PROCESSED_DIR / "norm_stats_matlab_combined.npz"
SPLITS_4TO8 = PROCESSED_DIR / "split_indices_4to8.npz"
CKPT_DEFAULT = CHECKPOINTS_DIR / "residual_multiscale_with_anchor" / "best_generator.pt"
RESULTS_DEFAULT = RESULTS_DIR / "residual_multiscale_with_anchor"


def synth_at_16x16(G, recon_loss, mat16, anchor16, hfss16, meta, mean_mat, std_mat):
    """Run G on 100-sample 16x16 batch, return predicted dB patterns.

    anchor16 : (N, 181, 360) hfss_pred_8x8 patterns (in dB, max-normed to 0
               in source) — stacked as ch4 after z-scoring with the same
               combined matlab stats as ch0.
    """
    n = len(mat16)
    preds = np.zeros_like(mat16, dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, n, BATCH_SIZE), desc="inference"):
            j = min(i + BATCH_SIZE, n)
            mat_batch = mat16[i:j]
            anchor_batch = anchor16[i:j]
            mat_n = (mat_batch - mean_mat[None]) / np.maximum(std_mat[None], 1e-6)
            anchor_n = (anchor_batch - mean_mat[None]) / np.maximum(std_mat[None], 1e-6)
            dpx = (meta[i:j, 0:1, None] / 180.0).astype(np.float32) * np.ones_like(mat_n)
            dpy = (meta[i:j, 1:2, None] / 180.0).astype(np.float32) * np.ones_like(mat_n)
            stk = np.full_like(mat_n, fill_value=1.0, dtype=np.float32)  # 16/16
            x = np.stack([mat_n, dpx, dpy, stk, anchor_n], axis=1)
            xt = torch.from_numpy(x).to(DEVICE)
            mt = torch.from_numpy(mat_batch[:, None]).to(DEVICE)
            delta_n = G(xt)
            pred_dB = recon_loss.compose(delta_n, mt)
            preds[i:j] = pred_dB.cpu().numpy()[:, 0]
    return preds


def report(name, preds, truth, mat16):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    pm = compute_batch_metrics(preds, truth)
    print("\n--- Pattern metrics ---")
    for k, v in pm.items():
        print(f"  {k}: {v:.6f}")

    ae = []
    for i in range(len(preds)):
        try:
            ae.append(compute_antenna_metric_errors(preds[i], truth[i]))
        except Exception:
            pass
    avg_ant = {}
    if ae:
        for key in ae[0]:
            vals = [e[key] for e in ae if not np.isnan(e[key])]
            avg_ant[key] = float(np.mean(vals)) if vals else float("nan")
        print("\n--- Antenna metrics ---")
        for k, v in avg_ant.items():
            print(f"  {k}: {v:.6f}")

    null_rmses, depth_errs = [], []
    nfa = nft = 0
    for i in range(len(preds)):
        mat = mat16[i]
        mask_null = mat < (mat.max() + NULL_THRESHOLD_DB)
        if mask_null.sum() > 0:
            null_rmses.append(float(np.sqrt(np.mean(
                (preds[i][mask_null] - truth[i][mask_null]) ** 2))))
            pf = preds[i][mask_null] - mat[mask_null]
            tf = truth[i][mask_null] - mat[mask_null]
            nfa += int((np.abs(pf - tf) < 2.0).sum())
            nft += int(mask_null.sum())
        lm = minimum_filter(mat, size=5)
        ilm = (mat == lm) & mask_null
        if ilm.sum() > 0:
            mp = np.argwhere(ilm); mv = mat[ilm]
            di = np.argsort(mv)[:10]
            for k in di:
                t, p = mp[k]
                depth_errs.append(abs(preds[i][t, p] - truth[i][t, p]))
    nm = {
        "rmse_at_nulls_db": float(np.mean(null_rmses)) if null_rmses else float("nan"),
        "null_depth_error_db": float(np.mean(depth_errs)) if depth_errs else float("nan"),
        "null_fill_accuracy_pct": float(nfa / max(nft, 1) * 100),
    }
    print("\n--- Null metrics (matlab_16x16 -20 dB threshold) ---")
    for k, v in nm.items():
        print(f"  {k}: {v:.6f}")

    return {**{f"pattern_{k}": v for k, v in pm.items()},
            **{f"antenna_{k}": v for k, v in avg_ant.items()},
            **{f"null_{k}": v for k, v in nm.items()}}


def main():
    p = argparse.ArgumentParser(description="Evaluate 5-ch residual multiscale on held-out test_16x16.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(CKPT_DEFAULT),
        help="EMA generator state dict (best_generator.pt)",
    )
    p.add_argument(
        "--results",
        type=str,
        default=str(RESULTS_DEFAULT),
        help="Directory for metrics.txt, predictions_test.npz, plots",
    )
    args = p.parse_args()
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.results)

    out_dir.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    run_tag = out_dir.name

    s = np.load(NORM_COMBINED)
    mean_mat = s["mean"].astype(np.float32)
    std_mat = s["std"].astype(np.float32)
    sigma_res = float(s["residual_std"])

    sp = np.load(SPLITS_4TO8)
    test48 = np.sort(sp["test"].astype(np.int64))
    val16_idx = test48[:100]
    test16_idx = test48[100:]
    print(f"val_16x16 (selection set):  {len(val16_idx)} samples")
    print(f"test_16x16 (held-out, this script): {len(test16_idx)} samples")

    # Load matlab_16x16 and split into val/test halves
    pack = np.load(M16_TEST)
    if not np.array_equal(pack["test_idx"], test48):
        raise RuntimeError("matlab_16x16_test.npz test_idx mismatch")
    mat16_all = pack["arr"].astype(np.float32)
    val_mat16 = mat16_all[:100]
    test_mat16 = mat16_all[100:]

    with h5py.File(H5_16X16, "r") as f:
        val_hfss16 = f["hfss_16x16"][val16_idx].astype(np.float32)
        test_hfss16 = f["hfss_16x16"][test16_idx].astype(np.float32)
        val_anchor = f["hfss_pred_8x8"][val16_idx].astype(np.float32)
        test_anchor = f["hfss_pred_8x8"][test16_idx].astype(np.float32)
    with h5py.File(H5_4TO8, "r") as f:
        val_meta = f["metadata"][val16_idx].astype(np.float32)
        test_meta = f["metadata"][test16_idx].astype(np.float32)

    # 5 input channels (matlab_n, dpx, dpy, scale_tok, anchor_n).
    G = EnhancedResUNetGenerator(in_ch=5, out_ch=1, base=GEN_BASE,
                                 attn_heads=ATTN_HEADS).to(DEVICE)
    G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    G.eval()
    recon_loss = ResidualReconLoss(sigma_res=sigma_res).to(DEVICE)

    print("\n--- val_16x16 (selection set, sanity check) ---")
    val_preds = synth_at_16x16(G, recon_loss, val_mat16, val_anchor, val_hfss16,
                               val_meta, mean_mat, std_mat)
    val_metrics = report("val_16x16 (selection set; not the honest number)",
                         val_preds, val_hfss16, val_mat16)

    print("\n--- test_16x16 (HELD-OUT, never touched during training) ---")
    test_preds = synth_at_16x16(G, recon_loss, test_mat16, test_anchor,
                                test_hfss16, test_meta, mean_mat, std_mat)
    test_metrics = report("test_16x16 (HELD-OUT honest result)",
                          test_preds, test_hfss16, test_mat16)

    base_test = report("Baseline (matlab_16x16 alone, no model) on test_16x16",
                       test_mat16, test_hfss16, test_mat16)

    with open(out_dir / "metrics.txt", "w") as f:
        f.write("Multi-scale residual cGAN (5-channel with-anchor) — final evaluation\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"sigma_residual: {sigma_res:.4f} dB\n\n")
        for name, d in [("VAL_16X16 (selection set)", val_metrics),
                        ("TEST_16X16 (HELD-OUT honest)", test_metrics),
                        ("BASELINE_TEST (matlab_16x16 alone)", base_test)]:
            f.write(f"--- {name} ---\n")
            for k, v in d.items():
                f.write(f"{k}: {v:.6f}\n")
            f.write("\n")
    print(f"\nMetrics saved to {out_dir}/metrics.txt", flush=True)

    # Plots from test_16x16 (the honest set)
    np.savez(out_dir / "predictions_test.npz",
             preds=test_preds, truth=test_hfss16, idx=test16_idx,
             mat16=test_mat16)
    vis_indices = np.linspace(0, len(test_preds) - 1, 10, dtype=int)
    for k, idx in enumerate(vis_indices):
        plot_pattern_comparison(
            test_mat16[idx], test_preds[idx], test_hfss16[idx],
            title=f"{run_tag} test_16x16 sample {idx} "
                  f"(4to8_idx={test16_idx[idx]})",
            save_path=str(out_dir / f"comparison_{k:02d}.png"),
        )
        plot_plane_cuts(
            test_mat16[idx], test_preds[idx], test_hfss16[idx],
            title=f"{run_tag} test_16x16 sample {idx}",
            save_path=str(out_dir / f"cuts_{k:02d}.png"),
        )
    plot_error_distribution(
        test_preds - test_hfss16,
        title=f"{run_tag} test_16x16 error distribution",
        save_path=str(out_dir / "error_distribution.png"),
    )
    plot_scatter_pred_vs_true(
        test_preds, test_hfss16,
        title=f"{run_tag}: predicted vs hfss_16x16 (test_16x16)",
        save_path=str(out_dir / "scatter_pred_vs_true.png"),
    )

    import os
    print(f"\nGenerated {len(os.listdir(out_dir))} files in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
