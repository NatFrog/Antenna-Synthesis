"""Compute null metrics at the -30 dB threshold for both fusion models.

The 4x4 cGAN already has `null30_*` metrics in its metrics.txt; the two fusion
evaluation scripts only compute null metrics at the default −20 dB threshold.
This script runs inference on the held-out test split for both fusion models
and reports the -30 dB null metrics so a 3-way comparison is possible.

Usage:
    python -m scripts.compute_null30_fusion
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import minimum_filter

from src.config import PROCESSED_DIR, CHECKPOINTS_DIR, BATCH_SIZE, DEVICE
from scripts.train_cgan_2to4_fusion import FusionDataset, ResUNetGenerator
from scripts.train_cgan_2to4_fusion_uw import FusionDatasetUW

NULL30_THRESHOLD_DB = -30.0


def run_inference(h5_path, splits_path, norm_2x2_path, norm_4x4_path,
                  ckpt_path, dataset_cls):
    sp = np.load(splits_path)
    test_idx = np.sort(sp["test"].astype(np.int64))

    s2 = np.load(norm_2x2_path); s4 = np.load(norm_4x4_path)
    mean_2x2 = s2["mean"].astype(np.float32); std_2x2 = np.maximum(s2["std"].astype(np.float32), 1e-6)
    mean_4x4 = s4["mean"].astype(np.float32); std_4x4 = np.maximum(s4["std"].astype(np.float32), 1e-6)

    ds = dataset_cls(h5_path, test_idx, mean_2x2, std_2x2, mean_4x4, std_4x4)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    G = ResUNetGenerator(in_ch=5, out_ch=1, base=32).to(DEVICE)
    G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    G.eval()

    preds_n, targets_n = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"{ckpt_path.parent.name}"):
            preds_n.append(G(x.to(DEVICE)).cpu().numpy())
            targets_n.append(y.numpy())
    preds_db = np.concatenate(preds_n)[:, 0] * std_4x4 + mean_4x4
    targets_db = np.concatenate(targets_n)[:, 0] * std_4x4 + mean_4x4

    with h5py.File(h5_path, "r") as f:
        matlab_4x4_raw = f["matlab_4x4"][test_idx].astype(np.float32)

    return preds_db, targets_db, matlab_4x4_raw


def null30_metrics(preds_db, targets_db, matlab_4x4_raw, label):
    n = len(preds_db)
    null_rmses, depth_errs = [], []
    nfa, nft = 0, 0
    b_null_rmses, b_depth = [], []
    bnfa, bnft = 0, 0

    for i in range(n):
        pred, tgt, mat = preds_db[i], targets_db[i], matlab_4x4_raw[i]
        peak = mat.max()
        mask = mat < (peak + NULL30_THRESHOLD_DB)
        if mask.sum() > 0:
            # Model
            null_rmses.append(float(np.sqrt(np.mean((pred[mask] - tgt[mask]) ** 2))))
            pf = pred[mask] - mat[mask]
            tf2 = tgt[mask] - mat[mask]
            nfa += int((np.abs(pf - tf2) < 2.0).sum())
            nft += int(mask.sum())
            # Baseline (uncorrected matlab)
            b_null_rmses.append(float(np.sqrt(np.mean((mat[mask] - tgt[mask]) ** 2))))
            bnfa += int((np.abs(tf2) < 2.0).sum())
            bnft += int(mask.sum())
        lm = minimum_filter(mat, size=5)
        ilm = (mat == lm) & mask
        if ilm.sum() > 0:
            mp = np.argwhere(ilm)
            mv = mat[ilm]
            di = np.argsort(mv)[:10]
            for idx in di:
                t, p = mp[idx]
                depth_errs.append(abs(pred[t, p] - tgt[t, p]))
                b_depth.append(abs(mat[t, p] - tgt[t, p]))

    out = {
        "null30_rmse_at_nulls_db": float(np.mean(null_rmses)),
        "null30_null_depth_error_db": float(np.mean(depth_errs)),
        "null30_null_fill_accuracy_pct": float(nfa / max(nft, 1) * 100),
        "null30_baseline_rmse_at_nulls_db": float(np.mean(b_null_rmses)),
        "null30_baseline_null_depth_error_db": float(np.mean(b_depth)),
        "null30_baseline_null_fill_accuracy_pct": float(bnfa / max(bnft, 1) * 100),
    }
    print(f"\n=== {label} (-30 dB threshold) ===")
    for k, v in out.items():
        print(f"  {k:42s} {v:.6f}")
    return out


def append_to_metrics(results_dir, new_metrics):
    """Append the null30_* lines to the existing metrics.txt for this model."""
    path = results_dir / "metrics.txt"
    existing = path.read_text().splitlines() if path.exists() else []
    # Drop any prior null30_* lines so this is idempotent
    kept = [ln for ln in existing if not ln.startswith("null30_")]
    new_lines = [f"{k}: {v:.6f}" for k, v in new_metrics.items()]
    path.write_text("\n".join(kept + new_lines) + "\n")
    print(f"  appended {len(new_lines)} lines to {path}")


def main():
    # ── Synthetic fusion ──
    orig = run_inference(
        h5_path=PROCESSED_DIR / "antenna_data_2to4.h5",
        splits_path=PROCESSED_DIR / "split_indices_2to4.npz",
        norm_2x2_path=PROCESSED_DIR / "norm_stats_2x2.npz",
        norm_4x4_path=PROCESSED_DIR / "norm_stats.npz",
        ckpt_path=CHECKPOINTS_DIR / "cgan_resunet_patchgan_2to4" / "best_generator.pt",
        dataset_cls=FusionDataset,
    )
    orig_n30 = null30_metrics(*orig, label="Synthetic fusion")
    append_to_metrics(PROJECT_ROOT / "results" / "cgan_resunet_patchgan_2to4", orig_n30)

    # ── Actual fusion ──
    uw = run_inference(
        h5_path=PROCESSED_DIR / "antenna_data_2to4_uw.h5",
        splits_path=PROCESSED_DIR / "split_indices_2to4_uw.npz",
        norm_2x2_path=PROCESSED_DIR / "norm_stats_2x2_uw.npz",
        norm_4x4_path=PROCESSED_DIR / "norm_stats.npz",
        ckpt_path=CHECKPOINTS_DIR / "cgan_resunet_patchgan_2to4_uw" / "best_generator.pt",
        dataset_cls=FusionDatasetUW,
    )
    uw_n30 = null30_metrics(*uw, label="Actual fusion")
    append_to_metrics(PROJECT_ROOT / "results" / "cgan_resunet_patchgan_2to4_uw", uw_n30)

    # ── Side-by-side print ──
    print("\n\n=== -30 dB null comparison (3-way, incl. existing 4x4 cGAN values) ===")
    print(f"{'metric':38s} {'4x4 cGAN':>12s} {'synthetic':>12s} {'actual':>12s}")
    # 4x4 cGAN values from results/cgan_unet_patchgan/metrics.txt (already on disk):
    cgan4 = {
        "null30_rmse_at_nulls_db":        0.344453,
        "null30_null_depth_error_db":     0.290278,
        "null30_null_fill_accuracy_pct":  99.723124,
        "null30_baseline_rmse_at_nulls_db": 15.542624,
        "null30_baseline_null_fill_accuracy_pct": 12.561442,
    }
    keys = [
        "null30_rmse_at_nulls_db",
        "null30_null_depth_error_db",
        "null30_null_fill_accuracy_pct",
        "null30_baseline_rmse_at_nulls_db",
        "null30_baseline_null_fill_accuracy_pct",
    ]
    for k in keys:
        a = cgan4.get(k, float("nan"))
        b = orig_n30.get(k, float("nan"))
        c = uw_n30.get(k, float("nan"))
        print(f"{k:38s} {a:12.4f} {b:12.4f} {c:12.4f}")


if __name__ == "__main__":
    main()
