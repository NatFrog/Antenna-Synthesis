"""
Rank held-out test samples by prediction error and report CSV provenance.

Runs the residual generator at N=16 on the 4to8 **test** split (200 samples by
default, or all of test_16x16 only with --test-16-only), ranks by full-pattern
RMSE vs hfss_16x16, and prints / saves the top-K closest matches.

CSV mapping (same as 4to8 / 8to16 fusion, 40 files x 50 configs):
  global_idx -> patterns_global_{file_idx:04d}.csv, column s{global_idx+1:05d}
  file_idx   = global_idx // 50 + 1
  config_idx = global_idx % 50  (0-based position within the file)

Usage::
    python -m scripts.rank_residual_test_samples \\
        --checkpoint checkpoints/residual_bootstrap_phase_c/best_generator.pt \\
        --top 5
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
from tqdm import tqdm

from src.config import BATCH_SIZE, DEVICE, N_CONFIGS_PER_FILE, PROCESSED_DIR
from src.data.loader import get_file_path
from src.training.metrics import compute_pattern_metrics, rmse
from scripts.evaluate_bootstrap_phase_table import _load_g, _synth_scale
from scripts.train_cgan_2to4_fusion_no_m4 import GEN_BASE, ATTN_HEADS, EnhancedResUNetGenerator

H5_4TO8 = PROCESSED_DIR / "antenna_data_4to8.h5"
H5_16X16 = PROCESSED_DIR / "antenna_data_8to16_subarray.h5"
M16_TEST = PROCESSED_DIR / "matlab_16x16_test.npz"
NORM_COMBINED = PROCESSED_DIR / "norm_stats_matlab_combined.npz"
SPLITS_4TO8 = PROCESSED_DIR / "split_indices_4to8.npz"
HFSS_16_CSV_DIR = PROJECT_ROOT / "datasets_16x16_hfss" / "datasets_16x16_hfss"


def global_idx_to_csv_label(global_idx: int) -> dict[str, str | int]:
    gi = int(global_idx)
    file_idx = gi // N_CONFIGS_PER_FILE + 1
    config_idx = gi % N_CONFIGS_PER_FILE
    csv_name = f"patterns_global_{file_idx:04d}.csv"
    col_name = f"s{gi + 1:05d}"  # globally sequential (matches loader.get_config_columns)
    return {
        "global_idx": gi,
        "file_idx": file_idx,
        "config_idx": config_idx,
        "csv_file": csv_name,
        "csv_column": col_name,
        "csv_path": str(get_file_path(HFSS_16_CSV_DIR, file_idx)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints/residual_bootstrap_phase_c/best_generator.pt"),
    )
    ap.add_argument("--top", type=int, default=5, help="Number of best (lowest RMSE) samples")
    ap.add_argument(
        "--test-16-only",
        action="store_true",
        help="Use only honest test_16x16 (100 samples), not full 4to8 test (200)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "results/residual_bootstrap_phase_c/top_samples.csv"),
    )
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt

    s = np.load(NORM_COMBINED)
    mean_mat = s["mean"].astype(np.float32)
    std_mat = s["std"].astype(np.float32)
    sigma_res = float(s["residual_std"])

    sp = np.load(SPLITS_4TO8)
    test48 = np.sort(sp["test"].astype(np.int64))
    pack = np.load(M16_TEST)
    if not np.array_equal(pack["test_idx"], test48):
        raise RuntimeError("matlab_16x16_test.npz test_idx mismatch")
    mat16_all = pack["arr"].astype(np.float32)

    if args.test_16_only:
        eval_idx = test48[100:]
        mat16 = mat16_all[100:]
    else:
        eval_idx = test48
        mat16 = mat16_all

    with h5py.File(H5_16X16, "r") as f:
        hf16 = f["hfss_16x16"][eval_idx].astype(np.float32)
        a16 = f["hfss_pred_8x8"][eval_idx].astype(np.float32)
    with h5py.File(H5_4TO8, "r") as f:
        meta16 = f["metadata"][eval_idx].astype(np.float32)

    print(f"Checkpoint: {ckpt}", flush=True)
    print(
        f"Evaluating N=16 on {len(eval_idx)} samples "
        f"({'test_16x16 only' if args.test_16_only else 'full 4to8 test split'})",
        flush=True,
    )

    G, recon = _load_g(ckpt, sigma_res)
    preds = _synth_scale(G, recon, mat16, a16, meta16, mean_mat, std_mat, 16)

    records = []
    for i in range(len(eval_idx)):
        gi = int(eval_idx[i])
        pm = compute_pattern_metrics(preds[i], hf16[i])
        loc = global_idx_to_csv_label(gi)
        records.append(
            {
                "rank_key_rmse_db": pm["rmse_db"],
                **loc,
                "rmse_db": pm["rmse_db"],
                "mae_db": pm["mae_db"],
                "pearson_r": pm["pearson_r"],
                "ssim": pm["ssim"],
                "in_test_16x16_honest": gi in set(test48[100:].tolist()),
            }
        )

    records.sort(key=lambda r: r["rmse_db"])
    top_k = records[: args.top]

    lines = [
        f"# Top {args.top} samples closest to HFSS (lowest RMSE) — bootstrap phase C @ 16x16\n",
        f"Checkpoint: `{ckpt.relative_to(PROJECT_ROOT)}`\n",
        f"Scored {len(records)} test samples.\n\n",
        "| Rank | RMSE (dB) | MAE (dB) | Pearson r | SSIM | global_idx | CSV file | Column | honest test_16x16 |",
        "|------|-----------|----------|-----------|------|------------|----------|--------|-------------------|",
    ]
    for rank, r in enumerate(top_k, start=1):
        honest = "yes" if r["in_test_16x16_honest"] else "no (val half of test split)"
        lines.append(
            f"| {rank} | {r['rmse_db']:.4f} | {r['mae_db']:.4f} | {r['pearson_r']:.4f} | "
            f"{r['ssim']:.4f} | {r['global_idx']} | {r['csv_file']} | {r['csv_column']} | {honest} |"
        )

    md = "\n".join(lines)
    print("\n" + md, flush=True)

    out_p = Path(args.out)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)

    import csv

    with open(out_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "global_idx",
                "file_idx",
                "config_idx",
                "csv_file",
                "csv_column",
                "csv_path",
                "rmse_db",
                "mae_db",
                "pearson_r",
                "ssim",
                "in_test_16x16_honest",
            ],
        )
        w.writeheader()
        for rank, r in enumerate(top_k, start=1):
            row = {k: r.get(k) for k in w.fieldnames if k != "rank"}
            row["rank"] = rank
            w.writerow(row)

    summary_path = out_p.with_suffix(".md")
    summary_path.write_text(md, encoding="utf-8")
    print(f"\nWrote {out_p} and {summary_path}", flush=True)

    print("\n--- Detail (top samples) ---", flush=True)
    for rank, r in enumerate(top_k, start=1):
        print(
            f"\n#{rank}  RMSE={r['rmse_db']:.4f} dB  "
            f"file patterns_global_{r['file_idx']:04d}.csv  column {r['csv_column']}  "
            f"(global_idx={r['global_idx']})"
        )


if __name__ == "__main__":
    main()
