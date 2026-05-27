"""
Export top-ranked test samples as a canonical patterns_global-style CSV.

Reads rankings from ``rank_residual_test_samples`` (or recomputes them), runs
bootstrap phase C at N=16, and writes:

    - ``top_samples_predicted.csv`` — same layout as ``datasets_16x16_hfss/...``:
    ``theta_deg``, ``phi_deg``, globally indexed columns (e.g. ``s01572``),
    then the 181x360 grid (phi outer, theta inner).
  - ``top_samples_export.csv`` — small table: metrics + provenance + export column.

Usage::
    python -m scripts.export_top_samples_patterns_csv
    python -m scripts.export_top_samples_patterns_csv --rank-csv results/.../top_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch

from src.config import N_CONFIGS_PER_FILE, N_PHI, N_THETA, PROCESSED_DIR
from src.data.loader import get_file_path
from src.training.metrics import compute_pattern_metrics
from scripts.evaluate_bootstrap_phase_table import _load_g, _synth_scale
from scripts.rank_residual_test_samples import global_idx_to_csv_label

H5_4TO8 = PROCESSED_DIR / "antenna_data_4to8.h5"
H5_16X16 = PROCESSED_DIR / "antenna_data_8to16_subarray.h5"
M16_TEST = PROCESSED_DIR / "matlab_16x16_test.npz"
NORM_COMBINED = PROCESSED_DIR / "norm_stats_matlab_combined.npz"
SPLITS_4TO8 = PROCESSED_DIR / "split_indices_4to8.npz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/residual_bootstrap_phase_c"


def write_patterns_csv(
    out_path: Path,
    preds_db: np.ndarray,
    dphase_x: np.ndarray,
    dphase_y: np.ndarray,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    export_labels: list[str],
) -> None:
    """Write N samples in the same CSV layout as ``datasets_16x16_hfss`` files."""
    n = preds_db.shape[0]
    if len(export_labels) != n:
        raise ValueError("export_labels length must match number of patterns")

    theta_peak = np.zeros(n, dtype=np.float64)
    phi_peak = np.zeros(n, dtype=np.float64)
    for s in range(n):
        ti, pi = np.unravel_index(np.argmax(preds_db[s]), preds_db[s].shape)
        theta_peak[s] = theta_grid[ti]
        phi_peak[s] = phi_grid[pi]

    phi_grid_arr = phi_grid.astype(np.float64)
    theta_grid_arr = theta_grid.astype(np.float64)
    PHI_GRID, THETA_GRID = np.meshgrid(phi_grid_arr, theta_grid_arr, indexing="ij")
    theta_flat = THETA_GRID.reshape(-1)
    phi_flat = PHI_GRID.reshape(-1)

    flat_samples = np.empty((N_PHI * N_THETA, n), dtype=np.float32)
    for s in range(n):
        flat_samples[:, s] = preds_db[s].T.reshape(-1)

    headers = ["theta_deg", "phi_deg"] + export_labels
    out_rows: list[list[str]] = [headers]
    for label, values in (
        ("dphase_x_deg", dphase_x),
        ("dphase_y_deg", dphase_y),
        ("phi_peak_deg", phi_peak),
        ("theta_peak_deg", theta_peak),
    ):
        out_rows.append([label, ""] + [f"{v:.6f}" for v in values])

    data_block = np.column_stack(
        [theta_flat.reshape(-1, 1), phi_flat.reshape(-1, 1), flat_samples]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        for row in out_rows:
            f.write(",".join(map(str, row)) + "\n")
        pd.DataFrame(data_block).to_csv(
            f, header=False, index=False, float_format="%.6f"
        )


def load_rank_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints/residual_bootstrap_phase_c/best_generator.pt"),
    )
    ap.add_argument(
        "--rank-csv",
        type=str,
        default=str(DEFAULT_OUT_DIR / "top_samples.csv"),
        help="Rankings from rank_residual_test_samples (must include global_idx)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
    )
    ap.add_argument(
        "--patterns-name",
        type=str,
        default="top_samples_predicted.csv",
    )
    ap.add_argument(
        "--summary-name",
        type=str,
        default="top_samples_export.csv",
    )
    args = ap.parse_args()

    rank_path = Path(args.rank_csv)
    if not rank_path.is_absolute():
        rank_path = PROJECT_ROOT / rank_path
    ranks = load_rank_csv(rank_path)
    if not ranks:
        raise RuntimeError(f"No rows in {rank_path}")

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    global_indices = [int(r["global_idx"]) for r in ranks]
    n = len(global_indices)

    s = np.load(NORM_COMBINED)
    mean_mat = s["mean"].astype(np.float32)
    std_mat = s["std"].astype(np.float32)
    sigma_res = float(s["residual_std"])

    sp = np.load(SPLITS_4TO8)
    test48 = np.sort(sp["test"].astype(np.int64))
    pack = np.load(M16_TEST)
    if not np.array_equal(pack["test_idx"], test48):
        raise RuntimeError("matlab_16x16_test.npz test_idx mismatch")

    pos = {int(gi): i for i, gi in enumerate(test48)}
    missing = [gi for gi in global_indices if gi not in pos]
    if missing:
        raise RuntimeError(
            f"global_idx not in 4to8 test split: {missing}. "
            "Re-run rank_residual_test_samples on the same split."
        )
    rows = [pos[gi] for gi in global_indices]
    mat16 = pack["arr"][rows].astype(np.float32)

    gi_arr = np.asarray(global_indices, dtype=np.int64)
    order = np.argsort(gi_arr)
    gi_sorted = gi_arr[order]
    inv = np.empty_like(order)
    inv[order] = np.arange(n)

    with h5py.File(H5_16X16, "r") as f:
        hf16 = f["hfss_16x16"][gi_sorted].astype(np.float32)[inv]
        a16 = f["hfss_pred_8x8"][gi_sorted].astype(np.float32)[inv]
        meta16 = f["metadata"][gi_sorted].astype(np.float32)[inv]
        theta_grid = f["theta_grid"][:].astype(np.float64)
        phi_grid = f["phi_grid"][:].astype(np.float64)

    print(f"Checkpoint: {ckpt}", flush=True)
    print(f"Exporting {n} predicted 16x16 patterns -> {out_dir}", flush=True)

    G, recon = _load_g(ckpt, sigma_res)
    preds = _synth_scale(G, recon, mat16, a16, meta16, mean_mat, std_mat, 16)

    export_labels = [f"s{gi + 1:05d}" for gi in global_indices]
    patterns_path = out_dir / args.patterns_name
    write_patterns_csv(
        patterns_path,
        preds,
        meta16[:, 0].astype(np.float64),
        meta16[:, 1].astype(np.float64),
        theta_grid,
        phi_grid,
        export_labels,
    )

    summary_rows = []
    for i, r in enumerate(ranks):
        gi = global_indices[i]
        loc = global_idx_to_csv_label(gi)
        pm = compute_pattern_metrics(preds[i], hf16[i])
        summary_rows.append(
            {
                "rank": r.get("rank", i + 1),
                "export_column": export_labels[i],
                "global_idx": gi,
                "source_csv_file": loc["csv_file"],
                "source_csv_column": loc["csv_column"],
                "source_csv_path": loc["csv_path"],
                "rmse_db": f"{pm['rmse_db']:.6f}",
                "mae_db": f"{pm['mae_db']:.6f}",
                "pearson_r": f"{pm['pearson_r']:.6f}",
                "ssim": f"{pm['ssim']:.6f}",
                "in_test_16x16_honest": r.get("in_test_16x16_honest", ""),
            }
        )

    summary_path = out_dir / args.summary_name
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Wrote {patterns_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print("\nExport mapping (predicted column -> source HFSS CSV):", flush=True)
    for row in summary_rows:
        print(
            f"  {row['export_column']}  in  {row['source_csv_file']}  "
            f"(global_idx={row['global_idx']}, RMSE={row['rmse_db']} dB)",
            flush=True,
        )


if __name__ == "__main__":
    main()
