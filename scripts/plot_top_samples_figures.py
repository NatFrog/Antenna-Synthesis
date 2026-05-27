"""
Plot comparison + plane-cut figures for top-ranked samples in one PDF.

Reads ``top_samples_export.csv``, runs bootstrap phase C @ 16x16, and writes
``top_samples_figures.pdf`` (one page per sample, labelled by global column id).

Usage::
    python -m scripts.plot_top_samples_figures
    python -m scripts.plot_top_samples_figures --export-csv results/.../top_samples_export.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from src.config import N_PHI, N_THETA, PROCESSED_DIR
from scripts.evaluate_bootstrap_phase_table import _load_g, _synth_scale

H5_4TO8 = PROCESSED_DIR / "antenna_data_4to8.h5"
H5_16X16 = PROCESSED_DIR / "antenna_data_8to16_subarray.h5"
M16_TEST = PROCESSED_DIR / "matlab_16x16_test.npz"
NORM_COMBINED = PROCESSED_DIR / "norm_stats_matlab_combined.npz"
SPLITS_4TO8 = PROCESSED_DIR / "split_indices_4to8.npz"
DEFAULT_OUT = PROJECT_ROOT / "results/residual_bootstrap_phase_c/top_samples_figures.pdf"


def _load_export_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plot_sample_page(
    matlab: np.ndarray,
    predicted: np.ndarray,
    hfss: np.ndarray,
    column_id: str,
    global_idx: int,
    rmse_db: float,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
) -> plt.Figure:
    """One page: 4-panel heatmaps + E/H plane cuts."""
    error = predicted - hfss
    extent = [-179.5, 179.5, 180, 0]
    peak_idx = np.unravel_index(np.argmax(hfss), hfss.shape)
    peak_theta_idx, peak_phi_idx = peak_idx

    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.28)

    for j, (data, label) in enumerate(
        zip(
            [matlab, predicted, hfss],
            ["MATLAB 16x16 (input)", "Predicted HFSS 16x16", "HFSS 16x16 (truth)"],
        )
    ):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(data, aspect="auto", extent=extent, vmin=-40, vmax=0, cmap="jet")
        ax.set_xlabel("Phi (deg)")
        ax.set_ylabel("Theta (deg)")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, label="dB", fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 3])
    err_max = max(abs(float(error.min())), abs(float(error.max())), 5.0)
    im = ax.imshow(
        error, aspect="auto", extent=extent, vmin=-err_max, vmax=err_max, cmap="RdBu_r"
    )
    ax.set_xlabel("Phi (deg)")
    ax.set_ylabel("Theta (deg)")
    ax.set_title(f"Error (Pred - HFSS)\nRMSE={rmse_db:.3f} dB")
    plt.colorbar(im, ax=ax, label="dB", fraction=0.046, pad=0.04)

    ax_e = fig.add_subplot(gs[1, 0:2])
    ax_e.plot(theta_grid, matlab[:, peak_phi_idx], "b--", label="MATLAB", linewidth=1.5)
    ax_e.plot(theta_grid, predicted[:, peak_phi_idx], "r-", label="Predicted", linewidth=2)
    ax_e.plot(theta_grid, hfss[:, peak_phi_idx], "k-", label="HFSS", linewidth=1.5, alpha=0.7)
    ax_e.set_xlabel("Theta (deg)")
    ax_e.set_ylabel("Pattern (dB)")
    ax_e.set_title(f"E-plane cut (phi = {phi_grid[peak_phi_idx]:.1f} deg)")
    ax_e.legend(loc="lower left")
    ax_e.grid(True, alpha=0.3)
    ax_e.set_xlim(0, 180)

    ax_h = fig.add_subplot(gs[1, 2:4])
    ax_h.plot(phi_grid, matlab[peak_theta_idx, :], "b--", label="MATLAB", linewidth=1.5)
    ax_h.plot(phi_grid, predicted[peak_theta_idx, :], "r-", label="Predicted", linewidth=2)
    ax_h.plot(phi_grid, hfss[peak_theta_idx, :], "k-", label="HFSS", linewidth=1.5, alpha=0.7)
    ax_h.set_xlabel("Phi (deg)")
    ax_h.set_ylabel("Pattern (dB)")
    ax_h.set_title(f"H-plane cut (theta = {theta_grid[peak_theta_idx]:.1f} deg)")
    ax_h.legend(loc="lower left")
    ax_h.grid(True, alpha=0.3)
    ax_h.set_xlim(-180, 180)

    fig.suptitle(
        f"{column_id}  |  global_idx={global_idx}  |  bootstrap phase C @ 16x16",
        fontsize=15,
        y=0.98,
    )
    fig.subplots_adjust(top=0.90)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints/residual_bootstrap_phase_c/best_generator.pt"),
    )
    ap.add_argument(
        "--export-csv",
        type=str,
        default=str(PROJECT_ROOT / "results/residual_bootstrap_phase_c/top_samples_export.csv"),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
    )
    args = ap.parse_args()

    export_path = Path(args.export_csv)
    if not export_path.is_absolute():
        export_path = PROJECT_ROOT / export_path
    rows = _load_export_rows(export_path)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    global_indices = [int(r["global_idx"]) for r in rows]
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
    mat16 = pack["arr"][[pos[gi] for gi in global_indices]].astype(np.float32)

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
    print(f"Plotting {n} samples -> {out_path}", flush=True)

    G, recon = _load_g(ckpt, sigma_res)
    preds = _synth_scale(G, recon, mat16, a16, meta16, mean_mat, std_mat, 16)

    with PdfPages(out_path) as pdf:
        for i, r in enumerate(rows):
            col = r["export_column"]
            gi = int(r["global_idx"])
            rmse = float(r.get("rmse_db", np.sqrt(np.mean((preds[i] - hf16[i]) ** 2))))
            fig = _plot_sample_page(
                mat16[i], preds[i], hf16[i], col, gi, rmse, theta_grid, phi_grid
            )
            pdf.savefig(fig, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  page {i + 1}/{n}: {col} (global_idx={gi})", flush=True)

    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
