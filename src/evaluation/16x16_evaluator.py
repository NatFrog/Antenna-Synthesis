"""
Shared 16×16 residual multi-scale evaluation: inference, sanity checks, metrics.

Assumes patterns are max-normalised to 0 dB peak and the split protocol:
``test48 = sort(split_indices_4to8['test'])``, ``matlab_16x16_test['test_idx'] == test48``,
first 100 rows of ``matlab_16x16_test['arr']`` = val selection, last 100 = held-out test.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
from scipy.ndimage import minimum_filter
from tqdm import tqdm

from src.config import BATCH_SIZE, DEVICE, NULL_THRESHOLD_DB
from src.evaluation.antenna_metrics import compute_antenna_metric_errors
from src.training.metrics import (
    compute_batch_hfss_region_metrics,
    compute_batch_metrics,
)

PAPER_DB_FLOORS = (-40.0, -50.0)
AMP_DEVICE_T = "cuda" if DEVICE.type == "cuda" else "cpu"


def verify_matlab_16x16_indexing(pack_test_idx: np.ndarray, test48_sorted: np.ndarray) -> None:
    if not np.array_equal(pack_test_idx, test48_sorted):
        raise RuntimeError(
            "matlab_16x16_test.npz test_idx must equal np.sort(split_indices_4to8['test'])."
        )


def verify_max_normalised_hfss(hfss: np.ndarray, peak_tol: float = 0.02) -> None:
    """HFSS patterns should be per-sample max-normalised (peak ≈ 0 dB)."""
    bad = []
    for i in range(len(hfss)):
        mx = float(np.max(hfss[i]))
        if mx > peak_tol or mx < -peak_tol:
            bad.append((i, mx))
    if bad:
        raise RuntimeError(
            f"Expected HFSS peak ≈ 0 dB per sample (tol {peak_tol} dB); "
            f"found {len(bad)} violations, e.g. index {bad[0][0]} max={bad[0][1]:.4f}"
        )


def infer_residual_16x16(
    G: torch.nn.Module,
    recon_loss: Any,
    mat16: np.ndarray,
    anchor16: np.ndarray,
    meta: np.ndarray,
    mean_mat: np.ndarray,
    std_mat: np.ndarray,
    batch_size: int | None = None,
    use_bf16_autocast: bool = True,
) -> np.ndarray:
    """
    Run generator + recon_loss.compose; matches training when AMP uses bf16 on CUDA.

    ``recon_loss`` must implement ``compose(delta_n, matlab_dB)`` like training.
    """
    bs = batch_size if batch_size is not None else BATCH_SIZE
    n = len(mat16)
    mean_b = mean_mat.astype(np.float32)
    std_b = np.maximum(std_mat.astype(np.float32), 1e-6)
    preds = np.zeros_like(mat16, dtype=np.float32)
    amp_on = bool(use_bf16_autocast and DEVICE.type == "cuda")

    G.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n, bs), desc="inference"):
            j = min(i + bs, n)
            mat_batch = mat16[i:j]
            anchor_batch = anchor16[i:j]
            mat_n = (mat_batch - mean_b[None]) / std_b[None]
            anchor_n = (anchor_batch - mean_b[None]) / std_b[None]
            dpx = (meta[i:j, 0:1, None] / 180.0).astype(np.float32) * np.ones_like(mat_n)
            dpy = (meta[i:j, 1:2, None] / 180.0).astype(np.float32) * np.ones_like(mat_n)
            stk = np.full_like(mat_n, fill_value=1.0, dtype=np.float32)
            x = np.stack([mat_n, dpx, dpy, stk, anchor_n], axis=1)
            xt = torch.from_numpy(x).to(DEVICE)
            mt = torch.from_numpy(mat_batch[:, None]).to(DEVICE)
            if amp_on:
                with torch.amp.autocast(
                    device_type=AMP_DEVICE_T, dtype=torch.bfloat16, enabled=True
                ):
                    delta_n = G(xt)
                    pred_dB = recon_loss.compose(delta_n, mt)
            else:
                delta_n = G(xt)
                pred_dB = recon_loss.compose(delta_n, mt)
            preds[i:j] = pred_dB.float().cpu().numpy()[:, 0]
    return preds


def build_residual_16x16_metrics_dict(
    preds: np.ndarray,
    truth_hfss: np.ndarray,
    mat16: np.ndarray,
    db_floors: Tuple[float, ...] = PAPER_DB_FLOORS,
) -> Dict[str, float]:
    """All scalar metrics for metrics.txt (pattern, antenna, nulls, paper regions)."""
    pm = compute_batch_metrics(preds, truth_hfss)

    ae = []
    for i in range(len(preds)):
        try:
            ae.append(compute_antenna_metric_errors(preds[i], truth_hfss[i]))
        except Exception:
            pass
    avg_ant: Dict[str, float] = {}
    if ae:
        for key in ae[0]:
            vals = [e[key] for e in ae if not np.isnan(e[key])]
            avg_ant[key] = float(np.mean(vals)) if vals else float("nan")

    null_rmses, depth_errs = [], []
    nfa = nft = 0
    for i in range(len(preds)):
        mat = mat16[i]
        mask_null = mat < (mat.max() + NULL_THRESHOLD_DB)
        if mask_null.sum() > 0:
            null_rmses.append(
                float(
                    np.sqrt(np.mean((preds[i][mask_null] - truth_hfss[i][mask_null]) ** 2))
                )
            )
            pf = preds[i][mask_null] - mat[mask_null]
            tf = truth_hfss[i][mask_null] - mat[mask_null]
            nfa += int((np.abs(pf - tf) < 2.0).sum())
            nft += int(mask_null.sum())
        lm = minimum_filter(mat, size=5)
        ilm = (mat == lm) & mask_null
        if ilm.sum() > 0:
            mp = np.argwhere(ilm)
            mv = mat[ilm]
            di = np.argsort(mv)[:10]
            for k in di:
                t, p = mp[k]
                depth_errs.append(abs(preds[i][t, p] - truth_hfss[i][t, p]))
    nm = {
        "rmse_at_nulls_db": float(np.mean(null_rmses)) if null_rmses else float("nan"),
        "null_depth_error_db": float(np.mean(depth_errs)) if depth_errs else float("nan"),
        "null_fill_accuracy_pct": float(nfa / max(nft, 1) * 100),
    }

    paper = compute_batch_hfss_region_metrics(preds, truth_hfss, db_floors=db_floors)

    return {
        **{f"pattern_{k}": v for k, v in pm.items()},
        **{f"antenna_{k}": v for k, v in avg_ant.items()},
        **{f"null_{k}": v for k, v in nm.items()},
        **paper,
    }


def print_residual_eval_report(title: str, metrics: Dict[str, float]) -> None:
    """Pretty-print groups written by ``build_residual_16x16_metrics_dict``."""
    print(f"\n{'='*60}\n{title}\n{'='*60}")

    def _fmt(v: float) -> str:
        return f"{v:.6f}" if np.isfinite(v) else "nan"

    def _block(label: str, prefix: str) -> None:
        keys = sorted(k for k in metrics if k.startswith(prefix))
        if not keys:
            return
        print(f"\n--- {label} ---")
        for k in keys:
            print(f"  {k}: {_fmt(metrics[k])}")

    _block("Pattern metrics (full sphere)", "pattern_")
    _block("Antenna metrics", "antenna_")
    _block("Null metrics (matlab_16x16 -20 dB null definition)", "null_")
    print(
        "\n--- Paper-style regions (GT HFSS > floor; excludes deep nulls / noise-dominated) ---"
    )
    for floor in PAPER_DB_FLOORS:
        tag = f"{int(round(floor))}".replace("-", "neg")
        print(f"  Floor {floor:.0f} dB (mask: truth_hfss > {floor:.0f}):")
        for suffix in (
            f"paper_region_rmse_db_{tag}_mean",
            f"paper_region_rmse_db_{tag}_pooled",
            f"paper_region_mae_db_{tag}_mean",
            f"paper_region_mae_db_{tag}_pooled",
            f"paper_region_frac_sphere_{tag}",
        ):
            if suffix in metrics:
                print(f"    {suffix}: {_fmt(metrics[suffix])}")
