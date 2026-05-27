"""Evaluation metrics for antenna pattern prediction."""

import numpy as np
import torch
from typing import Dict, Sequence


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error in dB."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error in dB."""
    return float(np.mean(np.abs(pred - target)))


def max_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Maximum absolute error in dB."""
    return float(np.max(np.abs(pred - target)))


def pearson_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation coefficient (flattened)."""
    p = pred.flatten()
    t = target.flatten()
    return float(np.corrcoef(p, t)[0, 1])


def ssim_2d(pred: np.ndarray, target: np.ndarray, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """
    Structural Similarity Index (simplified, single-scale).

    Computed over the full 2D pattern.
    """
    mu_p = pred.mean()
    mu_t = target.mean()
    sigma_p = pred.std()
    sigma_t = target.std()
    sigma_pt = np.mean((pred - mu_p) * (target - mu_t))

    numerator = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
    denominator = (mu_p**2 + mu_t**2 + C1) * (sigma_p**2 + sigma_t**2 + C2)

    return float(numerator / denominator)


def compute_pattern_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute all pattern-level metrics.

    Args:
        pred: predicted pattern (N_THETA, N_PHI) in dB
        target: ground truth pattern (N_THETA, N_PHI) in dB

    Returns:
        dict of metric_name -> value
    """
    return {
        'rmse_db': rmse(pred, target),
        'mae_db': mae(pred, target),
        'max_error_db': max_error(pred, target),
        'pearson_r': pearson_correlation(pred, target),
        'ssim': ssim_2d(pred, target),
    }


def compute_batch_metrics(
    preds: np.ndarray, targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute average metrics over a batch of patterns.

    Args:
        preds: (batch, N_THETA, N_PHI) in dB
        targets: (batch, N_THETA, N_PHI) in dB

    Returns:
        dict of averaged metrics
    """
    all_metrics = []
    for i in range(len(preds)):
        all_metrics.append(compute_pattern_metrics(preds[i], targets[i]))

    # Average each metric
    avg = {}
    for key in all_metrics[0]:
        avg[key] = float(np.mean([m[key] for m in all_metrics]))

    return avg


def compute_batch_hfss_region_metrics(
    preds: np.ndarray,
    truth_hfss: np.ndarray,
    db_floors: Sequence[float] = (-40.0, -50.0),
) -> Dict[str, float]:
    """Pattern errors on pixels where ground-truth HFSS exceeds each dB floor.

    Masks use **truth_hfss > floor** (e.g. main beam and sidelobes, not deep nulls).
    For each floor, reports per-sample mean RMSE/MAE and pooled RMSE/MAE over all
    masked pixels in the batch.
    """
    if preds.shape != truth_hfss.shape:
        raise ValueError(f"shape mismatch: preds {preds.shape} vs truth {truth_hfss.shape}")
    n_pix = int(preds[0].size)
    out: Dict[str, float] = {}
    for floor in db_floors:
        tag = f"{int(round(floor))}".replace("-", "neg")
        per_rmse, per_mae = [], []
        pooled_sq: list[float] = []
        pooled_abs: list[float] = []
        fracs: list[float] = []
        for i in range(len(preds)):
            mask = truth_hfss[i] > float(floor)
            n_m = int(mask.sum())
            if n_m == 0:
                continue
            err = preds[i][mask] - truth_hfss[i][mask]
            per_rmse.append(float(np.sqrt(np.mean(err ** 2))))
            per_mae.append(float(np.mean(np.abs(err))))
            pooled_sq.extend((err ** 2).tolist())
            pooled_abs.extend(np.abs(err).tolist())
            fracs.append(n_m / n_pix)
        out[f"paper_region_rmse_db_{tag}_mean"] = (
            float(np.mean(per_rmse)) if per_rmse else float("nan")
        )
        out[f"paper_region_rmse_db_{tag}_pooled"] = (
            float(np.sqrt(np.mean(pooled_sq))) if pooled_sq else float("nan")
        )
        out[f"paper_region_mae_db_{tag}_mean"] = (
            float(np.mean(per_mae)) if per_mae else float("nan")
        )
        out[f"paper_region_mae_db_{tag}_pooled"] = (
            float(np.mean(pooled_abs)) if pooled_abs else float("nan")
        )
        out[f"paper_region_frac_sphere_{tag}"] = (
            float(np.mean(fracs)) if fracs else float("nan")
        )
    return out


# ── Torch versions for use during training ──────────────────────────────────

def rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))
