"""Shared stage-1 6×6 helpers (matlab-relative residual target)."""

from __future__ import annotations

import numpy as np
import torch

from src.training.metrics import rmse


def residual_target(hfss: np.ndarray, matlab: np.ndarray) -> np.ndarray:
    """Training target: HFSS correction relative to analytical matlab baseline."""
    return (hfss - matlab).astype(np.float32)


@torch.no_grad()
def evaluate_composed_matlab_rel(
    model,
    ema,
    loader,
    res_mean: torch.Tensor,
    res_std: torch.Tensor,
    m_mean: torch.Tensor,
    m_std: torch.Tensor,
    sb_mean: torch.Tensor,
    sb_std: torch.Tensor,
    *,
    amp_device_type: str,
    use_amp: bool,
    device: torch.device,
    pin_memory: bool,
    val_weight_main: float,
    val_weight_main_sl: float,
    val_weight_null: float,
    gate_floor: float | None = None,
) -> dict[str, float]:
    """
    Validation reconstruction: matlab + pred_residual vs HFSS (matlab + true_residual).

    Main-beam / null masks are still derived from the sub-block peak geometry.
    """
    live = ema.swap_into(model)
    if gate_floor is not None and hasattr(model, "gate_floor"):
        model.gate_floor = gate_floor
    try:
        model.eval()
        main_err, main_sl_err, null_err, all_err = [], [], [], []
        for batch in loader:
            x = batch[0].to(device, non_blocking=pin_memory)
            y = batch[1].to(device, non_blocking=pin_memory)
            null_mask = batch[2].to(device, non_blocking=pin_memory)
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                pred_n = model(x)
            pred_db = pred_n * res_std + res_mean
            tgt_db = y * res_std + res_mean
            m_db = x[:, 0:1] * m_std + m_mean
            sb_db = x[:, 1:2] * sb_std + sb_mean
            composed_pred = m_db + pred_db
            composed_true = m_db + tgt_db
            err = (composed_pred - composed_true).float()

            sb_peak = sb_db.amax(dim=(-2, -1), keepdim=True)
            main = composed_true >= (sb_peak - 3.0)
            main_sl = composed_true >= (sb_peak - 20.0)
            for i in range(err.shape[0]):
                e = err[i, 0].cpu().numpy()
                t = composed_true[i, 0].cpu().numpy()
                p = t + e
                nm = null_mask[i, 0].cpu().numpy() > 0.5
                mb = main[i, 0].cpu().numpy()
                ms = main_sl[i, 0].cpu().numpy()
                if mb.any():
                    main_err.append(rmse(p[mb], t[mb]))
                if ms.any():
                    main_sl_err.append(rmse(p[ms], t[ms]))
                if nm.any():
                    null_err.append(rmse(p[nm], t[nm]))
                all_err.append(rmse(p.ravel(), t.ravel()))

        main_rmse = float(np.mean(main_err)) if main_err else float("nan")
        main_sl_rmse = float(np.mean(main_sl_err)) if main_sl_err else float("nan")
        null_rmse = float(np.mean(null_err)) if null_err else float("nan")
        full_rmse = float(np.mean(all_err)) if all_err else float("nan")
        score = (
            val_weight_main * main_rmse
            + val_weight_main_sl * main_sl_rmse
            + val_weight_null * null_rmse
        )
        return {
            "score": score,
            "main_rmse": main_rmse,
            "main_sl_rmse": main_sl_rmse,
            "null_rmse": null_rmse,
            "full_rmse": full_rmse,
        }
    finally:
        ema.restore(model, live)
