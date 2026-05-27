"""Quick probe: what RMSE can pure-physics AEP recomposition achieve if we
substitute the TRUE 4x4 HFSS in place of the model prediction?

This is an oracle baseline for the AEP path: it isolates how much error is
coming from `(AEP recovery + 8x8 array factor)` versus how much is coming
from the 4x4 model's coupling prediction.

Reports both full-pattern RMSE and masked RMSE on pixels where the truth is
above selected dB floors (default -40 dB and -50 dB), which is the user's
primary scoring region.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np

from src.config import PROCESSED_DIR
from src.training.metrics import (
    compute_batch_hfss_region_metrics,
    compute_batch_metrics,
)
from scripts.infer_resunet_4x4_aep_8x8 import (
    aep_reconstruct_8x8,
    af_complex_batch,
    build_grids,
    peak_norm_db,
    precompute_position_phase,
)


def main() -> None:
    n = 1000
    print(f"Loading {n} samples ...", flush=True)
    with h5py.File(PROCESSED_DIR / "antenna_data_4x4_subblock.h5", "r") as fs:
        sb4 = fs["subblock_4x4"][:n].astype(np.float32)
        cpl4 = fs["coupling_4x4"][:n].astype(np.float32)
    sb_mean = sb4.mean(axis=1)
    sb_mean = sb_mean - sb_mean.max(axis=(-2, -1), keepdims=True)
    hfss4_real = peak_norm_db(sb_mean + cpl4)

    with h5py.File(PROCESSED_DIR / "antenna_data_8x8.h5", "r") as f8:
        hfss8 = peak_norm_db(f8["hfss_patterns"][:n].astype(np.float32))
        dpx = f8["metadata"][:n, 0].astype(np.float32)
        dpy = f8["metadata"][:n, 1].astype(np.float32)

    npz = np.load(PROCESSED_DIR / "subblock_compositions.npz")
    sb8 = peak_norm_db(npz["sub_block_8x8"][:n].astype(np.float32))

    print("Computing array factors ...", flush=True)
    TH, PH = build_grids()
    pos4 = precompute_position_phase(TH, PH, 4)
    pos8 = precompute_position_phase(TH, PH, 8)
    af4 = af_complex_batch(dpx, dpy, pos4, parent=4)
    af8 = af_complex_batch(dpx, dpy, pos8, parent=8)

    print("AEP (using TRUE 4x4 HFSS) ...", flush=True)
    pred_oracle = aep_reconstruct_8x8(hfss4_real, af4, af8)

    methods = {
        "baseline_sb8x8": sb8,
        "matlab_8x8": peak_norm_db(npz["sub_block_4x4"][:n].astype(np.float32)),
        "oracle_AEP_from_true_4x4": pred_oracle,
    }
    print()
    for name, pred in methods.items():
        m_full = compute_batch_metrics(pred, hfss8)
        m_mask = compute_batch_hfss_region_metrics(pred, hfss8, db_floors=(-40.0, -30.0, -50.0))
        print(f"=== {name} ===")
        print(
            f"  full       rmse={m_full['rmse_db']:.3f}  mae={m_full['mae_db']:.3f}  "
            f"r={m_full['pearson_r']:.4f}  ssim={m_full['ssim']:.4f}"
        )
        for floor in (-30, -40, -50):
            tag = f"{floor}".replace("-", "neg")
            mean_rmse = m_mask.get(f"paper_region_rmse_db_{tag}_mean")
            pooled_rmse = m_mask.get(f"paper_region_rmse_db_{tag}_pooled")
            frac = m_mask.get(f"paper_region_frac_sphere_{tag}")
            print(
                f"  > {floor} dB  mean_rmse={mean_rmse:.3f}  pooled_rmse={pooled_rmse:.3f}  "
                f"frac={frac:.4f}"
            )
        print()


if __name__ == "__main__":
    main()
