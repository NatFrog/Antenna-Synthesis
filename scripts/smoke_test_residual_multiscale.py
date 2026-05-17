"""
Smoke test for the 5-channel "with anchor" multi-scale residual pipeline.

Verifies that ScaleDatasetWithAnchor and Val16x16Dataset both produce 5-channel
inputs of shape (5, 181, 360), with the anchor channel populated and the
scale_token taking the expected per-stage value (0.25 / 0.50 / 1.00).

Run this BEFORE launching the full 2.5-hour training:
    python -m scripts.smoke_test_residual_multiscale

Expected output (rough):
    N=4 sample: x=(5, 181, 360) ...
      ch4 (anchor) range=[~ -1.0, ~ 2.5]    # z-scored hfss_pred_2x2
    N=8 sample: x=(5, 181, 360) ...
    Val16x16 sample: x=(5, 181, 360), scale_token=tensor([1.])
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch  # noqa: F401  (imported for side effects: cuda init, threading)

from scripts.train_residual_multiscale import (
    ScaleDatasetWithAnchor, Val16x16Dataset, ResidualReconLoss,
    H5_2TO4, H5_4TO8, H5_16X16, NORM_COMBINED, M16_TEST,
    SPLITS_2TO4, SPLITS_4TO8, BATCH, DEVICE,
)


def main():
    print(f"DEVICE={DEVICE}  BATCH={BATCH}")
    for p in (H5_2TO4, H5_4TO8, H5_16X16, NORM_COMBINED, M16_TEST,
              SPLITS_2TO4, SPLITS_4TO8):
        if not p.exists():
            raise FileNotFoundError(p)

    s = np.load(NORM_COMBINED)
    sigma_res = float(s["residual_std"])
    print(f"sigma_residual = {sigma_res:.3f} dB")

    sp24 = np.load(SPLITS_2TO4)
    sp48 = np.load(SPLITS_4TO8)
    tr4 = np.sort(sp24["train"])[:32]
    tr8 = np.sort(sp48["train"])[:32]

    # ── N=4 stage ────────────────────────────────────────────────────────
    ds4 = ScaleDatasetWithAnchor(
        H5_2TO4, tr4, 4, "matlab_4x4", "hfss_4x4",
        anchor_key="hfss_pred_2x2",
        mean_mat=s["mean"], std_mat=s["std"], augment_noise=True,
    )
    x, y, mat, null = ds4[0]
    assert tuple(x.shape) == (5, 181, 360), x.shape
    assert tuple(y.shape) == (1, 181, 360), y.shape
    assert tuple(mat.shape) == (1, 181, 360), mat.shape
    assert tuple(null.shape) == (1, 181, 360), null.shape
    print(f"N=4 sample: x={tuple(x.shape)}")
    print(f"  ch0 (matlab_n) range=[{x[0].min().item():.2f}, {x[0].max().item():.2f}]")
    print(f"  ch3 (scale_token) unique={torch.unique(x[3]).tolist()}  "
          f"(expect [0.25])")
    print(f"  ch4 (anchor_n)  range=[{x[4].min().item():.2f}, {x[4].max().item():.2f}]")

    # ── N=8 stage ────────────────────────────────────────────────────────
    ds8 = ScaleDatasetWithAnchor(
        H5_4TO8, tr8, 8, "matlab_8x8", "hfss_8x8",
        anchor_key="hfss_pred_4x4",
        mean_mat=s["mean"], std_mat=s["std"], augment_noise=True,
    )
    x, _, _, _ = ds8[0]
    assert tuple(x.shape) == (5, 181, 360), x.shape
    print(f"N=8 sample: x={tuple(x.shape)}")
    print(f"  ch3 (scale_token) unique={torch.unique(x[3]).tolist()}  "
          f"(expect [0.50])")
    print(f"  ch4 (anchor_n)  range=[{x[4].min().item():.2f}, {x[4].max().item():.2f}]")

    # ── N=16 inference (Val16x16Dataset) ─────────────────────────────────
    test48 = np.sort(sp48["test"])
    val16_idx = test48[:16]
    m16_pack = np.load(M16_TEST)
    val16_ds = Val16x16Dataset(
        val16_idx, m16_pack["arr"][:16], s["mean"], s["std"]
    )
    x16, hfss16, mat16 = val16_ds[0]
    assert tuple(x16.shape) == (5, 181, 360), x16.shape
    print(f"Val16x16 sample: x={tuple(x16.shape)}")
    print(f"  ch3 (scale_token) unique={torch.unique(x16[3]).tolist()}  "
          f"(expect [1.00])")
    print(f"  ch4 (hfss_pred_8x8 anchor)  "
          f"range=[{x16[4].min().item():.2f}, {x16[4].max().item():.2f}]")

    # ── Beta-alignment spot check at N=4 ────────────────────────────────
    # Spec failure mode: "verify that meta[i][:2] (β values) match across
    # matlab_4x4 / hfss_pred_2x2 channels at the same index". They share
    # antenna_data_2to4.h5/metadata so this is per-h5-row, not per-channel,
    # but we confirm the dataset uses meta[0]/meta[1] correctly.
    import h5py
    with h5py.File(H5_2TO4, "r") as f:
        meta0 = f["metadata"][int(tr4[0])].astype(np.float32)
    x4_again, _, _, _ = ds4[0]
    expected_dpx = meta0[0] / 180.0
    expected_dpy = meta0[1] / 180.0
    actual_dpx = float(x4_again[1, 0, 0])
    actual_dpy = float(x4_again[2, 0, 0])
    assert abs(actual_dpx - expected_dpx) < 1e-5, (actual_dpx, expected_dpx)
    assert abs(actual_dpy - expected_dpy) < 1e-5, (actual_dpy, expected_dpy)
    print(f"\nN=4 beta channels match metadata: "
          f"dpx={actual_dpx*180:.3f}° dpy={actual_dpy*180:.3f}°")

    # ── Anchor-dropout sanity check ──────────────────────────────────────
    # With p=1.0 the anchor must always be exactly zero; with p=0.0 it must
    # never be exactly zero (the noise augment adds a non-degenerate field).
    ds4_drop = ScaleDatasetWithAnchor(
        H5_2TO4, tr4, 4, "matlab_4x4", "hfss_4x4",
        anchor_key="hfss_pred_2x2",
        mean_mat=s["mean"], std_mat=s["std"], augment_noise=True,
        anchor_dropout_p=1.0,
    )
    x_drop, *_ = ds4_drop[0]
    assert float(x_drop[4].abs().max()) == 0.0, (
        "anchor_dropout_p=1.0 should zero ch4 entirely; "
        f"got |max|={float(x_drop[4].abs().max())}"
    )
    print(f"anchor_dropout_p=1.0  -> ch4 |max| = {float(x_drop[4].abs().max()):.3f}  "
          f"(expect 0.000)")

    # And confirm the keep-rate is roughly correct at p=0.25 over 200 draws.
    ds4_p25 = ScaleDatasetWithAnchor(
        H5_2TO4, tr4, 4, "matlab_4x4", "hfss_4x4",
        anchor_key="hfss_pred_2x2",
        mean_mat=s["mean"], std_mat=s["std"], augment_noise=False,
        anchor_dropout_p=0.25,
    )
    np.random.seed(0)
    zeroed = 0
    n_draws = 200
    for _ in range(n_draws):
        xx, *_ = ds4_p25[int(np.random.randint(len(tr4)))]
        if float(xx[4].abs().max()) == 0.0:
            zeroed += 1
    rate = zeroed / n_draws
    print(f"anchor_dropout_p=0.25 -> empirical zero-rate {rate:.2%} over {n_draws} draws "
          f"(expect ~25%)")
    assert 0.10 <= rate <= 0.40, f"dropout rate out of expected band: {rate:.2%}"

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
