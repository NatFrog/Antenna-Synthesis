"""
Zero-shot evaluation of the dual-scale coupling model at 8×8 and 16×16.

Uses phase-aware sub-block composition and the zeroshot eval cache:

  processed/phase_aware_8x8_compose.npz
  processed/phase_aware_16x16_compose.npz
  processed/zeroshot_eval_cache_8x8.npz
  processed/zeroshot_eval_cache_16x16.npz

Reconstruction (compose-deficit target):
  pred = sub_block + α_c · shield(δ_coupling) + α_s · δ_scale

  α_c, α_s — LS fit on train split at target scale (HFSS − sub_block ground truth).
  scale_token = log(N) / log(6) fed to the scale head.

Prep once:
    python -m scripts.prep_phase_aware_compose
    python -m scripts.prep_zeroshot_eval_cache

Usage:
    python -m scripts.evaluate_dual_residual_zeroshot
    python -m scripts.evaluate_dual_residual_zeroshot --scale 8
    python -m scripts.evaluate_dual_residual_zeroshot --no-alpha --no-shield
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.config import (
    CHECKPOINTS_DIR,
    DEVICE,
    NULL_THRESHOLD_DB,
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.evaluation.comprehensive_metrics import (
    evaluate_predictions,
    pack_metrics_npz,
    print_metrics_block,
    write_metrics_txt,
)
from src.evaluation.visualization import (
    plot_error_distribution,
    plot_pattern_comparison,
    plot_plane_cuts,
    plot_scatter_pred_vs_true,
)
from src.evaluation.zeroshot_varalpha_report import write_masked_comparison_txt
from scripts.stage1_6x6_fingerprint import (
    fingerprint_channel,
    load_sub6_concatenated,
    pair_sub6_indices,
)
from scripts.train_cgan_2to4_fusion_no_m4 import ATTN_HEADS, GEN_BASE
from src.models.dual_residual import DualScaleCouplingModel, scale_token_from_n
from src.models.multihead_v4 import TOP_K_NULLS, fit_ls_alpha, shield_main_beam_residual

CKPT = CHECKPOINTS_DIR / "dual_residual" / "best_model.pt"
NORM_DR = PROCESSED_DIR / "norm_stats_dual_residual.npz"
PA_8 = PROCESSED_DIR / "phase_aware_8x8_compose.npz"
PA_16 = PROCESSED_DIR / "phase_aware_16x16_compose.npz"
CACHE_8 = PROCESSED_DIR / "zeroshot_eval_cache_8x8.npz"
CACHE_16 = PROCESSED_DIR / "zeroshot_eval_cache_16x16.npz"
SB6_DIR = PROJECT_ROOT / "datasets_6x6sub-block_hfss"
OUT_ROOT = PROJECT_ROOT / "results" / "dual_residual_zeroshot"
INFER_BATCH = 8
N_CUTS = 10

NTH, NPH = 181, 360


def _peaknorm(x: np.ndarray) -> np.ndarray:
    return (x - x.max(axis=(1, 2), keepdims=True)).astype(np.float32)


def _pixel_stats(arr: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = arr[train_idx].astype(np.float32)
    return s.mean(0), np.maximum(s.std(0), 1e-6).astype(np.float32)


def _gap_stats(
    matlab: np.ndarray,
    sub_block: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gap = (sub_block[train_idx] - matlab[train_idx]).astype(np.float32)
    return gap.mean(0), np.maximum(gap.std(0), 1e-6).astype(np.float32)


def _make_split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    tr = np.sort(perm[:n_train])
    va = np.sort(perm[n_train : n_train + n_val])
    te = np.sort(perm[n_train + n_val :])
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def _cache_path(n_elem: int) -> Path:
    return CACHE_8 if n_elem == 8 else CACHE_16


def _steering_matches(pa: np.lib.npyio.NpzFile, cache: np.lib.npyio.NpzFile) -> bool:
    return (
        len(pa["ids"]) == len(cache["ids"])
        and np.array_equal(pa["ids"], cache["ids"])
        and np.allclose(pa["dpx"], cache["dpx"], atol=1e-4)
        and np.allclose(pa["dpy"], cache["dpy"], atol=1e-4)
    )


def _cache_has_fingerprint(path: Path) -> bool:
    if not path.exists():
        return False
    with np.load(path) as z:
        return "fingerprint" in z.files


def _build_fingerprint(
    matlab: np.ndarray,
    dpx: np.ndarray,
    dpy: np.ndarray,
    sub6_all: np.ndarray,
    dpx6: np.ndarray,
    dpy6: np.ndarray,
) -> np.ndarray:
    pair_idx = pair_sub6_indices(dpx, dpy, dpx6, dpy6, strict=False)
    n_missing = int((pair_idx < 0).sum())
    if n_missing:
        print(f"  warning: {n_missing} steerings have no 6x6 sub-block match", flush=True)
    fingerprint = np.empty((len(dpx), NTH, NPH), dtype=np.float32)
    for i in range(len(dpx)):
        j = int(pair_idx[i])
        fingerprint[i] = fingerprint_channel(matlab[i], sub6_all[j]) if j >= 0 else 0.0
    return fingerprint


def load_model(stats: dict) -> DualScaleCouplingModel:
    if not CKPT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CKPT}\n"
            "Train first: python -m scripts.train_dual_residual --init-v4"
        )
    model = DualScaleCouplingModel(
        base=GEN_BASE,
        attn_heads=ATTN_HEADS,
        top_k=TOP_K_NULLS,
        sb_mean=stats["sub_block_mean"],
        sb_std=stats["sub_block_std"],
        res_mean=stats["compose_deficit_mean"],
        res_std=stats["compose_deficit_std"],
        scale_std=stats["scale_residual_std"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def run_inference(
    model: DualScaleCouplingModel,
    stats: dict,
    *,
    matlab: np.ndarray,
    sub_block: np.ndarray,
    fingerprint: np.ndarray,
    dpx: np.ndarray,
    dpy: np.ndarray,
    m_mean: np.ndarray,
    m_std: np.ndarray,
    sb_mean: np.ndarray,
    sb_std: np.ndarray,
    gap_mean: np.ndarray,
    gap_std: np.ndarray,
    n_elem: int,
    indices: np.ndarray,
    apply_shield: bool,
    batch: int = INFER_BATCH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns denormalised δ_coupling and δ_scale (dB), each (len(indices), H, W).
    """
    fp_mean = stats["fingerprint_mean"]
    fp_std = stats["fingerprint_std"]
    c_mean = stats["compose_deficit_mean"].astype(np.float32)
    c_std = stats["compose_deficit_std"].astype(np.float32)
    s_mean = stats["scale_residual_mean"].astype(np.float32)
    s_std = stats["scale_residual_std"].astype(np.float32)
    scale_tok = scale_token_from_n(n_elem)

    n = len(indices)
    pred_c = np.empty((n, NTH, NPH), dtype=np.float32)
    pred_s = np.empty((n, NTH, NPH), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch):
            end = min(start + batch, n)
            xs = []
            for k in range(start, end):
                s = int(indices[k])
                gap = (sub_block[s] - matlab[s] - gap_mean) / gap_std
                xs.append(np.stack([
                    (matlab[s] - m_mean) / m_std,
                    (sub_block[s] - sb_mean) / sb_std,
                    (fingerprint[s] - fp_mean) / fp_std,
                    np.full((NTH, NPH), dpx[s] / 180.0, np.float32),
                    np.full((NTH, NPH), dpy[s] / 180.0, np.float32),
                    gap.astype(np.float32),
                ], axis=0))
            xt = torch.from_numpy(np.stack(xs).astype(np.float32)).to(DEVICE)
            st = torch.full((end - start, 1), scale_tok, dtype=torch.float32, device=DEVICE)
            out = model(xt, st, return_aux=True)
            dc_n = out["residual_coupling"].cpu().numpy()
            ds_n = out["residual_scale"].cpu().numpy()
            for j in range(end - start):
                s_idx = int(indices[start + j])
                dc = dc_n[j, 0] * c_std + c_mean
                ds = ds_n[j, 0] * s_std + s_mean
                if apply_shield:
                    dc = shield_main_beam_residual(dc, sub_block[s_idx])
                pred_c[start + j] = dc
                pred_s[start + j] = ds
            if start == 0 or end == n:
                print(f"    inferred {end}/{n}", flush=True)
    return pred_c, pred_s


def reconstruct(
    sub_block: np.ndarray,
    delta_c: np.ndarray,
    delta_s: np.ndarray,
    *,
    alpha_c: float,
    alpha_s: float,
) -> np.ndarray:
    return (sub_block + alpha_c * delta_c + alpha_s * delta_s).astype(np.float32)


def _load_phase_aware_scale(
    pa_path: Path,
    n_elem: int,
    sub6_all: np.ndarray | None,
    dpx6: np.ndarray | None,
    dpy6: np.ndarray | None,
) -> dict:
    if not pa_path.exists():
        raise FileNotFoundError(
            f"{pa_path} missing — run: python -m scripts.prep_phase_aware_compose"
        )
    pa = np.load(pa_path)
    n = len(pa["ids"])
    dpx = pa["dpx"].astype(np.float32)
    dpy = pa["dpy"].astype(np.float32)
    sub_block = pa["composed_dB"].astype(np.float32)
    hfss = pa["truth_dB"].astype(np.float32)
    ids = pa["ids"].astype(np.int64)
    theta_deg = pa["theta"].astype(np.float32) if "theta" in pa else None
    phi_deg = pa["phi"].astype(np.float32) if "phi" in pa else None

    cache_path = _cache_path(n_elem)
    if cache_path.exists():
        cache = np.load(cache_path)
        if int(cache["n_elem"]) == n_elem and _steering_matches(pa, cache):
            print(f"  loading cached matlab/fingerprint from {cache_path.name}", flush=True)
            matlab = cache["matlab_dB"].astype(np.float32)
            if theta_deg is None and "theta" in cache:
                theta_deg = cache["theta"].astype(np.float32)
            if phi_deg is None and "phi" in cache:
                phi_deg = cache["phi"].astype(np.float32)
            if "fingerprint" in cache:
                fingerprint = cache["fingerprint"].astype(np.float32)
            elif sub6_all is not None and dpx6 is not None and dpy6 is not None:
                print("  cache has matlab only — building fingerprint ...", flush=True)
                fingerprint = _build_fingerprint(matlab, dpx, dpy, sub6_all, dpx6, dpy6)
            else:
                raise FileNotFoundError(
                    f"{cache_path.name} has no fingerprint; rerun:\n"
                    "  python -m scripts.prep_zeroshot_eval_cache"
                )
        else:
            raise RuntimeError(
                f"{cache_path.name} is stale (steering mismatch with {pa_path.name}). "
                "Regenerate: python -m scripts.prep_zeroshot_eval_cache"
            )
    else:
        raise FileNotFoundError(
            f"{cache_path.name} missing — run once before evaluation:\n"
            "  python -m scripts.prep_zeroshot_eval_cache"
        )

    tr, va, te = _make_split(n)
    return dict(
        scale=f"{n_elem}x{n_elem}",
        n_elem=n_elem,
        matlab=matlab,
        sub_block=sub_block,
        hfss=hfss,
        fingerprint=fingerprint,
        dpx=dpx,
        dpy=dpy,
        ids=ids,
        train_idx=tr,
        test_idx=te,
        val_idx=va,
        pa_path=pa_path,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
    )


def eval_scale(
    model: DualScaleCouplingModel,
    stats: dict,
    data: dict,
    *,
    use_alpha: bool,
    apply_shield: bool,
    out_dir: Path,
    write_plots: bool,
) -> None:
    scale = data["scale"]
    n_elem = data["n_elem"]
    tr, te = data["train_idx"], data["test_idx"]
    matlab = data["matlab"]
    sub_block = data["sub_block"]
    hfss = data["hfss"]
    fingerprint = data["fingerprint"]
    dpx, dpy = data["dpx"], data["dpy"]

    m_mean, m_std = _pixel_stats(matlab, tr)
    sb_mean, sb_std = _pixel_stats(sub_block, tr)
    gap_mean, gap_std = _gap_stats(matlab, sub_block, tr)
    scale_tok = scale_token_from_n(n_elem)

    print(f"\n{'=' * 100}", flush=True)
    print(
        f"{scale} ZERO-SHOT — dual_residual  "
        f"(train n={len(tr)}, test n={len(te)}, scale_token={scale_tok:.4f})",
        flush=True,
    )
    print(f"  sub_block: {data['pa_path'].name} (phase-aware composed_dB)", flush=True)
    print(f"  shield={apply_shield}  alpha_fit={use_alpha}", flush=True)
    print(f"{'=' * 100}", flush=True)

    dc_tr, ds_tr = run_inference(
        model, stats,
        matlab=matlab, sub_block=sub_block, fingerprint=fingerprint,
        dpx=dpx, dpy=dpy,
        m_mean=m_mean, m_std=m_std, sb_mean=sb_mean, sb_std=sb_std,
        gap_mean=gap_mean, gap_std=gap_std,
        n_elem=n_elem, indices=tr, apply_shield=apply_shield,
    )
    true_def_tr = (hfss[tr] - sub_block[tr]).astype(np.float32)

    alpha_c = 1.0
    alpha_s = 1.0
    if use_alpha:
        alpha_c = fit_ls_alpha(true_def_tr, dc_tr)
        residual_after_c = true_def_tr - alpha_c * dc_tr
        alpha_s = fit_ls_alpha(residual_after_c, ds_tr)
        print(f"  LS alpha_coupling (train): {alpha_c:.4f}", flush=True)
        print(f"  LS alpha_scale    (train): {alpha_s:.4f}", flush=True)

    dc_te, ds_te = run_inference(
        model, stats,
        matlab=matlab, sub_block=sub_block, fingerprint=fingerprint,
        dpx=dpx, dpy=dpy,
        m_mean=m_mean, m_std=m_std, sb_mean=sb_mean, sb_std=sb_std,
        gap_mean=gap_mean, gap_std=gap_std,
        n_elem=n_elem, indices=te, apply_shield=apply_shield,
    )

    pred_raw_te = reconstruct(
        sub_block[te], dc_te, ds_te, alpha_c=alpha_c, alpha_s=alpha_s,
    )
    pred_pn = _peaknorm(pred_raw_te)
    true_pn = _peaknorm(hfss[te])
    mat_pn = _peaknorm(matlab[te])
    sb_pn = _peaknorm(sub_block[te])
    matlab_te = matlab[te]

    alpha_tag = f"c={alpha_c:.3f},s={alpha_s:.3f}" if use_alpha else "1.000,1.000"
    model_label = f"dual_residual alpha({alpha_tag})"

    metrics_model = evaluate_predictions(pred_pn, true_pn, matlab_te)
    metrics_baseline = evaluate_predictions(sb_pn, true_pn, matlab_te)
    metrics_reference = evaluate_predictions(mat_pn, true_pn, matlab_te)

    print_metrics_block(f"MODEL ({model_label})", metrics_model)
    print_metrics_block("BASELINE (phase-aware sub_block only)", metrics_baseline)
    print_metrics_block("REFERENCE (matlab analytical)", metrics_reference)

    per_sample = np.mean(np.abs(pred_pn - true_pn), axis=(1, 2))
    print(
        f"Per-sample recon MAE: mean={per_sample.mean():.3f}  "
        f"median={np.median(per_sample):.3f}",
        flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_masked_comparison_txt(
        out_dir / "comparison.txt",
        scale_label=scale,
        n_test=len(te),
        predictors={
            "MATLAB (analytic, no coupling)": mat_pn,
            "Phase-aware sub-block compose": sb_pn,
            model_label: pred_pn,
        },
        true_pn=true_pn,
    )

    header = (
        f"Zero-shot — dual_residual @ {scale}\n"
        f"Checkpoint: {CKPT}\n"
        f"Sub-block input: {data['pa_path']} (composed_dB)\n"
        f"Reconstruction: sub_block + alpha_c*shield(delta_c) + alpha_s*delta_s\n"
        f"scale_token={scale_tok:.6f}  alpha_c={alpha_c:.6f}  alpha_s={alpha_s:.6f}\n"
        f"shield={apply_shield}  alpha_fit={use_alpha}\n"
        f"Train/test split seed={RANDOM_SEED}  test n={len(te)}\n"
        f"Null mask: matlab < peak + {NULL_THRESHOLD_DB} dB\n"
    )
    write_metrics_txt(
        out_dir / "metrics.txt",
        header,
        [
            ("TEST_MODEL", metrics_model),
            ("TEST_BASELINE", metrics_baseline),
            ("TEST_REFERENCE", metrics_reference),
        ],
    )

    npz_payload = pack_metrics_npz(
        test_indices=te,
        per_sample_recon_mae=per_sample,
        alpha=alpha_c,
        model=metrics_model,
        baseline=metrics_baseline,
        reference=metrics_reference,
    )
    npz_payload["pred_raw_db"] = pred_raw_te.astype(np.float32)
    npz_payload["pred_pn_db"] = pred_pn.astype(np.float32)
    npz_payload["alpha_coupling"] = np.float32(alpha_c)
    npz_payload["alpha_scale"] = np.float32(alpha_s)
    npz_payload["delta_coupling_te"] = dc_te.astype(np.float32)
    npz_payload["delta_scale_te"] = ds_te.astype(np.float32)
    np.savez_compressed(out_dir / f"{scale}_test_metrics.npz", **npz_payload)
    print(f"Saved results to {out_dir}/", flush=True)

    if not write_plots or data["theta_deg"] is None or data["phi_deg"] is None:
        return

    theta = data["theta_deg"].astype(np.float64)
    phi = data["phi_deg"].astype(np.float64)
    order = np.argsort(per_sample)
    for name, idx in {
        "best": int(order[0]),
        "median": int(order[len(order) // 2]),
        "worst": int(order[-1]),
    }.items():
        ti = int(te[idx])
        sid = int(data["ids"][ti])
        plot_pattern_comparison(
            matlab=mat_pn[idx],
            predicted=pred_pn[idx],
            hfss=true_pn[idx],
            title=f"{scale} sample {sid}  MAE={per_sample[idx]:.3f} dB  [{name}]",
            save_path=str(out_dir / f"comparison_{name}_sample_{sid:04d}.png"),
        )

    picks_cuts = [
        order[int(round(i * (len(order) - 1) / max(N_CUTS - 1, 1)))]
        for i in range(N_CUTS)
    ]
    for k, idx in enumerate(picks_cuts):
        ti = int(te[idx])
        sid = int(data["ids"][ti])
        plot_plane_cuts(
            matlab=mat_pn[idx],
            predicted=pred_pn[idx],
            hfss=true_pn[idx],
            theta_grid=theta,
            phi_grid=phi,
            cut_type="both",
            title=f"{scale} sample {sid}  MAE={per_sample[idx]:.3f} dB",
            save_path=str(out_dir / f"cuts_{k:02d}.png"),
        )

    plot_error_distribution(
        (pred_pn - true_pn).ravel(),
        title=f"Pointwise dB error ({scale} dual_residual, n={len(te)})",
        save_path=str(out_dir / "error_distribution.png"),
    )
    rng = np.random.default_rng(0)
    mask = rng.choice(true_pn.size, size=min(200_000, true_pn.size), replace=False)
    plot_scatter_pred_vs_true(
        true=true_pn.ravel()[mask],
        pred=pred_pn.ravel()[mask],
        title=f"Predicted vs True ({scale} dual_residual)",
        save_path=str(out_dir / "scatter_pred_vs_true.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dual_residual at 8×8 / 16×16 (phase-aware compose + eval cache)",
    )
    parser.add_argument(
        "--scale",
        choices=("8", "16", "both"),
        default="both",
        help="Target array size (default: both)",
    )
    parser.add_argument(
        "--no-alpha",
        action="store_true",
        help="Skip LS alpha calibration on train split (use α_c=α_s=1)",
    )
    parser.add_argument(
        "--no-shield",
        action="store_true",
        help="Skip main-beam shield on coupling residual",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG outputs (metrics only)",
    )
    args = parser.parse_args()

    if not NORM_DR.exists():
        raise FileNotFoundError(
            f"{NORM_DR} missing — run: python -m scripts.prep_dual_residual_train"
        )

    stats = dict(np.load(NORM_DR))
    model = load_model(stats)
    print(
        f"Loaded {CKPT.parent.name}/{CKPT.name}  device={DEVICE}",
        flush=True,
    )

    sub6_all = dpx6 = dpy6 = None
    need_fp = False
    if args.scale in ("8", "both") and not _cache_has_fingerprint(CACHE_8):
        need_fp = True
    if args.scale in ("16", "both") and not _cache_has_fingerprint(CACHE_16):
        need_fp = True
    if need_fp:
        print("Loading 6x6 sub-blocks (fingerprint fallback) ...", flush=True)
        dpx6, dpy6, sub6_all = load_sub6_concatenated(SB6_DIR)
        print(f"  {sub6_all.shape[0]} sub-block rows", flush=True)

    use_alpha = not args.no_alpha
    apply_shield = not args.no_shield
    write_plots = not args.no_plots

    if args.scale in ("8", "both"):
        print("\nLoading 8x8 phase-aware compose ...", flush=True)
        data8 = _load_phase_aware_scale(PA_8, 8, sub6_all, dpx6, dpy6)
        eval_scale(
            model, stats, data8,
            use_alpha=use_alpha,
            apply_shield=apply_shield,
            out_dir=OUT_ROOT / "8x8",
            write_plots=write_plots,
        )

    if args.scale in ("16", "both"):
        print("\nLoading 16x16 phase-aware compose ...", flush=True)
        data16 = _load_phase_aware_scale(PA_16, 16, sub6_all, dpx6, dpy6)
        eval_scale(
            model, stats, data16,
            use_alpha=use_alpha,
            apply_shield=apply_shield,
            out_dir=OUT_ROOT / "16x16",
            write_plots=write_plots,
        )

    print("\nDone. Results:")
    if args.scale in ("8", "both"):
        print(f"  {OUT_ROOT / '8x8'}")
    if args.scale in ("16", "both"):
        print(f"  {OUT_ROOT / '16x16'}")


if __name__ == "__main__":
    main()
