"""
Zero-shot evaluation of stage-1 6x6 multi-head gated ResUNet at 6x6 (in-distribution),
8x8, and 16x16.

Mirrors evaluate_stage2_from_6x6.py + zeroshot_6x6_to_16x16.py using
checkpoints/stage1_6x6_multihead/best_generator.pt.

Results: results/stage2_multihead_from_6x6/

Usage:
    python -m scripts.evaluate_stage2_multihead_from_6x6
"""
import sys, csv, glob, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import h5py
import torch

from src.config import PROCESSED_DIR, CHECKPOINTS_DIR, DEVICE, NULL_THRESHOLD_DB
from src.training.metrics import rmse, mae, pearson_correlation
from src.models.stage1_multihead_resunet import GatedMultiHeadResUNet
from scripts.train_cgan_2to4_fusion_no_m4 import GEN_BASE, ATTN_HEADS

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

NTH, NPH = 181, 360
CKPT = CHECKPOINTS_DIR / "stage1_6x6_multihead" / "best_generator.pt"
OUT_DIR = PROJECT_ROOT / "results" / "stage2_multihead_from_6x6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NORM_6X6   = PROCESSED_DIR / "norm_stats_stage1_6x6.npz"
SPLIT_6X6  = PROCESSED_DIR / "split_indices_stage1_6x6.npz"
EXTRAS_6X6 = PROCESSED_DIR / "stage1_6x6_extras.npz"
SUBCOMP_V2 = PROCESSED_DIR / "subblock_compositions_v2.npz"
H5_4TO8    = PROCESSED_DIR / "antenna_data_4to8.h5"
NPZ_4X4    = PROCESSED_DIR / "subblock_4x4_compose.npz"
EXTRAS_OLD = PROCESSED_DIR / "stage1_extras.npz"
HFSS16_DIR = PROJECT_ROOT / "datasets_16x16_hfss" / "datasets_16x16_hfss"


def _resolve_csv_dir(*candidates):
    for d in candidates:
        if d.exists() and any(d.glob("patterns_global_*.csv")):
            return d
    return candidates[0]


SB6_DIR = _resolve_csv_dir(
    PROJECT_ROOT / "datasets_6x6sub-block_hfss" / "datasets_6x6sub-block_hfss",
    PROJECT_ROOT / "datasets_6x6sub-block_hfss",
)
CACHE_8X8 = OUT_DIR / "sub_block_8x8_cache.npz"


def pair_by_steering(dpx_tgt, dpy_tgt, dpx_pool, dpy_pool, tol=1e-3):
    """Map each target steering to the nearest pool index."""
    dpx_tgt = np.asarray(dpx_tgt, np.float64)
    dpy_tgt = np.asarray(dpy_tgt, np.float64)
    dpx_pool = np.asarray(dpx_pool, np.float64)
    dpy_pool = np.asarray(dpy_pool, np.float64)
    idx = np.empty(len(dpx_tgt), dtype=np.int64)
    for i in range(len(dpx_tgt)):
        dist = (dpx_pool - dpx_tgt[i]) ** 2 + (dpy_pool - dpy_tgt[i]) ** 2
        idx[i] = int(dist.argmin())
    err = np.sqrt((dpx_pool[idx] - dpx_tgt) ** 2 + (dpy_pool[idx] - dpy_tgt) ** 2)
    if err.max() > tol:
        raise RuntimeError(f"steering pairing failed: max err={err.max():.6f}")
    return idx


def load_model(stats):
    if not CKPT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CKPT}")
    G = GatedMultiHeadResUNet(
        in_ch=5, base=GEN_BASE, attn_heads=ATTN_HEADS, top_k=10,
        sb_mean=stats["sub_block_6x6_mean"], sb_std=stats["sub_block_6x6_std"],
        res_mean=stats["residual_mean"], res_std=stats["residual_std"],
    ).to(DEVICE)
    G.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    G.eval()
    print(f"Loaded: {CKPT}  device={DEVICE}", flush=True)
    return G


def infer_residual_batch(G, stats, matlab, sub_block, fingerprint, dpx, dpy,
                         m_mean, m_std, sb_mean, sb_std, batch=8):
    n = len(dpx)
    out = np.empty((n, NTH, NPH), dtype=np.float32)
    fp_mean, fp_std = stats["fingerprint_mean"], stats["fingerprint_std"]
    res_mean, res_std = stats["residual_mean"], stats["residual_std"]
    with torch.no_grad():
        for start in range(0, n, batch):
            end = min(start + batch, n)
            xs = []
            for i in range(start, end):
                xs.append(np.stack([
                    (matlab[i] - m_mean) / m_std,
                    (sub_block[i] - sb_mean) / sb_std,
                    (fingerprint[i] - fp_mean) / fp_std,
                    np.full((NTH, NPH), dpx[i] / 180.0, np.float32),
                    np.full((NTH, NPH), dpy[i] / 180.0, np.float32),
                ], axis=0))
            x = torch.from_numpy(np.stack(xs)).to(DEVICE)
            p = G(x).cpu().numpy()
            for j, i in enumerate(range(start, end)):
                out[i] = p[j, 0] * res_std + res_mean
            if start == 0 or end == n:
                print(f"    inferred {end}/{n}", flush=True)
    return out


def fit_alpha(true_res, pred_res):
    return float(np.sum(true_res * pred_res) / np.sum(pred_res ** 2))


def peaknorm(x):
    return (x - x.max(axis=(1, 2), keepdims=True)).astype(np.float32)


def report(name, pred, true, lines):
    line = (f"  {name:54s} RMSE={rmse(true.ravel(), pred.ravel()):6.3f}  "
            f"MAE={mae(true.ravel(), pred.ravel()):6.3f}  "
            f"Pearson={pearson_correlation(true.ravel(), pred.ravel()):.4f}")
    null = true < NULL_THRESHOLD_DB
    beam = ~null
    if null.any():
        line += (f"  beam_MAE={float(np.mean(np.abs(pred[beam]-true[beam]))):.3f}"
                 f"  null_MAE={float(np.mean(np.abs(pred[null]-true[null]))):.3f}")
    print(line, flush=True)
    lines.append(line + "\n")


def section(title, lines):
    bar = "=" * 100
    print(f"\n{bar}\n{title}\n{bar}", flush=True)
    lines.append(f"\n{title}\n" + "=" * 80 + "\n")


def eval_6x6_indistribution(G, stats, lines):
    """Held-out 6x6 test — same task the model was trained on."""
    section("6x6 IN-DISTRIBUTION (held-out test split)", lines)
    e = np.load(EXTRAS_6X6)
    sp = np.load(SPLIT_6X6)
    tr, te = sp["train"], sp["test"]

    matlab = e["matlab_6x6"].astype(np.float32)
    sb = e["sub_block_6x6"].astype(np.float32)
    hfss = e["hfss_6x6"].astype(np.float32)
    fp = e["fingerprint"].astype(np.float32)
    dpx, dpy = e["dpx"], e["dpy"]

    m_mean = matlab[tr].mean(0)
    m_std = np.maximum(matlab[tr].std(0), 1e-6)
    sb_mean = sb[tr].mean(0)
    sb_std = np.maximum(sb[tr].std(0), 1e-6)

    print(f"Inference on test n={len(te)} ...", flush=True)
    pred_te = infer_residual_batch(G, stats, matlab[te], sb[te], fp[te],
                                   dpx[te], dpy[te], m_mean, m_std, sb_mean, sb_std)
    true_res = hfss[te] - sb[te]
    pred_pn = peaknorm(sb[te] + pred_te)
    hf_pn = peaknorm(hfss[te])
    sb_pn = peaknorm(sb[te])

    report("model (sub_block + predicted residual)", pred_pn, hf_pn, lines)
    report("baseline (sub_block only)", sb_pn, hf_pn, lines)

    # Normalised residual RMSE (training metric)
    res_n_true = ((true_res - stats["residual_mean"]) / stats["residual_std"]).ravel()
    res_n_pred = ((pred_te - stats["residual_mean"]) / stats["residual_std"]).ravel()
    lines.append(f"  residual_n RMSE={rmse(res_n_true, res_n_pred):.4f}  "
                   f"MAE={mae(res_n_true, res_n_pred):.4f}\n")
    print(lines[-1], flush=True)

    np.savez_compressed(OUT_DIR / "6x6_test_metrics.npz",
                        test_indices=te,
                        per_sample_mae=np.mean(np.abs(pred_pn - hf_pn), axis=(1, 2)))


def read_csv(path, nb):
    rows = list(csv.reader(open(path)))
    hdr = rows[0]
    S = (len(hdr) - 2) // nb
    ids = [int(hdr[2 + nb * s].split("_")[0][1:]) for s in range(S)]
    dpx = np.array([float(rows[1][2 + nb * s]) for s in range(S)])
    dpy = np.array([float(rows[2][2 + nb * s]) for s in range(S)])
    blk = np.array(rows[5:], np.float64)
    pat = np.transpose(blk[:, 2:].T.reshape(S, nb, NPH, NTH), (0, 1, 3, 2))
    return np.array(ids), dpx, dpy, pat.astype(np.float32)


def build_sub_block_8x8(sub6_m, dpx_p, dpy_p, theta_deg, phi_deg):
    THETA, PHI = np.meshgrid(np.deg2rad(theta_deg), np.deg2rad(phi_deg), indexing="ij")
    sin_theta, cos_phi, sin_phi = np.sin(THETA), np.cos(PHI), np.sin(PHI)
    kd = np.pi

    def b_for_8x8(m, n):
        if (m, n) == (0, 0): return 0
        if (m, n) == (0, 3): return 2
        if (m, n) == (3, 0): return 6
        if (m, n) == (3, 3): return 8
        if m == 0: return 1
        if m == 3: return 7
        if n == 0: return 3
        if n == 3: return 5
        return 4

    sub6_lin = 10.0 ** (sub6_m / 20.0)
    N = len(dpx_p)
    out = np.empty((N, NTH, NPH), dtype=np.float32)
    psi_geom = np.empty((4, 4, NTH, NPH), dtype=np.float32)
    for m in range(4):
        for n in range(4):
            xm, yn = 2 * m - 3, 2 * n - 3
            psi_geom[m, n] = (kd * sin_theta * (xm * cos_phi + yn * sin_phi)).astype(np.float32)
    for s in range(N):
        dpx_r, dpy_r = np.deg2rad(dpx_p[s]), np.deg2rad(dpy_p[s])
        AF = np.zeros((NTH, NPH), dtype=np.complex128)
        for m in range(4):
            for n in range(4):
                xm, yn = 2 * m - 3, 2 * n - 3
                AF += sub6_lin[s, b_for_8x8(m, n)] * np.exp(
                    1j * (psi_geom[m, n] + xm * dpx_r + yn * dpy_r))
        dB = 20.0 * np.log10(np.maximum(np.abs(AF), 1e-12))
        out[s] = (dB - dB.max()).astype(np.float32)
    return out


def eval_8x8(G, stats, lines):
    if not H5_4TO8.exists():
        msg = f"SKIP 8x8: missing {H5_4TO8.name} (run data prep or git lfs pull)"
        print(msg, flush=True)
        lines.append(msg + "\n")
        return

    section("8x8 ZERO-SHOT (6x6 multi-head model)", lines)
    sp = np.load(SPLIT_6X6)
    tr, te = sp["train"], sp["test"]
    e6 = np.load(EXTRAS_6X6)

    if SUBCOMP_V2.exists():
        sub_block_8x8 = np.load(SUBCOMP_V2)["sub_block_8x8"].astype(np.float32)
    elif CACHE_8X8.exists():
        print(f"  loading cached {CACHE_8X8.name} ...", flush=True)
        sub_block_8x8 = np.load(CACHE_8X8)["sub_block_8x8"].astype(np.float32)
    else:
        print("  building sub_block_8x8 from 6x6 sub-blocks ...", flush=True)
        if not SB6_DIR.exists():
            raise FileNotFoundError(f"Need {SB6_DIR} or {SUBCOMP_V2.name}")
        files = sorted(glob.glob(str(SB6_DIR / "patterns_global_*.csv")))
        ids_l, dpx_l, dpy_l, sub_l = [], [], [], []
        for f in files:
            ids, dpx, dpy, pat = read_csv(f, 9)
            ids_l.append(ids); dpx_l.append(dpx); dpy_l.append(dpy); sub_l.append(pat)
        dpx6 = np.concatenate(dpx_l); dpy6 = np.concatenate(dpy_l)
        sub6 = np.concatenate(sub_l, axis=0)
        pair_idx = pair_by_steering(e6["dpx"], e6["dpy"], dpx6, dpy6)
        sub6_p = sub6[pair_idx]
        sub_block_8x8 = build_sub_block_8x8(sub6_p, e6["dpx"], e6["dpy"],
                                            e6["theta"], e6["phi"])
        np.savez_compressed(CACHE_8X8, sub_block_8x8=sub_block_8x8)
        print(f"  cached {CACHE_8X8.name}", flush=True)

    if EXTRAS_OLD.exists():
        ex = np.load(EXTRAS_OLD)
        fp = ex["matlab_2x2"].astype(np.float32) - ex["hfss_2x2_mean"].astype(np.float32)
    else:
        fp = e6["fingerprint"].astype(np.float32)

    with h5py.File(H5_4TO8, "r") as f:
        N = sub_block_8x8.shape[0]
        matlab_8x8 = f["matlab_8x8"][:N].astype(np.float32)
        hfss_8x8 = f["hfss_8x8"][:N].astype(np.float32)
    dpx_all, dpy_all = e6["dpx"], e6["dpy"]

    m8_mean = matlab_8x8[tr].mean(0)
    m8_std = np.maximum(matlab_8x8[tr].std(0), 1e-6)
    sb_mean = sub_block_8x8[tr].mean(0)
    sb_std = np.maximum(sub_block_8x8[tr].std(0), 1e-6)

    print(f"Inference train n={len(tr)}, test n={len(te)} ...", flush=True)
    pred_tr = infer_residual_batch(G, stats, matlab_8x8[tr], sub_block_8x8[tr], fp[tr],
                                   dpx_all[tr], dpy_all[tr], m8_mean, m8_std, sb_mean, sb_std)
    pred_te = infer_residual_batch(G, stats, matlab_8x8[te], sub_block_8x8[te], fp[te],
                                   dpx_all[te], dpy_all[te], m8_mean, m8_std, sb_mean, sb_std)

    true_res_tr = hfss_8x8[tr] - sub_block_8x8[tr]
    alpha = fit_alpha(true_res_tr, pred_tr)
    lines.append(f"alpha (LS on 8x8 train): {alpha:.4f}\n")
    print(f"alpha (LS on 8x8 train): {alpha:.4f}", flush=True)

    pred_pn = peaknorm(sub_block_8x8[te] + alpha * pred_te)
    hf_pn = peaknorm(hfss_8x8[te])
    sb_pn = peaknorm(sub_block_8x8[te])
    m8_pn = peaknorm(matlab_8x8[te])

    report("model  (sub_block_8x8_v2 + alpha*residual)", pred_pn, hf_pn, lines)
    report("baseline (sub_block_8x8_v2 only)", sb_pn, hf_pn, lines)
    report("reference (matlab_8x8 analytical)", m8_pn, hf_pn, lines)

    per_sample = np.mean(np.abs(pred_pn - hf_pn), axis=(1, 2))
    lines.append(f"Per-sample MAE: mean={per_sample.mean():.3f}  "
                 f"median={np.median(per_sample):.3f}\n")
    print(lines[-1], flush=True)

    np.savez_compressed(OUT_DIR / "8x8_test_metrics.npz",
                        test_indices=te, per_sample_recon_mae=per_sample, alpha=alpha)


def build_16x16_arrays():
    if not HFSS16_DIR.exists():
        return None
    files16 = sorted(glob.glob(str(HFSS16_DIR / "patterns_global_*.csv")))
    ids_l, dpx_l, dpy_l, pat_l = [], [], [], []
    for f in files16:
        ids, dpx, dpy, pat = read_csv(f, 1)
        ids_l.append(ids); dpx_l.append(dpx); dpy_l.append(dpy); pat_l.append(pat[:, 0])
    dpx16 = np.concatenate(dpx_l); dpy16 = np.concatenate(dpy_l)
    hf16 = np.concatenate(pat_l, axis=0).astype(np.float32)

    filessb = sorted(glob.glob(str(SB6_DIR / "patterns_global_*.csv")))
    ids_l, dpx_l, dpy_l, sub_l = [], [], [], []
    for f in filessb:
        ids, dpx, dpy, pat = read_csv(f, 9)
        ids_l.append(ids); dpx_l.append(dpx); dpy_l.append(dpy); sub_l.append(pat)
    dpx6 = np.concatenate(dpx_l); dpy6 = np.concatenate(dpy_l)
    sub6 = np.concatenate(sub_l, axis=0).astype(np.float32)

    idx6 = pair_by_steering(dpx16, dpy16, dpx6, dpy6)
    matched_16 = np.arange(len(dpx16), dtype=np.int64)
    matched_6 = idx6
    N = len(dpx16)
    hf16_m = hf16[matched_16]
    sub6_m = sub6[matched_6]
    dpx_p = dpx16[matched_16]
    dpy_p = dpy16[matched_16]

    e6 = np.load(EXTRAS_6X6)
    theta_deg = e6["theta"].astype(np.float32)
    phi_deg = e6["phi"].astype(np.float32)
    THETA, PHI = np.meshgrid(np.deg2rad(theta_deg), np.deg2rad(phi_deg), indexing="ij")
    sin_theta, cos_phi, sin_phi = np.sin(THETA), np.cos(PHI), np.sin(PHI)
    kd = np.pi

    def b_for_16x16(m, n):
        corners = {(0, 0): 0, (0, 7): 2, (7, 0): 6, (7, 7): 8}
        if (m, n) in corners: return corners[(m, n)]
        if m == 0: return 1
        if m == 7: return 7
        if n == 0: return 3
        if n == 7: return 5
        return 4

    sub6_lin = 10.0 ** (sub6_m / 20.0)
    sub_block_16x16 = np.empty((N, NTH, NPH), dtype=np.float32)
    psi_geom = np.empty((8, 8, NTH, NPH), dtype=np.float32)
    for m in range(8):
        for n in range(8):
            xm, yn = 2 * m - 7, 2 * n - 7
            psi_geom[m, n] = (kd * sin_theta * (xm * cos_phi + yn * sin_phi)).astype(np.float32)
    for s in range(N):
        dpx_r, dpy_r = np.deg2rad(dpx_p[s]), np.deg2rad(dpy_p[s])
        AF = np.zeros((NTH, NPH), dtype=np.complex128)
        for m in range(8):
            for n in range(8):
                xm, yn = 2 * m - 7, 2 * n - 7
                AF += sub6_lin[s, b_for_16x16(m, n)] * np.exp(
                    1j * (psi_geom[m, n] + xm * dpx_r + yn * dpy_r))
        dB = 20.0 * np.log10(np.maximum(np.abs(AF), 1e-12))
        sub_block_16x16[s] = (dB - dB.max()).astype(np.float32)

    matlab_16x16 = np.empty((N, NTH, NPH), dtype=np.float32)
    half = 7.5
    kx = (kd * sin_theta * cos_phi).astype(np.float64)
    ky = (kd * sin_theta * sin_phi).astype(np.float64)
    for s in range(N):
        dpx_r, dpy_r = np.deg2rad(dpx_p[s]), np.deg2rad(dpy_p[s])
        AF_x = sum(np.exp(1j * (i - half) * (kx + dpx_r)) for i in range(16))
        AF_y = sum(np.exp(1j * (j - half) * (ky + dpy_r)) for j in range(16))
        dB = 20.0 * np.log10(np.maximum(np.abs(AF_x * AF_y), 1e-12))
        matlab_16x16[s] = (dB - dB.max()).astype(np.float32)

    fp = e6["fingerprint"].astype(np.float32)
    key6_steer = {(round(dpx_p[i], 4), round(dpy_p[i], 4)): i for i in range(N)}
    fingerprint = np.empty((N, NTH, NPH), dtype=np.float32)
    fp_mean = fp.mean(0)
    for s in range(N):
        k = (round(dpx_p[s], 4), round(dpy_p[s], 4))
        j = key6_steer.get(k, -1)
        # match to 6x6 extras index if same steering in 1000 set
        fingerprint[s] = fp_mean  # fallback; refined below
    e6_dpx, e6_dpy = e6["dpx"], e6["dpy"]
    key_e6 = {(round(e6_dpx[i], 4), round(e6_dpy[i], 4)): i for i in range(len(e6_dpx))}
    for s in range(N):
        k = (round(dpx_p[s], 4), round(dpy_p[s], 4))
        j = key_e6.get(k, -1)
        fingerprint[s] = fp[j] if j >= 0 else fp_mean

    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    n_tr = int(0.80 * N); n_va = int(0.10 * N)
    tr = np.sort(perm[:n_tr])
    te = np.sort(perm[n_tr + n_va:])
    return dict(hf16_m=hf16_m, sub_block_16x16=sub_block_16x16, matlab_16x16=matlab_16x16,
                fingerprint=fingerprint, dpx_p=dpx_p, dpy_p=dpy_p, tr=tr, te=te)


def eval_16x16(G, stats, data, lines):
    section("16x16 ZERO-SHOT (6x6 multi-head model)", lines)
    tr, te = data["tr"], data["te"]
    matlab_16x16 = data["matlab_16x16"]
    sub_block_16x16 = data["sub_block_16x16"]
    hf16_m = data["hf16_m"]
    fingerprint = data["fingerprint"]
    dpx_p, dpy_p = data["dpx_p"], data["dpy_p"]

    def chan_stats(a, idx):
        s = a[idx].astype(np.float32)
        return s.mean(0), np.maximum(s.std(0), 1e-6)

    m16_mean, m16_std = chan_stats(matlab_16x16, tr)
    sb_mean, sb_std = chan_stats(sub_block_16x16, tr)

    print(f"Inference train n={len(tr)}, test n={len(te)} ...", flush=True)
    pred_tr = infer_residual_batch(G, stats, matlab_16x16[tr], sub_block_16x16[tr],
                                   fingerprint[tr], dpx_p[tr], dpy_p[tr],
                                   m16_mean, m16_std, sb_mean, sb_std)
    pred_te = infer_residual_batch(G, stats, matlab_16x16[te], sub_block_16x16[te],
                                   fingerprint[te], dpx_p[te], dpy_p[te],
                                   m16_mean, m16_std, sb_mean, sb_std)

    alpha = fit_alpha(hf16_m[tr] - sub_block_16x16[tr], pred_tr)
    lines.append(f"alpha (LS on 16x16 train): {alpha:.4f}\n")
    print(f"alpha (LS on 16x16 train): {alpha:.4f}", flush=True)

    pred_pn = peaknorm(sub_block_16x16[te] + alpha * pred_te)
    true_pn = peaknorm(hf16_m[te])
    sb_pn = peaknorm(sub_block_16x16[te])
    mat_pn = peaknorm(matlab_16x16[te])

    report("model  (sub_block_16x16 + alpha*residual)", pred_pn, true_pn, lines)
    report("baseline (sub_block_16x16 only)", sb_pn, true_pn, lines)
    report("reference (matlab_16x16 analytical)", mat_pn, true_pn, lines)

    per_sample = np.mean(np.abs(pred_pn - true_pn), axis=(1, 2))
    lines.append(f"Per-sample MAE: mean={per_sample.mean():.3f}  "
                 f"median={np.median(per_sample):.3f}\n")
    print(lines[-1], flush=True)

    np.savez_compressed(OUT_DIR / "16x16_test_metrics.npz",
                        test_indices=te, per_sample_recon_mae=per_sample, alpha=alpha,
                        pred_16x16_db=pred_pn, true_16x16_db=true_pn)


def main():
    stats = dict(np.load(NORM_6X6))
    G = load_model(stats)
    lines = [f"Evaluation: stage1_6x6_multihead\nCheckpoint: {CKPT}\nDevice: {DEVICE}\n"]

    eval_6x6_indistribution(G, stats, lines)
    eval_8x8(G, stats, lines)

    if HFSS16_DIR.exists():
        data16 = build_16x16_arrays()
        eval_16x16(G, stats, data16, lines)
    else:
        msg = f"SKIP 16x16: missing {HFSS16_DIR}"
        print(msg, flush=True)
        lines.append(msg + "\n")

    path = OUT_DIR / "metrics.txt"
    path.write_text("".join(lines))
    print(f"\nSaved to {OUT_DIR}/", flush=True)
    print(f"  {path.name}", flush=True)


if __name__ == "__main__":
    main()
