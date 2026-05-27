"""Wrapper around the 5-channel residual generator for multi-scale HFSS-style prediction.

The network is trained (e.g. with ``scripts.train_residual_physics_2to4``) only on
2x2 and 4x4 supervised examples. At inference, the **scale token** ``N/16`` and a
**one-step-smaller anchor** channel let you query N=8 or N=16 **without** ever showing
those sizes during training — the model extrapolates from learned small-array physics.

Convention matches ``train_residual_multiscale``: inputs are analytical matlab patterns
and cascade anchors in dB (max-normalised to 0 in the HDF5 sources); outputs are
predicted dB patterns after residual composition + per-sample max-normalisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from src.config import DEVICE, PROCESSED_DIR
from scripts.train_cgan_2to4_fusion_no_m4 import (
    ATTN_HEADS,
    GEN_BASE,
    EnhancedResUNetGenerator,
)
from scripts.train_residual_multiscale import (
    AMP_DEVICE_TYPE,
    AMP_DTYPE,
    ResidualReconLoss,
    USE_AMP,
)

NORM_COMBINED = PROCESSED_DIR / "norm_stats_matlab_combined.npz"


class ResidualPhysicsPredictor:
    """Load EMA (or raw) generator weights and run residual forward at any supported scale.

    Parameters
    ----------
    checkpoint:
        Path to ``best_generator.pt`` (state dict). Bootstrap/stage checkpoints save
        EMA weights under this name.
    norm_npz:
        ``norm_stats_matlab_combined.npz`` (default: ``processed/`` artefact).
    device:
        Torch device; defaults to ``src.config.DEVICE``.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        norm_npz: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or DEVICE
        path = Path(norm_npz or NORM_COMBINED)
        if not path.exists():
            raise FileNotFoundError(path)

        stats = np.load(path)
        self._mean = torch.from_numpy(stats["mean"].astype(np.float32)).to(self.device)
        self._std = torch.from_numpy(
            np.maximum(stats["std"].astype(np.float32), 1e-6)
        ).to(self.device)
        sigma = float(stats["residual_std"])
        self._recon = ResidualReconLoss(sigma_res=sigma).to(self.device)

        self.G = EnhancedResUNetGenerator(
            in_ch=5, out_ch=1, base=GEN_BASE, attn_heads=ATTN_HEADS
        ).to(self.device)
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        self.G.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.G.eval()

    @torch.no_grad()
    def forward_tensor(
        self,
        matlab_db: torch.Tensor,
        anchor_db: torch.Tensor,
        meta_deg: torch.Tensor,
        scale_n: int,
    ) -> torch.Tensor:
        """Batched forward.

        matlab_db : (B, H, W) dB
        anchor_db : (B, H, W) dB — use zeros for N=2
        meta_deg  : (B, 6) — columns 0,1 are dphase_x, dphase_y in degrees
        scale_n   : element edge length N in {2, 4, 8, 16}

        Returns (B, H, W) predicted dB (max-normalised per sample).
        """
        if matlab_db.dim() != 3:
            raise ValueError(f"matlab_db expected (B,H,W), got {matlab_db.shape}")
        B, H, W = matlab_db.shape
        matlab_db = matlab_db.to(self.device)
        anchor_db = anchor_db.to(self.device)
        meta_deg = meta_deg.to(self.device)

        mat_n = (matlab_db - self._mean) / self._std
        anchor_n = (anchor_db - self._mean) / self._std
        dpx = meta_deg[:, 0:1, None, None] / 180.0
        dpy = meta_deg[:, 1:2, None, None] / 180.0
        dpx = dpx.expand(-1, -1, H, W)
        dpy = dpy.expand(-1, -1, H, W)
        tok = torch.full(
            (B, 1, H, W),
            fill_value=float(scale_n) / 16.0,
            device=self.device,
            dtype=mat_n.dtype,
        )
        x = torch.cat([mat_n.unsqueeze(1), dpx, dpy, tok, anchor_n.unsqueeze(1)], dim=1)
        m1 = matlab_db.unsqueeze(1)
        with torch.amp.autocast(
            device_type=AMP_DEVICE_TYPE, dtype=AMP_DTYPE, enabled=USE_AMP
        ):
            delta_n = self.G(x)
            pred = self._recon.compose(delta_n, m1)
        return pred[:, 0]

    def predict_numpy(
        self,
        matlab_db: np.ndarray,
        anchor_db: np.ndarray,
        meta: np.ndarray,
        scale_n: int,
    ) -> np.ndarray:
        """Numpy API; supports batch (B,H,W) or single (H,W)."""
        single = matlab_db.ndim == 2
        if single:
            matlab_db = matlab_db[None]
            anchor_db = anchor_db[None]
            meta = meta[None]
        if matlab_db.shape != anchor_db.shape:
            raise ValueError("matlab_db and anchor_db must match shape")
        m = torch.from_numpy(matlab_db.astype(np.float32))
        a = torch.from_numpy(anchor_db.astype(np.float32))
        md = torch.from_numpy(meta.astype(np.float32))
        out = self.forward_tensor(m, a, md, scale_n).float().cpu().numpy()
        return out[0] if single else out

    def cascade_to_4x4(
        self,
        matlab_2x2: np.ndarray,
        matlab_4x4: np.ndarray,
        meta: np.ndarray,
        anchor_2x2: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Two-step cascade: N=2 then N=4.

        If ``anchor_2x2`` is None, uses a zero anchor at N=2, then model prediction
        at 2x2 as ch4 for the 4x4 pass (self-consistent with training, which uses
        dataset ``hfss_pred_2x2`` as anchor for N=4).

        Returns (pred_2x2, pred_4x4), each (H,W) or (B,H,W).
        """
        if anchor_2x2 is None:
            anchor_2x2 = np.zeros_like(matlab_2x2, dtype=np.float32)
        p2 = self.predict_numpy(matlab_2x2, anchor_2x2, meta, scale_n=2)
        p4 = self.predict_numpy(matlab_4x4, p2, meta, scale_n=4)
        return p2, p4

    def predict_at_scale_with_anchor(
        self,
        matlab_db: np.ndarray,
        anchor_db: np.ndarray,
        meta: np.ndarray,
        scale_n: int,
    ) -> np.ndarray:
        """Alias for ``predict_numpy`` (explicit name for 8x8 / 16x16 calls)."""
        return self.predict_numpy(matlab_db, anchor_db, meta, scale_n)
