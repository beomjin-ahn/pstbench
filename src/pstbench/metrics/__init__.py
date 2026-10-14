from __future__ import annotations

import numpy as np

from pstbench.metrics.deltae import ciede2000, deltae00
from pstbench.metrics.hcorr import hcorr
from pstbench.metrics.lpips_ import LPIPS
from pstbench.metrics.psnr import psnr
from pstbench.metrics.ssim import ssim
from pstbench.protocol.spec import Protocol

__all__ = [
    "LPIPS",
    "MetricSuite",
    "build_suite",
    "ciede2000",
    "deltae00",
    "hcorr",
    "psnr",
    "ssim",
]

COLUMNS = ("psnr", "ssim", "lpips", "deltae00", "hcorr_gt", "hcorr_style")


class MetricSuite:
    columns = COLUMNS

    def __init__(self, protocol: Protocol, device: str = "cpu"):
        self.protocol = protocol
        p = protocol["metrics"]
        self._psnr = p["psnr"]
        self._ssim = p["ssim"]
        self._de = p["deltae00"]
        self._hc = p["hcorr"]
        self._lpips = LPIPS(net=p["lpips"]["net"], version=p["lpips"]["version"], device=device)

    def score_paired(
        self, pred: np.ndarray, ref: np.ndarray, style: np.ndarray | None = None
    ) -> dict[str, float | bool | None]:
        if pred.shape != ref.shape:
            raise ValueError(f"prediction {pred.shape} and reference {ref.shape} differ in size")
        v_psnr, capped = psnr(pred, ref, self._psnr["data_range"], self._psnr["cap_db"])
        s = self._ssim
        row: dict[str, float | bool | None] = {
            "psnr": v_psnr,
            "ssim": ssim(pred, ref, s["sigma"], s["k1"], s["k2"], s["data_range"]),
            "lpips": self._lpips(pred, ref),
            "deltae00": deltae00(pred, ref, self._de["kL"], self._de["kC"], self._de["kH"]),
            "hcorr_gt": hcorr(pred, ref, self._hc["bins"]),
            "hcorr_style": hcorr(pred, style, self._hc["bins"]) if style is not None else None,
            "psnr_capped": capped,
        }
        return row


def build_suite(protocol: Protocol, device: str = "cpu"):
    return MetricSuite(protocol, device)
