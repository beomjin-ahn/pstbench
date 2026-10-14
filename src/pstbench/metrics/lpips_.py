from __future__ import annotations

import warnings

import numpy as np
import torch


class LPIPS:
    def __init__(self, net: str = "alex", version: str = "0.1", device: str = "cpu"):
        self.net = net
        self.version = version
        self.device = torch.device(device)
        self._model = None

    def _load(self):
        import lpips

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = lpips.LPIPS(net=self.net, version=self.version, verbose=False)
        self._model = model.eval().to(self.device)

    @staticmethod
    def _to_tensor(img: np.ndarray, unit_range: bool) -> torch.Tensor:
        if img.dtype != np.uint8:
            raise ValueError(f"LPIPS expects uint8 images, got {img.dtype}")
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return t if unit_range else t * 2.0 - 1.0

    @torch.no_grad()
    def __call__(self, pred: np.ndarray, ref: np.ndarray, unit_range: bool = False) -> float:
        if self._model is None:
            self._load()
        p = self._to_tensor(pred, unit_range).to(self.device)
        r = self._to_tensor(ref, unit_range).to(self.device)
        return float(self._model(p, r, normalize=False).item())
