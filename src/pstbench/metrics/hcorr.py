from __future__ import annotations

import numpy as np


def channel_histograms(img: np.ndarray, bins: int = 256) -> np.ndarray:
    if img.dtype != np.uint8:
        raise ValueError(f"hcorr expects uint8 images, got {img.dtype}")
    if bins == 256:
        return np.stack([np.bincount(img[..., c].ravel(), minlength=256) for c in range(3)])
    return np.stack([np.histogram(img[..., c], bins=bins, range=(0, 256))[0] for c in range(3)])


def _pearson(h1: np.ndarray, h2: np.ndarray) -> float:
    x = h1.astype(np.float64) - h1.mean()
    y = h2.astype(np.float64) - h2.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0.0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float((x * y).sum() / denom)


def hcorr(pred: np.ndarray, ref: np.ndarray, bins: int = 256) -> float:
    hp = channel_histograms(pred, bins)
    hr = channel_histograms(ref, bins)
    return float(np.mean([_pearson(hp[c], hr[c]) for c in range(3)]))
