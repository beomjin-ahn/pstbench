from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def ssim(
    pred: np.ndarray,
    ref: np.ndarray,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 255.0,
) -> float:
    return float(
        structural_similarity(
            pred.astype(np.float64),
            ref.astype(np.float64),
            gaussian_weights=True,
            sigma=sigma,
            use_sample_covariance=False,
            K1=k1,
            K2=k2,
            data_range=data_range,
            channel_axis=-1,
        )
    )
