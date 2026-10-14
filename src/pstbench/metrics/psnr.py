from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, ref: np.ndarray, data_range: float = 255.0, cap_db: float = 100.0):
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return cap_db, True
    return min(10.0 * np.log10(data_range**2 / mse), cap_db), False
