from __future__ import annotations

import numpy as np

RGB_TO_XYZ = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)
WHITE_D65 = RGB_TO_XYZ.sum(axis=1)

_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_xyz(rgb: np.ndarray) -> np.ndarray:
    return rgb @ RGB_TO_XYZ.T


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray = WHITE_D65) -> np.ndarray:
    t = xyz / white
    f = np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def srgb_to_lab(img: np.ndarray) -> np.ndarray:
    v = img.astype(np.float64) / 255.0 if img.dtype == np.uint8 else img.astype(np.float64)
    return xyz_to_lab(linear_to_xyz(srgb_to_linear(v)))
