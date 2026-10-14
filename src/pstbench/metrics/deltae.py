from __future__ import annotations

import numpy as np

from pstbench.color import srgb_to_lab

_25_POW_7 = 25.0**7


def ciede2000(
    lab1: np.ndarray, lab2: np.ndarray, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0
) -> np.ndarray:
    L1, a1, b1 = (np.asarray(lab1, dtype=np.float64)[..., i] for i in range(3))
    L2, a2, b2 = (np.asarray(lab2, dtype=np.float64)[..., i] for i in range(3))

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar7 = ((C1 + C2) / 2.0) ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + _25_POW_7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p
    Cprod = C1p * C2p
    dh = h2p - h1p
    dhp = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dhp = np.where(Cprod == 0.0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(Cprod) * np.sin(np.radians(dhp) / 2.0)

    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    hp_bar = np.where(
        np.abs(h1p - h2p) <= 180.0,
        hsum / 2.0,
        np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0),
    )
    hp_bar = np.where(Cprod == 0.0, hsum, hp_bar)

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar**7
    R_C = 2.0 * np.sqrt(Cp_bar7 / (Cp_bar7 + _25_POW_7))
    L50sq = (Lp_bar - 50.0) ** 2
    S_L = 1.0 + 0.015 * L50sq / np.sqrt(20.0 + L50sq)
    S_C = 1.0 + 0.045 * Cp_bar
    S_H = 1.0 + 0.015 * Cp_bar * T
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    tL = dLp / (kL * S_L)
    tC = dCp / (kC * S_C)
    tH = dHp / (kH * S_H)
    return np.sqrt(tL**2 + tC**2 + tH**2 + R_T * tC * tH)


def deltae00(pred: np.ndarray, ref: np.ndarray, kL=1.0, kC=1.0, kH=1.0) -> float:
    d = ciede2000(srgb_to_lab(pred), srgb_to_lab(ref), kL, kC, kH)
    return float(d.mean())
