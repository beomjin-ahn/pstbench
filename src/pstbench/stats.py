from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values, level: float = 0.95, resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("bootstrap_ci expects a non-empty 1-D sequence")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(resamples, x.size))
    means = x[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(x.mean()), float(low), float(high)


def paired_diff_ci(
    a, b, level: float = 0.95, resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return bootstrap_ci(d, level=level, resamples=resamples, seed=seed)


def holm(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def paired_tests(samples: dict, tie: float = 0.0) -> dict:
    from itertools import combinations

    from scipy.stats import wilcoxon

    names = list(samples)
    pairs = list(combinations(names, 2))
    p = []
    for a, b in pairs:
        d = np.asarray(samples[a], dtype=np.float64) - np.asarray(samples[b], dtype=np.float64)
        d[np.abs(d) < tie] = 0.0
        p.append(float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
    adjusted = holm(p)
    return {
        frozenset(pair): {"p": p[i], "p_holm": float(adjusted[i])} for i, pair in enumerate(pairs)
    }
