from __future__ import annotations

import numpy as np

H, W = 48, 64


def _gradient() -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W]
    img = np.stack([x / (W - 1), y / (H - 1), 1 - x / (W - 1) * 0.5], axis=-1)
    return img


def _patches() -> np.ndarray:
    colors = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.8],
            [0.9, 0.8, 0.1],
            [0.5, 0.5, 0.5],
            [0.95, 0.95, 0.95],
            [0.05, 0.05, 0.05],
            [0.6, 0.3, 0.7],
        ]
    )
    img = np.zeros((H, W, 3))
    for k, c in enumerate(colors):
        r, q = divmod(k, 4)
        img[r * (H // 2) : (r + 1) * (H // 2), q * (W // 4) : (q + 1) * (W // 4)] = c
    return img


def _texture() -> np.ndarray:
    rng = np.random.default_rng(1234)
    y, x = np.mgrid[0:H, 0:W]
    base = 0.5 + 0.25 * np.sin(x / 3.0)[..., None] * np.cos(y / 5.0)[..., None]
    return np.clip(base * np.array([1.0, 0.8, 0.6]) + rng.normal(0, 0.05, (H, W, 3)), 0, 1)


def _lowlight() -> np.ndarray:
    return _gradient() ** 2.2 * 0.25


def _grade(img: np.ndarray) -> np.ndarray:
    g = img**0.9 * np.array([1.05, 0.97, 0.90])
    lum = g.mean(axis=-1, keepdims=True)
    return np.clip(0.85 * g + 0.15 * lum, 0, 1)


def _u8(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)


def make_pairs() -> dict[str, dict[str, np.ndarray]]:
    contents = {
        "fx-gradient": _gradient(),
        "fx-patches": _patches(),
        "fx-texture": _texture(),
        "fx-lowlight": _lowlight(),
    }
    styles = list(contents.values())[::-1]
    return {
        pid: {"content": _u8(c), "style": _u8(_grade(s)), "reference": _u8(_grade(c))}
        for (pid, c), s in zip(contents.items(), styles, strict=True)
    }


def fixture_methods(content: np.ndarray) -> dict[str, np.ndarray]:
    shifted = np.clip(content.astype(np.int16) + np.array([6, -4, 3]), 0, 255).astype(np.uint8)
    return {"identity": content, "shifted": shifted}
