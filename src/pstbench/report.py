from __future__ import annotations

_LABELS = {
    "psnr": "PSNR↑",
    "ssim": "SSIM↑",
    "lpips": "LPIPS↓",
    "deltae00": "ΔE00↓",
    "hcorr_gt": "H-Corr(gt)↑",
    "hcorr_style": "H-Corr(style)↑",
}


def format_table(rows: list[tuple[str, dict]], precision: dict[str, int]) -> str:
    metrics = [m for m in _LABELS if any(m in s for _, s in rows)]
    header = ["method", *(_LABELS[m] for m in metrics)]
    body = []
    for name, s in rows:
        cells = [name]
        for m in metrics:
            if m not in s:
                cells.append("–")
                continue
            d = precision.get(m, 4)
            half = (s[m]["ci_high"] - s[m]["ci_low"]) / 2
            cells.append(f"{s[m]['mean']:.{d}f} ± {half:.{d}f}")
        body.append(cells)
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in [header, *body]]
    return "\n".join(line.rstrip() for line in lines)
