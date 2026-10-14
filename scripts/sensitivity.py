from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity

from pstbench import io
from pstbench.color import srgb_to_lab, srgb_to_linear
from pstbench.data import get_dataset, get_track
from pstbench.metrics.deltae import ciede2000
from pstbench.metrics.hcorr import hcorr as hcorr_rgb
from pstbench.metrics.lpips_ import LPIPS
from pstbench.metrics.psnr import psnr as psnr_u8
from pstbench.metrics.ssim import ssim as ssim_wang


def psnr_linear(p, g):
    lp = srgb_to_linear(p.astype(np.float64) / 255.0)
    lg = srgb_to_linear(g.astype(np.float64) / 255.0)
    return float(10 * np.log10(1.0 / np.mean((lp - lg) ** 2)))


def ssim_skimage_default(p, g):
    return float(structural_similarity(p, g, channel_axis=-1, data_range=255))


def de(p, g, formula):
    lp, lg = srgb_to_lab(p), srgb_to_lab(g)
    if formula == "76":
        return float(np.sqrt(((lp - lg) ** 2).sum(-1)).mean())
    return float(ciede2000(lp, lg).mean())


def hcorr_hsv64(p, g):
    import cv2

    ph, gh = cv2.cvtColor(p, cv2.COLOR_RGB2HSV), cv2.cvtColor(g, cv2.COLOR_RGB2HSV)
    vals = []
    for c, hi in ((0, 180), (1, 256), (2, 256)):
        a = cv2.calcHist([ph], [c], None, [64], [0, hi]).ravel()
        b = cv2.calcHist([gh], [c], None, [64], [0, hi]).ravel()
        vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(vals))


def pil_resize(img, size, resample):
    return np.asarray(Image.fromarray(img).resize(size, resample))


def torch_resize(img, size):
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(size[1], size[0]), mode="bilinear", align_corners=False)
    return np.rint(t[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


class Lpips:
    def __init__(self, device):
        self.alex = LPIPS("alex", device=device)
        self.vgg = LPIPS("vgg", device=device)


def score_pair(pred, gt, style, lp: Lpips, pred_orig=None):
    if pred_orig is None:
        pred_orig = pred
    rows = []

    def add(variant, metric, value):
        rows.append((variant, metric, float(value)))

    add("native", "psnr", psnr_u8(pred, gt)[0])
    add("native", "psnr_linear", psnr_linear(pred, gt))
    add("native", "ssim", ssim_wang(pred, gt))
    add("native", "ssim_skimage_default", ssim_skimage_default(pred, gt))
    add("native", "lpips_alex", lp.alex(pred, gt))
    add("native", "lpips_vgg", lp.vgg(pred, gt))
    add("native", "lpips_alex_unnormalized", lp.alex(pred, gt, unit_range=True))
    add("native", "de00", de(pred, gt, "00"))
    add("native", "de76", de(pred, gt, "76"))
    add("native", "hcorr_rgb256_gt", hcorr_rgb(pred, gt, 256))
    add("native", "hcorr_rgb64_gt", hcorr_rgb(pred, gt, 64))
    add("native", "hcorr_hsv64_gt", hcorr_hsv64(pred, gt))
    add("native", "hcorr_rgb256_style", hcorr_rgb(pred, style, 256))

    p, g = (
        pil_resize(pred_orig, (256, 256), Image.BILINEAR),
        pil_resize(gt, (256, 256), Image.BILINEAR),
    )
    add("sq256", "psnr", psnr_u8(p, g)[0])
    add("sq256", "psnr_linear", psnr_linear(p, g))
    add("sq256", "ssim", ssim_wang(p, g))
    add("sq256", "lpips_alex", lp.alex(p, g))
    add("sq256", "lpips_vgg", lp.vgg(p, g))
    add("sq256", "de00", de(p, g, "00"))
    add("sq256", "de76", de(p, g, "76"))
    add("sq256", "hcorr_rgb256_gt", hcorr_rgb(p, g, 256))
    add("sq256", "hcorr_hsv64_gt", hcorr_hsv64(p, g))

    p, g = torch_resize(pred_orig, (256, 256)), torch_resize(gt, (256, 256))
    add("sq256_torch_noaa", "psnr", psnr_u8(p, g)[0])
    add("sq256_torch_noaa", "ssim", ssim_wang(p, g))
    add("sq256_torch_noaa", "lpips_alex", lp.alex(p, g))

    h, w = gt.shape[:2]
    size = (512, round(512 * h / w))
    p, g = pil_resize(pred_orig, size, Image.BICUBIC), pil_resize(gt, size, Image.BICUBIC)
    add("w512", "psnr", psnr_u8(p, g)[0])
    add("w512", "ssim", ssim_wang(p, g))
    add("w512", "lpips_alex", lp.alex(p, g))
    add("w512", "de00", de(p, g, "00"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--track", default="pst50-paired")
    ap.add_argument("methods", nargs="+")
    args = ap.parse_args(argv)
    track = get_track(args.track)
    ds = get_dataset(track.dataset)
    splits = track.splits
    if any(spec.endswith("=log:identity") for spec in args.methods):
        splits = (*splits, "content_log")
    ds.verify(splits)
    root = ds.root()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lp = Lpips(args.device)
    for spec in args.methods:
        name, _, d = spec.partition("=")
        dst = out / f"{name}.csv"
        if dst.exists():
            print(f"skip {name}: {dst} exists", file=sys.stderr)
            continue
        t0 = time.time()
        tmp = dst.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["method", "pair_id", "upsampled", "variant", "metric", "value"])
            for pair in track.pairs:
                gt, _ = io.read_image(root / pair.reference)
                style, _ = io.read_image(root / pair.style)
                if d == "log:identity":
                    pred, _ = io.read_image(
                        root / pair.content.replace("content_709", "content_log")
                    )
                else:
                    pred, _ = io.read_image(Path(d) / f"{pair.id}.png")
                pred_orig = pred
                up = int(pred.shape != gt.shape)
                if up:
                    pred = pil_resize(pred, (gt.shape[1], gt.shape[0]), Image.BICUBIC)
                for variant, metric, value in score_pair(pred, gt, style, lp, pred_orig):
                    wcsv.writerow([name, pair.id, up, variant, metric, f"{value:.8g}"])
        tmp.rename(dst)
        print(f"{name}: {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
