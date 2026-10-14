from __future__ import annotations

import io as _io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

_HIGH_BIT_DEPTH_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


class ImageFormatError(ValueError):
    pass


@dataclass
class ImageInfo:
    path: str
    mode: str
    size: tuple[int, int]
    exif_transposed: bool = False
    icc_converted_from: str | None = None


def _is_srgb_profile(profile: ImageCms.ImageCmsProfile) -> bool:
    desc = ImageCms.getProfileDescription(profile) or ""
    return "srgb" in desc.lower()


def read_image(path: str | Path) -> tuple[np.ndarray, ImageInfo]:
    path = Path(path)
    with Image.open(path) as im:
        rawmode = im.tile[0][3] if im.tile else None
        if isinstance(rawmode, str) and ";16" in rawmode:
            raise ImageFormatError(f"{path}: {rawmode} (more than 8 bits) is not supported")
        im.load()
        info = ImageInfo(path=str(path), mode=im.mode, size=im.size)

        if im.mode in _HIGH_BIT_DEPTH_MODES:
            raise ImageFormatError(f"{path}: {im.mode} (more than 8 bits) is not supported")
        if im.mode not in ("RGB", "P", "RGBA"):
            raise ImageFormatError(f"{path}: mode {im.mode} is not supported; expected RGB")

        if im.getexif().get(0x0112, 1) != 1:
            im = ImageOps.exif_transpose(im)
            info.exif_transposed = True

        if im.mode == "P":
            im = im.convert("RGBA")
        if im.mode == "RGBA":
            alpha = np.asarray(im.getchannel("A"))
            if alpha.min() < 255:
                raise ImageFormatError(f"{path}: has transparent pixels")
            im = im.convert("RGB")

        icc = im.info.get("icc_profile")
        if icc:
            src = ImageCms.ImageCmsProfile(_io.BytesIO(icc))
            if not _is_srgb_profile(src):
                dst = ImageCms.createProfile("sRGB")
                im = ImageCms.profileToProfile(
                    im,
                    src,
                    dst,
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    outputMode="RGB",
                )
                info.icc_converted_from = ImageCms.getProfileDescription(src).strip()

        arr = np.asarray(im, dtype=np.uint8).copy()
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ImageFormatError(f"{path}: decoded to shape {arr.shape}, expected HxWx3")
    return arr, info


def write_png(path: str | Path, img: np.ndarray) -> None:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ImageFormatError(f"write_png expects uint8 HxWx3, got {img.dtype} {img.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, format="PNG", compress_level=6)
