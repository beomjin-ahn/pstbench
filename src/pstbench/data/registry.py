from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pair:
    id: str
    content: str
    style: str
    reference: str


@dataclass(frozen=True)
class Track:
    name: str
    dataset: str
    input_encoding: str
    pairs: tuple[Pair, ...]

    @property
    def splits(self) -> tuple[str, ...]:
        paths = [p.content for p in self.pairs] + [p.style for p in self.pairs]
        paths += [p.reference for p in self.pairs]
        return tuple(sorted({path.split("/")[0] for path in paths}))


def _pst50_pairs(content_split: str):
    return tuple(
        Pair(
            id=f"pst50-{i:03d}",
            content=f"{content_split}/in{i}.png",
            style=f"paired_style/tar{i}.png",
            reference=f"paired_gt/gt{i}.png",
        )
        for i in range(1, 51)
    )


def _acetone_pairs():
    return tuple(
        Pair(
            id=f"acetone-{i:04d}",
            content=f"raw_image/{i:06d}.png",
            style=f"ref_image/{i:06d}.png",
            reference=f"gt_image/{i:06d}.png",
        )
        for i in range(1024)
    )


TRACKS: dict[str, Track] = {
    t.name: t
    for t in (
        Track("pst50-paired", "pst50", "rec709", _pst50_pairs("content_709")),
        Track("pst50-paired-log", "pst50", "log", _pst50_pairs("content_log")),
        Track("acetone-transfer", "acetone", "srgb", _acetone_pairs()),
    )
}


def get_track(name: str) -> Track:
    try:
        return TRACKS[name]
    except KeyError:
        raise KeyError(f"unknown track {name!r}; available: {', '.join(TRACKS)}") from None
