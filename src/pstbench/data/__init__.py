from __future__ import annotations

import os
from pathlib import Path

from pstbench.data.registry import TRACKS, Pair, Track, get_track

__all__ = ["TRACKS", "Pair", "Track", "get_dataset", "get_track", "data_root"]


def data_root() -> Path:
    env = os.environ.get("PSTBENCH_DATA")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "pstbench" / "data"


def get_dataset(name: str):
    if name == "pst50":
        from pstbench.data import pst50

        return pst50
    if name == "acetone":
        from pstbench.data import acetone

        return acetone
    raise KeyError(f"unknown dataset {name!r}; available: pst50, acetone")
