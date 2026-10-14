from __future__ import annotations

from pstbench.data.hub import DataIntegrityError, HubDataset

__all__ = [
    "DATASET",
    "DataIntegrityError",
    "SPLITS",
    "fetch",
    "manifest",
    "root",
    "verify",
]

SPLITS = ("content_709", "content_log", "paired_gt", "paired_style")

DATASET = HubDataset("pst50", "PST50", SPLITS)
manifest = DATASET.manifest
root = DATASET.root
verify = DATASET.verify


def fetch(splits=("content_709", "paired_style", "paired_gt")):
    return DATASET.fetch(splits)
