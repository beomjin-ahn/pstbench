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

SPLITS = ("raw_image", "ref_image", "gt_image")

DATASET = HubDataset("acetone", "AceTone-Bench-Transfer", SPLITS)
manifest = DATASET.manifest
root = DATASET.root
verify = DATASET.verify


def fetch(splits=("raw_image", "ref_image", "gt_image")):
    return DATASET.fetch(splits)
