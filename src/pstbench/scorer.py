from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from pstbench import io
from pstbench.data import Track
from pstbench.metrics import build_suite
from pstbench.protocol.spec import Protocol
from pstbench.stats import bootstrap_ci


class OutputSetError(ValueError):
    pass


@dataclass
class ScoreResult:
    track: str
    name: str
    rows: list[dict]
    summary: dict
    deviations: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)


def _dataset_root(track: Track) -> Path:
    from pstbench.data import get_dataset

    ds = get_dataset(track.dataset)
    ds.verify(track.splits)
    return ds.root()


def check_output_set(track: Track, outputs: Path) -> None:
    expected = {p.id for p in track.pairs}
    found = {f.stem for f in outputs.glob("*.png")}
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:

        def show(ids):
            return ", ".join(ids[:5]) + (" ..." if len(ids) > 5 else "")

        msg = [f"outputs in {outputs} do not match track {track.name}:"]
        if missing:
            msg.append(f"  missing {len(missing)}: {show(missing)}")
        if extra:
            msg.append(f"  unexpected {len(extra)}: {show(extra)}")
        msg.append("  expected files named <pair_id>.png, e.g. pst50-001.png")
        raise OutputSetError("\n".join(msg))


def _resize_like(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    h, w = ref.shape[:2]
    return np.asarray(Image.fromarray(img).resize((w, h), Image.Resampling.BICUBIC))


def summarize(rows: list[dict], protocol: Protocol, columns) -> dict:
    agg = protocol["aggregation"]
    out = {}
    for m in columns:
        vals = [r[m] for r in rows if r.get(m) is not None]
        if not vals:
            continue
        mean, lo, hi = bootstrap_ci(
            vals, agg["ci_level"], agg["bootstrap_resamples"], agg["bootstrap_seed"]
        )
        out[m] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)}
    return out


def score_track(
    track: Track,
    outputs: Path,
    name: str,
    protocol: Protocol,
    device: str = "cpu",
    allow_resize: bool = False,
    dataset_root: Path | None = None,
    progress=None,
) -> ScoreResult:
    outputs = Path(outputs)
    check_output_set(track, outputs)
    root = dataset_root or _dataset_root(track)
    suite = build_suite(protocol, device=device)
    rows, deviations, failures = [], [], {}
    for i, pair in enumerate(track.pairs):
        if progress:
            progress(i, len(track.pairs), pair.id)
        try:
            pred, info = io.read_image(outputs / f"{pair.id}.png")
        except io.ImageFormatError as e:
            failures[pair.id] = str(e)
            continue
        ref, _ = io.read_image(root / pair.reference)
        style, _ = io.read_image(root / pair.style)
        if pred.shape != ref.shape:
            if not allow_resize:
                failures[pair.id] = f"output {pred.shape[:2]} != reference {ref.shape[:2]}"
                continue
            pred = _resize_like(pred, ref)
            deviations.append(f"resized_by_kit:{pair.id}")
        if info.icc_converted_from:
            deviations.append(f"icc_converted:{pair.id}:{info.icc_converted_from}")
        row = {"pair_id": pair.id, **suite.score_paired(pred, ref, style)}
        rows.append(row)
    summary = summarize(rows, protocol, suite.columns)
    result = ScoreResult(track.name, name, rows, summary, deviations, failures)
    result.columns = list(suite.columns)
    return result


def write_run(
    result: ScoreResult, track: Track, protocol: Protocol, run_dir: Path, dataset: dict
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    extras = ["psnr_capped"] if any("psnr_capped" in r for r in result.rows) else []
    cols = ["pair_id", *result.columns, *extras]
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in result.rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    (run_dir / "summary.json").write_text(json.dumps(result.summary, indent=1) + "\n")
    run = {
        "protocol": protocol.id,
        "track": track.name,
        "input_encoding": track.input_encoding,
        "dataset": dataset,
        "method": result.name,
        "deviations": result.deviations,
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=1) + "\n")
