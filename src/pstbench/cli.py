from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pstbench import __version__


def _cmd_data_fetch(args) -> int:
    from pstbench.data import get_dataset
    from pstbench.data.hub import DataIntegrityError

    try:
        ds = get_dataset(args.dataset)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2
    try:
        root = ds.fetch(args.splits) if args.splits else ds.fetch()
    except (ValueError, DataIntegrityError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{ds.DATASET.title} verified at {root}")
    return 0


def _progress(label: str):
    def progress(i, n, pair_id):
        print(f"\r{label} [{i + 1}/{n}] {pair_id}", end="", file=sys.stderr, flush=True)
        if i + 1 == n:
            print(file=sys.stderr)

    return progress


def _score_and_report(args, track, outputs: Path, name: str) -> int:
    from pstbench.data import get_dataset
    from pstbench.data.hub import DataIntegrityError
    from pstbench.protocol.spec import load_protocol
    from pstbench.report import format_table
    from pstbench.scorer import OutputSetError, score_track, write_run

    protocol = load_protocol()
    run_dir = Path(args.results_dir) / track.name / name
    if run_dir.exists():
        print(f"error: {run_dir} exists; choose another --name or remove it", file=sys.stderr)
        return 2
    try:
        result = score_track(
            track,
            outputs,
            name,
            protocol,
            device=args.device,
            allow_resize=args.allow_resize,
            progress=_progress("score"),
        )
    except (OutputSetError, DataIntegrityError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if result.failures:
        print(f"{len(result.failures)} pair(s) failed; no record written:", file=sys.stderr)
        for pid, why in result.failures.items():
            print(f"  {pid}: {why}", file=sys.stderr)
        return 1
    ds = get_dataset(track.dataset)
    m = ds.manifest()
    write_run(result, track, protocol, run_dir, {"name": m["dataset"], "revision": m["revision"]})
    print(
        f"track: {track.name}   protocol: {protocol.id}   "
        f"dataset: {m['dataset']}@{m['revision'][:7]}   n={len(result.rows)}"
    )
    print(format_table([(name, result.summary)], protocol["display_precision"]))
    if result.deviations:
        print(f"\ndeviations from {protocol.id}: {len(result.deviations)} (see run.json)")
    print(f"\n{run_dir}/  (metrics.csv, summary.json, run.json)")
    return 0


def _cmd_score(args) -> int:
    from pstbench.data import get_track

    try:
        track = get_track(args.track)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2
    return _score_and_report(args, track, Path(args.outputs), args.name)


def _cmd_compare(args) -> int:
    import csv

    from pstbench.protocol.spec import load_protocol
    from pstbench.stats import paired_diff_ci, paired_tests

    runs = []
    for d in args.runs:
        d = Path(d)
        try:
            meta = json.loads((d / "run.json").read_text())
        except FileNotFoundError:
            print(f"error: {d} has no run.json", file=sys.stderr)
            return 2
        with open(d / "metrics.csv", newline="") as f:
            rows = {r["pair_id"]: r for r in csv.DictReader(f)}
        runs.append((meta["method"], meta, rows, str(d)))
    names = [n for n, *_ in runs]
    dup = {n for n in names if names.count(n) > 1}
    runs = [(rid if n in dup else n, m, r) for n, m, r, rid in runs]
    keys = {(m["protocol"], m["track"], m["dataset"]["revision"]) for _, m, _ in runs}
    if len(keys) > 1:
        print(
            "error: the runs do not share one protocol, track and dataset revision:",
            file=sys.stderr,
        )
        for name, m, _ in runs:
            print(
                f"  {name}: {m['protocol']} {m['track']} {m['dataset']['revision']}",
                file=sys.stderr,
            )
        return 2
    protocol = load_protocol(runs[0][1]["protocol"])
    ties = protocol["compare"]["ties"]
    agg = protocol["aggregation"]
    pair_ids = sorted(set.intersection(*(set(r) for _, _, r in runs)))
    if not pair_ids:
        print("error: the runs share no pair ids", file=sys.stderr)
        return 2
    metric_cols = [
        c for c in next(iter(runs[0][2].values())) if c not in ("pair_id", "psnr_capped")
    ]
    if args.metrics:
        metric_cols = [c for c in metric_cols if c in args.metrics]
    first = runs[0][0]
    w = max(14, *(len(n) for n, _, _ in runs))
    print(
        f"protocol {runs[0][1]['protocol']}   track {runs[0][1]['track']}   "
        f"n = {len(pair_ids)} pairs   "
        f"Holm over {len(runs) * (len(runs) - 1) // 2} pair(s) of runs per metric"
    )
    for name, m, _ in runs:
        if m.get("deviations"):
            print(f"  {name}: {len(m['deviations'])} deviation(s) from the protocol (see run.json)")
    for metric in metric_cols:
        samples = {}
        for name, _, rows in runs:
            try:
                samples[name] = np.array([float(rows[p][metric]) for p in pair_ids])
            except (KeyError, ValueError):
                samples = None
                break
        if not samples:
            continue
        tests = paired_tests(samples, tie=ties.get(metric, 0.0))
        print(f"\n{metric}")
        print(
            f"  {'method':{w}s} {'mean':>10s} {'diff vs ' + first:>18s} "
            f"{'95% CI':>20s} {'p':>8s} {'p (Holm)':>9s}"
        )
        for name in samples:
            mean = float(samples[name].mean())
            if name == first:
                print(f"  {name:{w}s} {mean:10.4f}")
                continue
            d, lo, hi = paired_diff_ci(
                samples[name],
                samples[first],
                agg["ci_level"],
                agg["bootstrap_resamples"],
                agg["bootstrap_seed"],
            )
            t = tests[frozenset((name, first))]
            print(
                f"  {name:{w}s} {mean:10.4f} {d:+18.4f} [{lo:+8.4f}, {hi:+8.4f}] "
                f"{t['p']:8.4f} {t['p_holm']:9.4f}"
            )
    return 0


def _cmd_demo(args) -> int:
    import tempfile

    from pstbench import io
    from pstbench.data.registry import Pair, Track
    from pstbench.fixtures import fixture_methods, make_pairs
    from pstbench.protocol.spec import load_protocol
    from pstbench.report import format_table
    from pstbench.scorer import score_track

    protocol = load_protocol("pstbench")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        pairs = []
        outputs = {}
        for pid, imgs in make_pairs().items():
            for kind in ("content", "style", "reference"):
                io.write_png(root / kind / f"{pid}.png", imgs[kind])
            pairs.append(
                Pair(pid, f"content/{pid}.png", f"style/{pid}.png", f"reference/{pid}.png")
            )
            for name, img in fixture_methods(imgs["content"]).items():
                io.write_png(Path(tmp) / name / f"{pid}.png", img)
                outputs[name] = Path(tmp) / name
        track = Track("fixtures", "fixtures", "rec709", tuple(pairs))
        rows = [
            (name, score_track(track, out, name, protocol, dataset_root=root).summary)
            for name, out in outputs.items()
        ]
    print(f"demo: {len(pairs)} synthetic 64x48 pairs, protocol {protocol.id}")
    print(format_table(rows, protocol["display_precision"]))
    print(
        "\nNext: pstbench data fetch pst50 && "
        "pstbench score --track pst50-paired --outputs DIR --name M"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pstbench")
    p.add_argument("--version", action="version", version=f"pstbench {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    fetch = data_sub.add_parser("fetch")
    fetch.add_argument("dataset")
    fetch.add_argument(
        "--splits",
        nargs="+",
        default=None,
    )
    fetch.set_defaults(func=_cmd_data_fetch)

    score = sub.add_parser("score")
    score.add_argument("--track", required=True)
    score.add_argument("--outputs", required=True)
    score.add_argument("--name", required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--results-dir", default="results")
    score.add_argument(
        "--allow-resize",
        action="store_true",
    )
    score.set_defaults(func=_cmd_score)

    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("runs", nargs="+")
    cmp_.add_argument("--metrics", nargs="*", default=None)
    cmp_.set_defaults(func=_cmd_compare)

    demo = sub.add_parser("demo")
    demo.set_defaults(func=_cmd_demo)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
