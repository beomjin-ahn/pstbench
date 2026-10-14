from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path


class DataIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class HubDataset:
    def __init__(self, name: str, title: str, splits: tuple[str, ...]):
        self.name = name
        self.title = title
        self.splits = splits

    def manifest(self) -> dict:
        raw = resources.files("pstbench.data.manifests").joinpath(f"{self.name}.json").read_text()
        return json.loads(raw)

    def root(self) -> Path:
        from pstbench.data import data_root

        return data_root() / self.name

    def fetch(self, splits) -> Path:
        from huggingface_hub import snapshot_download

        bad = [s for s in splits if s not in self.splits]
        if bad:
            raise ValueError(
                f"unknown {self.title} splits {bad}; available: {', '.join(self.splits)}"
            )
        m = self.manifest()
        snapshot_download(
            repo_id=m["dataset"],
            repo_type="dataset",
            revision=m["revision"],
            allow_patterns=[f"{s}/*" for s in splits],
            local_dir=self.root(),
        )
        self.verify(splits)
        return self.root()

    def verify(self, splits) -> None:
        m = self.manifest()
        base = self.root()
        cache_path = base / ".pstbench-verified.json"
        try:
            cache = json.loads(cache_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}
        problems = []
        for rel, meta in m["files"].items():
            if rel.split("/")[0] not in splits:
                continue
            p = base / rel
            if not p.is_file():
                problems.append(f"missing: {rel}")
                continue
            st = p.stat()
            key = f"{st.st_size}:{st.st_mtime_ns}"
            if cache.get(rel) == [key, meta["sha256"]]:
                continue
            if st.st_size != meta["size"] or _sha256(p) != meta["sha256"]:
                problems.append(f"hash mismatch: {rel}")
                continue
            cache[rel] = [key, meta["sha256"]]
        try:
            cache_path.write_text(json.dumps(cache))
        except OSError:
            pass
        if problems:
            shown = "\n  ".join(problems[:10])
            more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
            raise DataIntegrityError(
                f"{self.title} at {base} does not match revision {m['revision'][:7]}:\n  "
                f"{shown}{more}\nRun: pstbench data fetch {self.name} --splits {' '.join(splits)}"
            )
