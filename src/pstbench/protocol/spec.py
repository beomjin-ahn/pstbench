from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml

_REQUIRED_TOP_LEVEL = (
    "id",
    "metrics",
    "compare",
    "aggregation",
    "display_precision",
)


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Protocol:
    id: str
    params: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.params[key]


def load_protocol(name: str = "pstbench") -> Protocol:
    raw = resources.files("pstbench.protocol").joinpath(f"{name}.yaml").read_bytes()
    params = yaml.safe_load(raw)
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in params]
    if missing:
        raise ProtocolError(f"protocol {name} is missing sections: {missing}")
    if params["id"] != name:
        raise ProtocolError(f"protocol file {name}.yaml declares id {params['id']!r}")
    return Protocol(id=name, params=params)
