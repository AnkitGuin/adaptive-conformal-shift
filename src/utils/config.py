"""YAML configuration loading with inheritance and command-line overrides.

A config file may contain a top-level ``inherit: other.yaml`` key; the parent
is loaded first (relative to the child file) and the child is deep-merged on
top.  Overrides from the command line use dotted keys, e.g.
``--set experiment.seeds=[0,1] conformal.alpha=0.2``.

All paths inside the configuration are interpreted relative to the
repository root (the directory that contains ``configs/``), never as
absolute machine-specific paths.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = (REPO_ROOT / path) if not path.exists() else path.resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def set_by_dotted_key(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        set_by_dotted_key(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
