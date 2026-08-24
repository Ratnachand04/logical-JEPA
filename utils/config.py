"""YAML configuration loading with dotted access and CLI overrides."""

from __future__ import annotations

import copy
import os
from typing import Any, Iterable, Mapping

import yaml


class Config(dict):
    """A dict that also supports attribute and dotted-key access.

    ``cfg.model.embed_dim`` and ``cfg["model.embed_dim"]`` are equivalent, which
    keeps train/evaluate scripts readable without pulling in a heavy config
    framework.
    """

    def __init__(self, mapping: Mapping[str, Any] | None = None):
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, Mapping) and not isinstance(value, Config):
            value = Config(value)
        elif isinstance(value, list):
            value = [Config(v) if isinstance(v, Mapping) else v for v in value]
        super().__setitem__(key, value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Look up ``a.b.c`` returning ``default`` if any level is missing."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        """Assign ``a.b.c = value``, creating intermediate dicts as needed."""
        parts = dotted.split(".")
        node: Config = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Config):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = value

    def to_dict(self) -> dict:
        """Plain-dict copy, safe to json/yaml dump."""
        out: dict = {}
        for key, value in self.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                out[key] = value
        return out


def _coerce(text: str) -> Any:
    """Turn a CLI override string into a python scalar via the YAML parser."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """Recursively merge ``override`` on top of ``base``."""
    merged = Config(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | os.PathLike, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML config file and apply ``key=value`` CLI overrides.

    Supports a ``_base_`` key holding a path (relative to the current file) to
    inherit from, which is how ``configs/ablations.yaml`` reuses ``loco.yaml``.
    """
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base_ref = raw.pop("_base_", None)
    if base_ref:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base_ref)
        cfg = deep_merge(load_config(base_path), raw)
    else:
        cfg = Config(raw)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not in key=value form")
        key, _, value = item.partition("=")
        cfg.set_path(key.strip(), _coerce(value.strip()))

    return cfg


def save_config(cfg: Config, path: str | os.PathLike) -> None:
    """Dump a config next to the checkpoints so runs stay self-describing."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = cfg.to_dict() if isinstance(cfg, Config) else dict(cfg)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
