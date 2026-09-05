"""Shared configuration for teacher_rag (PC builder + Pi runtime)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or malformed."""


class Config:
    """Thin wrapper around the parsed config.yaml dict.

    Exposes ``pc`` and ``pi`` sections; the rest of the code reads values with
    plain dict access (``section.get("key", default)``) so keys are easy to add.
    """

    def __init__(self, data: dict[str, Any], source: Path) -> None:
        self._data = data
        self.source = source

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "Config":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        with p.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"config file must contain a YAML mapping: {p}")
        for section in ("pc", "pi"):
            if section not in data:
                raise ConfigError(f"config file must define a '{section}' section: {p}")
        return cls(data, p)

    @property
    def pc(self) -> dict[str, Any]:
        return self._data["pc"]

    @property
    def pi(self) -> dict[str, Any]:
        return self._data["pi"]

    def __repr__(self) -> str:
        return f"Config(source={self.source})"
