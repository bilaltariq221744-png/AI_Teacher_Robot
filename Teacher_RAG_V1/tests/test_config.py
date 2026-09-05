"""Tests for src/config.py (Phase 0 scaffold)."""
from __future__ import annotations

import pytest

from src.config import DEFAULT_CONFIG_PATH, Config, ConfigError


def test_default_config_loads():
    cfg = Config.load()
    assert cfg.pc["dpi"] == 300
    assert cfg.pc["ocr_langs"] == ["eng"]
    assert cfg.pi["vector_top_k"] == 20
    assert cfg.pi["llm"]["model"] == "qwen2.5:1.5b"


def test_config_requires_pc_and_pi_sections(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: 1\n")
    with pytest.raises(ConfigError):
        Config.load(bad)


def test_config_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        Config.load(tmp_path / "nope.yaml")


def test_default_config_path_points_at_repo_config():
    assert DEFAULT_CONFIG_PATH.name == "config.yaml"
    assert DEFAULT_CONFIG_PATH.exists()
