"""Strict YAML loading — reject duplicate top-level keys."""


import pytest

from src.runners.config_utils import REPO_ROOT, load_yaml


def test_all_config_yamls_load_without_duplicate_keys():
    yaml_dir = REPO_ROOT / "configs"
    paths = list(yaml_dir.rglob("*.yaml")) + list(yaml_dir.rglob("*.yml"))
    assert paths, "expected YAML configs under configs/"
    for path in sorted(paths):
        load_yaml(path.relative_to(REPO_ROOT))


def test_duplicate_yaml_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("temperature: 0.6\nrepetition_penalty: 1.0\nrepetition_penalty: 1.05\n")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_yaml(bad)
