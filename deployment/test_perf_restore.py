"""Rollback refuses later edits and corrupt backups before any project writes."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "perf_20261012" / "restore_baseline.py"
spec = importlib.util.spec_from_file_location("perf_restore", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_tree(tmp_path):
    root = tmp_path / "project"
    base = root / "deployment" / "perf_20261012"
    old, new = {}, {}
    for index, name in enumerate(module.SOURCES):
        before, after = f"baseline-{index}".encode(), f"optimized-{index}".encode()
        saved = base / "baseline" / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(before)
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(after)
        old[name] = hashlib.sha256(before).hexdigest()
        new[name] = hashlib.sha256(after).hexdigest()
    (base / "baseline_manifest.json").write_text(json.dumps({"files": old}))
    (base / "optimized_manifest.json").write_text(json.dumps({"files": new}))
    return root, base


def test_check_then_restore_is_scoped_and_idempotent(tmp_path):
    root, base = make_tree(tmp_path)
    other = root / "configs" / "user.env"
    other.parent.mkdir()
    other.write_text("user config")
    before = [(root / name).read_bytes() for name in module.SOURCES]
    module.restore(root, base, True)
    assert [(root / name).read_bytes() for name in module.SOURCES] == before
    module.restore(root, base)
    module.restore(root, base)
    for name in module.SOURCES:
        assert (root / name).read_bytes() == (base / "baseline" / name).read_bytes()
    assert other.read_text() == "user config"


@pytest.mark.parametrize("conflict", ["current_source", "saved_backup"])
def test_conflict_refuses_all_writes(tmp_path, conflict):
    root, base = make_tree(tmp_path)
    last = module.SOURCES[-1]
    target = (root / last) if conflict == "current_source" else (base / "baseline" / last)
    target.write_text("later user edit or corrupt backup")
    before = [(root / name).read_bytes() for name in module.SOURCES]
    with pytest.raises(RuntimeError):
        module.restore(root, base)
    assert [(root / name).read_bytes() for name in module.SOURCES] == before
