#!/usr/bin/env python3
"""Create an immutable training-only statistics view of the audited X5 dataset."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prepare_x5_cleaned_comparison import SOURCE, digest_file, fingerprint


def prepare(previous: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Use a new cohort directory: {output}")
    split_path = SOURCE / "meta/subtask_splits.json"
    supplied = json.loads(split_path.read_text())["splits"]
    splits = {
        "train": supplied["train"],
        "val": supplied["validation"],
        "test": supplied["test"],
    }
    joined = sum(splits.values(), [])
    assert sorted(joined) == list(range(337)), (
        "Episode partitions must be disjoint and exhaustive"
    )
    dataset = previous / "dataset"
    source_files = sorted((SOURCE / "data").rglob("*.parquet"))
    for source in source_files:
        assert digest_file(source) == digest_file(
            dataset / source.relative_to(SOURCE)
        ), "Source frame table changed"
    import pyarrow as pa

    table = pa.concat_tables([pq.read_table(p) for p in source_files])
    ep, task = np.asarray(table["episode_index"]), np.asarray(table["task_index"])
    state = np.asarray(table["observation.state"].to_pylist(), np.float32)
    action = np.asarray(table["action"].to_pylist(), np.float32)
    assert state.shape == action.shape == (382422, 14)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    training = np.isin(ep, splits["train"])
    stats = json.loads((dataset / "meta/stats.json").read_text())
    coverage = {}
    for key, values in [("observation.state", state), ("action", action)]:
        selected = values[training]
        lo, hi = selected.min(0), selected.max(0)
        assert (hi > lo).all(), "Unexpected constant dimension in X5"
        record = {
            "min": lo.tolist(),
            "max": hi.tolist(),
            "mean": selected.mean(0, dtype=np.float64).tolist(),
            "std": selected.std(0, dtype=np.float64).tolist(),
            "count": [len(selected)],
        }
        for q in (0.01, 0.1, 0.5, 0.9, 0.99):
            record[f"q{round(q * 100):02d}"] = np.quantile(selected, q, axis=0).tolist()
        stats[key] = record
        coverage[key] = {
            split: {
                "rows": int(np.isin(ep, ids).sum()),
                "any_dimension_outside_training_range": float(
                    ((values[np.isin(ep, ids)] < lo) | (values[np.isin(ep, ids)] > hi))
                    .any(-1)
                    .mean()
                ),
            }
            for split, ids in splits.items()
        }
    output.mkdir(parents=True)
    shutil.copytree(dataset, output / "dataset")
    (output / "dataset/meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (output / "packed_cache").symlink_to(
        (previous / "packed_cache").resolve(), target_is_directory=True
    )
    partitions = {
        "version": "x5-native-v2",
        "splits": splits,
        "source": str(split_path),
        "source_sha256": digest_file(split_path),
        "sampling": "Existing chronological whole-episode split; no phase labels used",
        "legacy_checkpoint_has_seen_all_splits": True,
        "split_summary": {
            s: {
                "episodes": len(ids),
                "frames": int(np.isin(ep, ids).sum()),
                "task_frames": {
                    str(t): int((np.isin(ep, ids) & (task == t)).sum())
                    for t in np.unique(task)
                },
            }
            for s, ids in splits.items()
        },
    }
    (output / "splits.json").write_text(json.dumps(partitions, indent=2) + "\n")
    old = json.loads((previous / "source_audit.json").read_text())
    receipt = {
        **old,
        "source_fingerprint": fingerprint(SOURCE),
        "normalization": "min_max fitted to train episodes only; native radians",
        "stats_sha256": digest_file(output / "dataset/meta/stats.json"),
        "split_sha256": digest_file(output / "splits.json"),
        "statistics_frames": int(training.sum()),
        "coverage": coverage,
        "video_integrity_note": "Reuses previously SHA-verified video cache; this preparation rechecks numeric tables and uses unchanged packed shards. Does not rehash videos.",
        "previous_audit": str(previous / "source_audit.json"),
    }
    (output / "source_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cohort": str(output),
                "split_summary": partitions["split_summary"],
                "coverage": coverage,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.previous, a.output)
