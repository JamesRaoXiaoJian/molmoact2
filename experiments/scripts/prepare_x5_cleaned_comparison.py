#!/usr/bin/env python3
"""Audit the existing cleaned JPEG cache and stage a matched local training view."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.dataset as pads
from olmo.data.tactile import X5_RGB_IMAGE_KEYS, tactile_keys_for_layout

SOURCE = Path("/mnt/tacumi_data/realman_data/hot_stamp_bag_merged_0918-20_cleaned")
PACKED = Path(
    "/root/RXJ/predecoded_datasets/hot_stamp_bag_merged_0918-20_cleaned_packed"
)
REFERENCE = Path("/root/RXJ/openpi_jax_tactile_mae_plain_0918_20_cleaned")


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def fingerprint(root):
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            st = path.stat()
            h.update(
                f"{path.relative_to(root)}:{st.st_size}:{st.st_mtime_ns}\n".encode()
            )
            if path.suffix == ".json" and path.parent.name == "meta":
                h.update(path.read_bytes())
    return h.hexdigest()


def prepare_static(cohort):
    started = time.monotonic()
    info = json.loads((SOURCE / "meta/info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["fps"]) == (
        337,
        382422,
        30,
    )
    episodes = (
        pads.dataset(SOURCE / "meta/episodes", format="parquet").to_table().to_pylist()
    )
    assert sorted(int(e["episode_index"]) for e in episodes) == list(range(337))
    assert sum(int(e["length"]) for e in episodes) == 382422
    audit = json.loads((REFERENCE / "cache_reuse_audit.json").read_text())
    old_rows = {(int(r["clean_episode"]), r["video_key"]): r for r in audit["shards"]}
    keys = [*X5_RGB_IMAGE_KEYS, *tactile_keys_for_layout("two")]
    original_manifest = json.loads((PACKED / "packed_manifest.json").read_text())
    assert original_manifest["padded_frames"] == 0

    def verify(item):
        episode, key = item
        index = int(episode["episode_index"])
        relative = Path(
            info["video_path"].format(
                video_key=key,
                chunk_index=int(episode[f"videos/{key}/chunk_index"]),
                file_index=int(episode[f"videos/{key}/file_index"]),
            )
        )
        video = SOURCE / relative
        row = old_rows[index, key]
        if digest_file(video) != row["video_sha256"]:
            raise ValueError(
                f"Existing JPEG cache does not match the current source video: {video}"
            )
        assert float(episode[f"videos/{key}/from_timestamp"]) == 0
        stem = PACKED / "packed" / relative.relative_to("videos").with_suffix("")
        offsets = np.load(
            stem.with_suffix(".idx.npy"), mmap_mode="r", allow_pickle=False
        )
        assert len(offsets) == int(episode["length"]) + 1
        assert int(offsets[0]) == 0 and np.all(offsets[1:] > offsets[:-1])
        assert int(offsets[-1]) == stem.with_suffix(".bin").stat().st_size
        return {
            "episode": index,
            "key": key,
            "video_sha256": row["video_sha256"],
            "frames": len(offsets) - 1,
            "jpeg_bytes": int(offsets[-1]),
        }

    tasks = [(episode, key) for episode in episodes for key in keys]
    rows = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for count, row in enumerate(pool.map(verify, tasks), 1):
            rows.append(row)
            if count % 100 == 0 or count == len(tasks):
                print(
                    f"Verified {count}/{len(tasks)} complete source videos and JPEG shards.",
                    flush=True,
                )

    table = pads.dataset(SOURCE / "data", format="parquet").to_table(
        columns=[
            "action",
            "observation.state",
            "index",
            "episode_index",
            "frame_index",
            "timestamp",
        ]
    )
    assert table.num_rows == 382422
    indices = np.asarray(table["index"].to_numpy())
    assert np.array_equal(indices, np.arange(382422))
    stats = json.loads((SOURCE / "meta/stats.json").read_text())
    for name in ("action", "observation.state"):
        values = np.asarray(table[name].to_pylist(), dtype=np.float32)
        assert values.shape == (382422, 14) and np.isfinite(values).all()
        record = {
            "min": values.min(0).tolist(),
            "max": values.max(0).tolist(),
            "mean": values.mean(0, dtype=np.float64).tolist(),
            "std": values.std(0, dtype=np.float64).tolist(),
            "count": [len(values)],
        }
        for q in (0.01, 0.10, 0.50, 0.90, 0.99):
            record[f"q{int(q * 100):02d}"] = np.quantile(values, q, axis=0).tolist()
        stats[name] = record

    dataset = cohort / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    for directory in ("meta", "data"):
        shutil.copytree(SOURCE / directory, dataset / directory, dirs_exist_ok=True)
    (dataset / "meta/stats.json").write_text(json.dumps(stats, indent=2))
    current_fingerprint = fingerprint(SOURCE)
    cache = cohort / "packed_cache"
    cache.mkdir(exist_ok=True)
    (cache / "packed").symlink_to(PACKED / "packed", target_is_directory=True)
    manifest = {
        **original_manifest,
        "dataset_fingerprint": current_fingerprint,
        "validation": "All 1685 source video SHA256 values and exact JPEG frame counts revalidated",
    }
    (cache / "packed_manifest.json").write_text(json.dumps(manifest, indent=2))
    report = {
        "source": str(SOURCE),
        "reference": str(REFERENCE),
        "source_fingerprint": current_fingerprint,
        "reference_fingerprint": original_manifest["dataset_fingerprint"],
        "episodes": 337,
        "frames": 382422,
        "verified_video_shards": len(rows),
        "padded_frames": 0,
        "normalization": "q01/q99 from all 382422 cleaned native state/action rows",
        "stats_sha256": digest_file(dataset / "meta/stats.json"),
        "elapsed_seconds": time.monotonic() - started,
    }
    (cohort / "source_audit.json").write_text(
        json.dumps({**report, "shards": rows}, indent=2)
    )
    print(json.dumps(report, indent=2), flush=True)


def stage(mode, cohort, run_dir, local_root):
    audit = json.loads((cohort / "source_audit.json").read_text())
    if fingerprint(SOURCE) != audit["source_fingerprint"]:
        raise ValueError("Source dataset changed after the complete cache audit.")
    keys = list(X5_RGB_IMAGE_KEYS)
    if mode == "mae":
        keys += tactile_keys_for_layout("two")
    source_cache = cohort / "packed_cache"
    destination = local_root / "packed"
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for key in keys:
        files += [
            (p, destination / p.relative_to(source_cache))
            for p in sorted((source_cache / "packed" / key).rglob("*"))
            if p.is_file()
        ]
    remaining = sum(
        src.stat().st_size
        for src, dst in files
        if not dst.exists() or dst.stat().st_size != src.stat().st_size
    )
    if shutil.disk_usage(local_root).free < remaining + 48 * 2**30:
        raise ValueError(
            "Node cache needs the selected packed streams plus 48 GiB host-memory reserve."
        )
    progress = run_dir / "cache_stage.json"
    copied = 0
    started = time.monotonic()

    def copy(pair):
        src, dst = pair
        dst.parent.mkdir(parents=True, exist_ok=True)
        size = src.stat().st_size
        if not dst.exists() or dst.stat().st_size != size:
            temp = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, temp)
            temp.replace(dst)
        return size

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(copy, pair) for pair in files]
        for count, future in enumerate(as_completed(futures), 1):
            copied += future.result()
            if count % 20 == 0 or count == len(files):
                status = {
                    "status": "staging",
                    "files": count,
                    "total_files": len(files),
                    "gib": copied / 2**30,
                    "seconds": time.monotonic() - started,
                }
                progress.write_text(json.dumps(status, indent=2))
                print(
                    f"Cache staging {count}/{len(files)} files, {copied / 2**30:.2f} GiB.",
                    flush=True,
                )
    manifest = json.loads((source_cache / "packed_manifest.json").read_text())
    manifest.update(
        video_keys=keys,
        shards=337 * len(keys),
        frames=382422 * len(keys),
        jpeg_bytes=sum(src.stat().st_size for src, _ in files if src.suffix == ".bin"),
    )
    (destination / "packed_manifest.json").write_text(json.dumps(manifest, indent=2))
    dataset = local_root / "data/local/hot_stamp_bag_merged_0918-20_cleaned"
    shutil.copytree(cohort / "dataset", dataset, dirs_exist_ok=True)
    assert digest_file(dataset / "meta/stats.json") == audit["stats_sha256"]
    status = {
        "status": "complete",
        "files": len(files),
        "gib": copied / 2**30,
        "seconds": time.monotonic() - started,
        "video_keys": keys,
        "dataset_root": str(dataset),
        "packed_root": str(destination),
    }
    progress.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--prepare-static", action="store_true")
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--mode", choices=("rgb", "mae"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--local-root", type=Path)
    args = parser.parse_args()
    args.cohort.mkdir(parents=True, exist_ok=True)
    if args.prepare_static:
        prepare_static(args.cohort)
    elif args.stage:
        if args.mode is None or args.run_dir is None or args.local_root is None:
            parser.error("staging requires mode, run-dir and local-root")
        stage(args.mode, args.cohort, args.run_dir, args.local_root)
    else:
        parser.error("Choose --prepare-static or --stage")


if __name__ == "__main__":
    main()
