import io
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from olmo.data.packed_images import PackedImageReader
from PIL import Image


def _cache(tmp_path, padded=0):
    key = "observation.images.left_d405"
    (tmp_path / "packed_manifest.json").write_text(
        json.dumps(
            {
                "format": "openpi-predecoded-packed-v1",
                "padded_frames": padded,
                "fps": 30,
                "video_keys": [key],
            }
        )
    )
    stem = tmp_path / "packed" / key / "chunk-000/file-000"
    stem.parent.mkdir(parents=True)
    offsets = [0]
    with stem.with_suffix(".bin").open("wb") as stream:
        for value in (20, 80, 140):
            buffer = io.BytesIO()
            Image.fromarray(np.full((16, 20, 3), value, dtype=np.uint8)).save(
                buffer, format="JPEG"
            )
            stream.write(buffer.getvalue())
            offsets.append(stream.tell())
    np.save(stem.with_suffix(".idx.npy"), np.asarray(offsets, dtype=np.int64))
    meta = SimpleNamespace(
        fps=30,
        episodes={
            0: {
                "dataset_from_index": 0,
                "dataset_to_index": 3,
                f"videos/{key}/from_timestamp": 0.0,
            }
        },
        features={key: {"shape": [16, 20, 3]}},
        get_video_file_path=lambda episode, camera: (
            Path("videos") / camera / "chunk-000/file-000.mp4"
        ),
    )
    return key, meta, stem


def test_packed_jpeg_queries_keep_requested_causal_order_and_duplicates(tmp_path):
    key, meta, _ = _cache(tmp_path)
    reader = PackedImageReader(tmp_path)
    images = reader.query(meta, {key: [0, 0, 1 / 30, 2 / 30]}, 0)[key]
    assert tuple(images.shape) == (4, 3, 16, 20)
    assert [int(frame[0, 0, 0]) for frame in images] == [20, 20, 80, 140]
    restored = pickle.loads(pickle.dumps(reader))
    assert int(restored.query(meta, {key: [2 / 30]}, 0)[key][0, 0, 0]) == 140
    reader.close()
    restored.close()


def test_missing_future_or_synthetic_packed_frames_are_rejected(tmp_path):
    key, meta, stem = _cache(tmp_path)
    reader = PackedImageReader(tmp_path)
    with pytest.raises(ValueError, match="outside its episode"):
        reader.query(meta, {key: [3 / 30]}, 0)
    reader.close()
    np.save(stem.with_suffix(".idx.npy"), np.asarray([0, 50], dtype=np.int64))
    with pytest.raises(ValueError, match="Incomplete packed"):
        PackedImageReader(tmp_path).query(meta, {key: [0]}, 0)
    manifest = json.loads((tmp_path / "packed_manifest.json").read_text())
    manifest["padded_frames"] = 1
    (tmp_path / "packed_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="synthetic"):
        PackedImageReader(tmp_path)
