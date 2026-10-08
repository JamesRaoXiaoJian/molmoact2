"""Strict reads of the existing OpenPI packed JPEG cache, without video seeking."""

import io
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class PackedImageReader:
    def __init__(self, root, max_open_files=64):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "packed_manifest.json").read_text())
        if self.manifest.get("format") != "openpi-predecoded-packed-v1":
            raise ValueError("Unsupported packed image cache format.")
        if self.manifest.get("padded_frames") != 0:
            raise ValueError(
                "Packed cache must be validated without synthetic frame padding."
            )
        self.max_open_files = max_open_files
        self._open = OrderedDict()
        self._pid = os.getpid()

    def _shard(self, stem, expected_frames):
        if self._pid != os.getpid():
            # fork workers must not share inherited seek offsets.
            self.close()
            self._pid = os.getpid()
        key = str(stem)
        if key in self._open:
            value = self._open.pop(key)
            self._open[key] = value
            return value
        offsets = np.load(
            stem.with_suffix(".idx.npy"), mmap_mode="r", allow_pickle=False
        )
        data = stem.with_suffix(".bin")
        if (
            len(offsets) != expected_frames + 1
            or int(offsets[0]) != 0
            or int(offsets[-1]) != data.stat().st_size
        ):
            raise ValueError(f"Incomplete packed image shard: {stem}")
        stream = data.open("rb")
        value = (offsets, stream)
        self._open[key] = value
        while len(self._open) > self.max_open_files:
            _, (_, old_stream) = self._open.popitem(last=False)
            old_stream.close()
        return value

    def query(self, metadata, timestamps, episode_index):
        if metadata.fps != self.manifest["fps"]:
            raise ValueError("Packed cache and dataset FPS differ.")
        episode = metadata.episodes[int(episode_index)]
        expected_frames = int(episode["dataset_to_index"]) - int(
            episode["dataset_from_index"]
        )
        result = {}
        for key, query_times in timestamps.items():
            if key not in self.manifest["video_keys"]:
                raise ValueError(
                    f"Required camera is absent from the packed cache: {key}"
                )
            if float(episode[f"videos/{key}/from_timestamp"]) != 0:
                raise ValueError(
                    "This packed cache requires one episode per video file."
                )
            video_path = metadata.get_video_file_path(int(episode_index), key)
            stem = (
                self.root
                / "packed"
                / Path(video_path).relative_to("videos").with_suffix("")
            )
            offsets, stream = self._shard(stem, expected_frames)
            frames = []
            for timestamp in query_times:
                position = float(timestamp) * metadata.fps
                frame = round(position)
                if abs(position - frame) > 0.04 or not 0 <= frame < expected_frames:
                    raise ValueError(
                        f"Packed frame outside its episode: {key} time={timestamp}"
                    )
                start, end = int(offsets[frame]), int(offsets[frame + 1])
                if end <= start:
                    raise ValueError(
                        f"Invalid packed frame offsets: {stem} frame={frame}"
                    )
                stream.seek(start)
                jpeg = stream.read(end - start)
                if len(jpeg) != end - start:
                    raise ValueError(f"Truncated packed JPEG: {stem} frame={frame}")
                with Image.open(io.BytesIO(jpeg)) as image:
                    rgb = np.array(image.convert("RGB"), copy=True)
                spec = metadata.features[key]
                shape = tuple(spec["shape"])
                if rgb.shape != shape:
                    raise ValueError(
                        f"Packed image dimensions differ from metadata: {key} {rgb.shape} != {shape}"
                    )
                frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
            result[key] = torch.stack(frames).squeeze(0)
        return result

    def close(self):
        for _, stream in self._open.values():
            stream.close()
        self._open.clear()

    def __getstate__(self):
        return {**self.__dict__, "_open": OrderedDict()}
