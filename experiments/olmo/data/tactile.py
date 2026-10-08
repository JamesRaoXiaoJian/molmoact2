"""DM tactile sensor names and ordering shared by LeRobot training and serving.

The layout follows openpi-arx/openpi/src/openpi/shared/tactile.py. Images use
the existing MolmoAct2 vision encoder (as_image); no Pi0.5 state transforms apply.
"""

from collections.abc import Mapping, Sequence

TACTILE_KEYS = tuple(
    f"{arm}_gripper_{finger}_tactile"
    for arm in ("left", "right")
    for finger in ("left", "right")
)
TACTILE_IMAGE_KEYS = tuple(f"observation.images.{key}" for key in TACTILE_KEYS)
X5_RGB_IMAGE_KEYS = tuple(
    f"observation.images.{key}" for key in ("left_d405", "right_d405", "head_camera")
)


def tactile_keys_for_layout(layout: str = "two") -> list[str]:
    if layout == "two":
        return [TACTILE_IMAGE_KEYS[0], TACTILE_IMAGE_KEYS[3]]
    if layout == "four":
        return list(TACTILE_IMAGE_KEYS)
    raise ValueError(f"Unknown tactile layout {layout!r}; expected 'two' or 'four'.")


def normalize_tactile_keys(keys: Sequence[str] | None) -> list[str]:
    if keys is None:
        return []
    if isinstance(keys, str):
        raise TypeError("tactile_keys must be a sequence of DM sensor names.")
    normalized = []
    for key in keys:
        if not isinstance(key, str):
            raise TypeError("tactile_keys must contain strings.")
        full_key = (
            key
            if key.startswith("observation.images.")
            else f"observation.images.{key}"
        )
        if full_key not in TACTILE_IMAGE_KEYS:
            raise ValueError(f"Unknown DM tactile sensor {key!r}.")
        if full_key in normalized:
            raise ValueError(f"Duplicate tactile sensor {key!r}.")
        normalized.append(full_key)
    # Preserve explicitly configured order; the built-in layouts use [0,3]/[0,1,2,3].
    return normalized


def tactile_keys_from_metadata(metadata: Mapping) -> list[str]:
    keys = normalize_tactile_keys(metadata.get("tactile_keys"))
    if keys and metadata.get("tactile_backend", "as_image") not in (
        "as_image",
        "anytouch2",
        "sparsh_vjepa",
        "tactile_mae",
    ):
        raise ValueError("Unknown tactile_backend in checkpoint metadata.")
    return keys


def observation_image_keys(metadata: Mapping) -> list[str]:
    """RGB slots followed by tactile slots, in checkpoint order."""
    rgb_keys = list(metadata.get("camera_keys") or [])
    tactile_keys = tactile_keys_from_metadata(metadata)
    if tactile_keys:
        if not rgb_keys:
            raise ValueError("Tactile metadata requires explicit RGB camera_keys.")
        if any(key in TACTILE_IMAGE_KEYS for key in rgb_keys):
            raise ValueError(
                "Keep DM sensors in tactile_keys, separate from RGB camera_keys."
            )
        if len(set(rgb_keys)) != len(rgb_keys):
            raise ValueError("RGB camera_keys must be unique.")
    return [*rgb_keys, *tactile_keys]


def ensure_tactile_image_capacity(model_config, metadata_by_tag: Mapping) -> None:
    """Prevent the base model's five-image cap from dropping sensors/history."""
    counts = [
        (
            len(metadata["camera_keys"])
            if metadata.get("tactile_backend", "as_image") != "as_image"
            else len(observation_image_keys(metadata))
        )
        for metadata in metadata_by_tag.values()
        if tactile_keys_from_metadata(metadata)
    ]
    if not counts:
        return
    image_config = model_config.mm_preprocessor.image
    if image_config is None:
        raise ValueError("Tactile observations require the multi-image preprocessor.")
    required = max(counts) * int(model_config.n_obs_steps)
    image_config.max_images = max(int(image_config.max_images or 0), required)
