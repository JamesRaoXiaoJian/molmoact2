# Tactile backbone provenance

The encoder structures and preprocessing contracts are adapted from the local
`/root/openpi-arx/openpi` tactile integration and its reference
`/root/lerobot-v0.5.1/src/lerobot/policies/pi05_tactile/` implementation.

- AnyTouch2: GeWu-Lab/AnyTouch2, commit
  `82c5677d9cf0176d97a1fe04745f63cd02dd6f54`, MIT. See `AnyTouch2-MIT.txt`.
- Sparsh V-JEPA: facebookresearch/sparsh, commit
  `fee6a05a97330eca014b59c71ac41596fe3974a7`. Repository and pretrained weights
  declare CC-BY-NC-4.0. See `Sparsh-CC-BY-NC-4.0.txt`. Upstream ViT/layer
  files additionally carry Apache-2.0 notices.
- Tactile-MAE: AnyTouch stage-1 ViT-L/14 image encoder, with downstream query
  injection as implemented by the reference LeRobot tactile MAE extractor.
  Reconstruction heads are excluded from this implementation.

No datasets or pretrained weights are included in Git. The implementation uses
PyTorch operations and the installed Transformers CLIP encoder submodules.
