import numpy as np
import pytest
import torch

from olmo.data.robot_processing import _FeatureNormalizer


@pytest.mark.parametrize("mode", ["min_max", "q01_q99", "q10_q90"])
@pytest.mark.parametrize("tensor", [False, True])
def test_unnormalize_clips_only_normalized_dimensions(mode, tensor):
    stats = {"min": [10., -5.], "max": [20., 5.],
             "q01": [10., -5.], "q99": [20., 5.],
             "q10": [10., -5.], "q90": [20., 5.], "mask": [True, False]}
    normalizer = _FeatureNormalizer.from_stats(stats, mode)
    values = np.array([[2., -3.5], [-2., 2.5]], dtype=np.float32)
    original = values.copy()
    actual = normalizer.unnormalize(torch.from_numpy(values) if tensor else values)
    np.testing.assert_allclose(actual, [[20., -3.5], [10., 2.5]])
    np.testing.assert_array_equal(values, original)
    if tensor:
        assert isinstance(actual, torch.Tensor) and actual.dtype == torch.float32
