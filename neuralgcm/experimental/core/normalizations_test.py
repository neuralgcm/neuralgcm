# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests that normalization modules work as expected."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax
from jax import config  # pylint: disable=g-importing-member
from neuralgcm.experimental.core import normalizations
import numpy as np


@functools.partial(nnx.jit, static_argnames=('update_stats',))
def _stream_norm_apply(
    norm: normalizations.StreamNorm,
    inputs: dict[str, cx.Field],
    update_stats: bool = True,
    mask: cx.Field | None = None,
):
  """Applies streaming normalizer and updates state."""
  return norm(inputs, update_stats=update_stats, mask=mask)


class StreamNormTest(parameterized.TestCase):

  def test_stream_norm_close_to_identity_at_init(self):
    # we get exact identity if epsilon is zero.
    s = cx.SizedAxis('s', 2)
    norm = normalizations.StreamNorm({'x': s}, epsilon=0.0)
    rng = np.random.RandomState(0)
    inputs = {'x': cx.field(rng.normal(size=(10, 2, s.size)), 'a', 'b', s)}

    outputs = _stream_norm_apply(norm, inputs, update_stats=False)
    np.testing.assert_allclose(outputs['x'].data, inputs['x'].data)

  def test_streaming_normalizer_first_step_estimate(self):
    x = cx.SizedAxis('x', 4)
    inputs_data = np.random.RandomState(0).normal(size=(10, x.size))
    inputs = {'u': cx.field(inputs_data, 'b', x)}

    normalizer = normalizations.StreamNorm({'u': x}, epsilon=0.0)
    _ = _stream_norm_apply(normalizer, inputs)
    means, vars_ = normalizer.stats(ddof=1)

    expected_mean = np.mean(inputs_data, axis=0)
    expected_var = np.var(inputs_data, axis=0, ddof=1)
    np.testing.assert_allclose(means['u'].data, expected_mean, rtol=1e-6)
    np.testing.assert_allclose(vars_['u'].data, expected_var, rtol=1e-6)

  def test_streaming_normalizer_normalizes_fixed_inputs(self):
    x = cx.SizedAxis('x', 3)
    inputs_data = np.random.RandomState(0).normal(size=(100, x.size))
    inputs = {'u': cx.field(inputs_data, 'b', x)}

    normalizer = normalizations.StreamNorm({'u': x}, epsilon=0.0)
    outputs = _stream_norm_apply(normalizer, inputs)

    out_data = outputs['u'].data
    np.testing.assert_allclose(np.mean(out_data, axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.var(out_data, axis=0, ddof=1), 1.0, atol=1e-5)

  def test_streaming_normalizer_estimates_stats_scalar(self):
    x = cx.SizedAxis('x', 5)
    rng = np.random.RandomState(42)
    u_data = rng.normal(loc=10.0, scale=2.0, size=(10, 5))
    v_data = rng.normal(loc=-5.0, scale=3.0, size=(10,))
    inputs = {
        'u': cx.field(u_data, 'batch', x),
        'v': cx.field(v_data, 'b'),
    }

    coords = {'u': cx.Scalar(), 'v': cx.Scalar()}  # scalar stats.
    normalizer = normalizations.StreamNorm(coords, epsilon=0.0)

    _ = _stream_norm_apply(normalizer, inputs)
    means, vars_ = normalizer.stats(ddof=1)

    with self.subTest('u_stats'):
      expected_mean_u = np.mean(u_data)
      expected_var_u = np.var(u_data, ddof=1)
      np.testing.assert_allclose(means['u'].data, expected_mean_u, rtol=1e-5)
      np.testing.assert_allclose(vars_['u'].data, expected_var_u, rtol=1e-5)

    with self.subTest('v_stats'):
      expected_mean_v = np.mean(v_data)
      expected_var_v = np.var(v_data, ddof=1)
      np.testing.assert_allclose(means['v'].data, expected_mean_v, rtol=1e-5)
      np.testing.assert_allclose(vars_['v'].data, expected_var_v, rtol=1e-5)

  def test_streaming_normalizer_estimates_stats_per_level(self):
    n_samples = 100
    n_levels = 4
    x, y = cx.SizedAxis('x', 32), cx.SizedAxis('y', 16)
    xy = cx.coords.compose(x, y)
    level = cx.SizedAxis('level', n_levels)
    means = np.array([0.5, 1.5, 2.5, 3.5])  # per level means.
    stds = np.array([2.5, 4.5, 6.5, 8.5])  # per level stds.

    rng = np.random.RandomState(0)
    data = np.zeros((n_samples, n_levels) + xy.shape)
    for i in range(n_levels):
      data[:, i, :, :] = means[i] + stds[i] * rng.normal(
          size=(n_samples,) + xy.shape
      )
    inputs = {'u': cx.field(data, 'b', level, xy)}
    coords = {'u': level}

    normalizer = normalizations.StreamNorm(coords, epsilon=0.0)
    _ = _stream_norm_apply(normalizer, inputs)
    means_out, vars_out = normalizer.stats(ddof=1)

    expected_mean = np.mean(data, axis=(0, 2, 3))
    expected_var = np.var(data, axis=(0, 2, 3), ddof=1)
    np.testing.assert_allclose(means_out['u'].data, expected_mean, rtol=1e-4)
    np.testing.assert_allclose(vars_out['u'].data, expected_var, rtol=5e-3)

  def test_streaming_normalizer_estimates_stats_multiscale(self):
    means = {'u': 0.5, 'clouds': 1e-9, 't': 273.0, 'z': 1e6}
    variances = {'u': 2.5, 'clouds': 4e-13, 't': 40.0, 'z': 1e5}
    n_samples = 100
    x, y = cx.SizedAxis('x', 32), cx.SizedAxis('y', 16)
    rng = np.random.RandomState(0)
    all_data = {}
    for k in means:
      shape = (n_samples, x.size, y.size)
      all_data[k] = means[k] + np.sqrt(variances[k]) * rng.normal(size=shape)

    coords = {k: cx.Scalar() for k in means}
    normalizer = normalizations.StreamNorm(coords, epsilon=0.0)

    batch_size = 10
    for i in range(0, n_samples, batch_size):
      inputs = {}
      for k in means:
        inputs[k] = cx.field(all_data[k][i : i + batch_size], 'batch', x, y)
      _ = _stream_norm_apply(normalizer, inputs)

    actual_means, actual_vars = normalizer.stats(ddof=1)
    expected_means = {k: np.mean(all_data[k]) for k in means}
    expected_vars = {k: np.var(all_data[k], ddof=1) for k in means}
    with self.subTest('u_scales'):
      np.testing.assert_allclose(
          actual_means['u'].data, expected_means['u'], rtol=1e-4
      )
      np.testing.assert_allclose(
          actual_vars['u'].data, expected_vars['u'], rtol=1e-3
      )
    with self.subTest('clouds_scales'):
      np.testing.assert_allclose(
          actual_means['clouds'].data, expected_means['clouds'], rtol=1e-4
      )
      np.testing.assert_allclose(
          actual_vars['clouds'].data, expected_vars['clouds'], rtol=1e-3
      )
    with self.subTest('t_scales'):
      np.testing.assert_allclose(
          actual_means['t'].data, expected_means['t'], rtol=1e-4
      )
      np.testing.assert_allclose(
          actual_vars['t'].data, expected_vars['t'], rtol=1e-3
      )
    with self.subTest('z_scales'):
      np.testing.assert_allclose(
          actual_means['z'].data, expected_means['z'], rtol=5e-4
      )
      np.testing.assert_allclose(
          actual_vars['z'].data, expected_vars['z'], rtol=5e-2
      )

  def test_streaming_normalizer_with_mask(self):
    x, y = cx.SizedAxis('x', 4), cx.SizedAxis('y', 4)
    time_dim = cx.SizedAxis('time', 10)

    rng = np.random.RandomState(0)
    inputs_u = rng.normal(size=(time_dim.size, x.size, y.size))
    inputs_field = cx.field(inputs_u, time_dim, x, y)
    inputs = {'u': inputs_field}
    coords = {'u': cx.coords.compose(x, y)}

    # Generate random mask
    rng_jax = jax.random.key(0)
    mask_shape = (time_dim.size, x.size, y.size)
    mask_arr = jax.random.bernoulli(rng_jax, p=0.8, shape=mask_shape)
    mask = cx.field(mask_arr, time_dim, x, y)

    normalizer = normalizations.StreamNorm(coords)
    _ = _stream_norm_apply(normalizer, inputs, mask=mask)
    means, vars_ = normalizer.stats(ddof=1)

    # We assemble expected means and variances by hand, skipping invalid data.
    mask_np = np.array(mask_arr)  # False values are expected to be skipped.
    expected_mean = np.zeros((x.size, y.size))
    expected_var = np.zeros((x.size, y.size))
    for i in range(x.size):
      for j in range(y.size):
        # Select data where mask is True (valid).
        valid_data = inputs_u[mask_np[:, i, j], i, j]
        assert valid_data.size > 1
        expected_mean[i, j] = np.mean(valid_data)
        expected_var[i, j] = np.var(valid_data, ddof=1)

    np.testing.assert_allclose(means['u'].data, expected_mean, atol=1e-5)
    np.testing.assert_allclose(vars_['u'].data, expected_var, atol=1e-5)

  def test_streaming_normalizer_skip_unspecified(self):
    normalizer = normalizations.StreamNorm(
        {'u': cx.Scalar()}, skip_unspecified=True, epsilon=0.0
    )
    rng = np.random.RandomState(0)
    inputs = {
        'u': cx.field(0.1 * rng.normal(size=(100)), 'b'),
        'v': cx.field(2.0 * rng.normal(size=(100)), 'b'),
    }

    _ = _stream_norm_apply(normalizer, inputs)
    outputs = _stream_norm_apply(normalizer, inputs, False)
    np.testing.assert_allclose(np.var(outputs['u'].data, ddof=1), 1, atol=1e-3)
    np.testing.assert_allclose(outputs['v'].data, inputs['v'].data, atol=1e-3)

    # Without skip_unspecified=True, this should raise an error.
    normalizer_strict = normalizations.StreamNorm(
        {'u': cx.Scalar()}, skip_unspecified=False
    )
    with self.assertRaisesRegex(
        ValueError, 'Inputs contain keys not in coords'
    ):
      _stream_norm_apply(normalizer_strict, inputs)

  def test_streaming_normalizer_allow_missing(self):
    # if allow_missing=False, inputs must contain all keys from coords.
    normalizer = normalizations.StreamNorm(
        {'u': cx.Scalar(), 'v': cx.Scalar()}, allow_missing=False
    )
    rng = np.random.RandomState(0)
    inputs = {'u': cx.field(0.1 * rng.normal(size=(10,)), 'b')}
    with self.assertRaisesRegex(
        ValueError, 'Inputs are missing keys in coords'
    ):
      _stream_norm_apply(normalizer, inputs)

    # if allow_missing=True, inputs can miss keys from coords.
    normalizer_allowing_missing = normalizations.StreamNorm(
        {'u': cx.Scalar(), 'v': cx.Scalar()}, allow_missing=True
    )
    # should not raise error
    _stream_norm_apply(normalizer_allowing_missing, inputs)

  def test_streaming_normalizer_skip_nans(self):
    x = cx.SizedAxis('x', 5)
    rng = np.random.RandomState(42)
    inputs_data = rng.normal(size=(10, x.size))
    # introduce NaNs
    inputs_data[0, 0] = np.nan
    inputs_data[2, 3] = np.nan
    inputs_data[5, 1] = np.nan

    inputs = {'u': cx.field(inputs_data, 'b', x)}

    # Enable skip_nans
    normalizer = normalizations.StreamNorm(
        {'u': x}, epsilon=0.0, skip_nans=True
    )
    _ = _stream_norm_apply(normalizer, inputs)
    means, vars_ = normalizer.stats(ddof=1)

    # Calculate expected stats ignoring NaNs
    expected_mean = np.nanmean(inputs_data, axis=0)
    expected_var = np.nanvar(inputs_data, axis=0, ddof=1)

    np.testing.assert_allclose(means['u'].data, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(vars_['u'].data, expected_var, rtol=1e-5)


if __name__ == '__main__':
  config.update('jax_enable_x64', True)
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
