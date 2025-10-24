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
from flax import nnx
import jax
from jax import config  # pylint: disable=g-importing-member
import jax.numpy as jnp
from neuralgcm.experimental.core import normalizations
import numpy as np


@functools.partial(nnx.jit, static_argnames=('update_stats',))
def _stream_norm_apply(
    norm: normalizations.StreamNorm,
    inputs: np.ndarray,
    update_stats: bool = True,
):
  """Applies stream norm and updates state."""
  return norm(inputs, update_stats=update_stats)


class NormalizationsTest(parameterized.TestCase):

  def test_stream_norm_close_to_identity_at_init(self):
    # we get exact identity if epsilon is zero.
    norm = normalizations.StreamNorm((2,), feature_axes=(-2,), epsilon=0.0)
    inputs = np.random.RandomState(0).normal(size=(10, 2, 2))
    outputs = _stream_norm_apply(norm, inputs, update_stats=False)
    np.testing.assert_allclose(outputs, inputs)

  def test_stream_norm_first_step_estimate(self):
    norm = normalizations.StreamNorm(epsilon=0.0)
    inputs = np.random.RandomState(0).normal(size=(10, 2, 2))
    _ = _stream_norm_apply(norm, inputs)
    mean, var = norm.stats()
    np.testing.assert_allclose(mean, np.mean(inputs), rtol=1e-6)
    np.testing.assert_allclose(var, np.var(inputs, ddof=1), rtol=1e-6)

  def test_stream_norm_normalizes_fixed_inputs(self):
    norm = normalizations.StreamNorm(epsilon=0.0)
    inputs = np.random.RandomState(0).normal(size=(10, 2, 2))
    output = _stream_norm_apply(norm, inputs)
    np.testing.assert_allclose(np.mean(output), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.var(output, ddof=1), 1.0, atol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='μ=0.5__σ=2.5', mean=0.5, var=2.5),
      dict(testcase_name='μ=1e-9__σ=4e-13', mean=1e-9, var=4e-13),  # clouds.
      dict(testcase_name='μ=273__σ=40', mean=273, var=40),  # temperature.
      dict(testcase_name='μ=1e6__σ=1e5', mean=1e6, var=1e5),  # geopotential.
  )
  def test_stream_norm_estimates_stats_correctly(self, mean, var):
    n_samples = 100
    n_levels = 4
    shape = (n_samples, n_levels, 32, 16)
    rng = np.random.RandomState(0)
    stream_norm = normalizations.StreamNorm()
    all_inputs = mean + np.sqrt(var) * rng.normal(size=shape)
    for i in range(n_samples):
      inputs = all_inputs[i]
      _ = _stream_norm_apply(stream_norm, inputs)

    expected_mean = all_inputs.mean()
    expected_variance = all_inputs.var(ddof=1)
    actual_mean, actual_variance = stream_norm.stats()
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-4)
    np.testing.assert_allclose(actual_variance, expected_variance, rtol=1e-3)

  def test_stream_norm_estimates_stats_correctly_per_level(self):
    n_samples = 100
    n_levels = 4
    spatial_shape = (32, 16)
    rng = jax.random.PRNGKey(0)
    stream_norm_per_level = normalizations.StreamNorm(
        feature_shape=(n_levels,),
        feature_axes=(-3,),
        epsilon=0.0,
    )
    means = np.array([0.5, 1.5, 2.5, 3.5])
    stds = np.array([2.5, 4.5, 6.5, 8.5])
    sample_level = lambda rng, i: jax.random.normal(
        jax.random.fold_in(rng, i), shape=(n_samples, *spatial_shape)
    )
    all_inputs = jnp.stack(
        [means[i] + stds[i] * sample_level(rng, i) for i in range(n_levels)],
        axis=1,
    )
    for i in range(n_samples):
      inputs = all_inputs[i]
      _ = _stream_norm_apply(stream_norm_per_level, inputs)

    expected_mean = all_inputs.mean(axis=(0, 2, 3))
    expected_variance = all_inputs.var(axis=(0, 2, 3), ddof=1)
    actual_mean, actual_variance = stream_norm_per_level.stats()
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-4)
    np.testing.assert_allclose(actual_variance, expected_variance, rtol=5e-3)

    with self.subTest('output_is_normalized'):
      expected_mean = all_inputs.mean(axis=(0, 2, 3), keepdims=True)[0, ...]
      expected_variance = all_inputs.var(axis=(0, 2, 3), keepdims=True)[0, ...]
      inputs = all_inputs[0, ...]  # pick one of the samples.
      actual_output = _stream_norm_apply(
          stream_norm_per_level, inputs, update_stats=False
      )
      expected_output = (inputs - expected_mean) / np.sqrt(expected_variance)
      np.testing.assert_allclose(actual_output, expected_output, atol=1e-4)


if __name__ == '__main__':
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
