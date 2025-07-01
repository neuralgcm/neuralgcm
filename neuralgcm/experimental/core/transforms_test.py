# Copyright 2024 Google LLC
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

"""Tests that transforms produce outputs with expected structure."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import transforms
import numpy as np


class TransformsTest(parameterized.TestCase):
  """Tests that transforms work as expected."""

  def test_select(self):
    select_transform = transforms.Select(regex_patterns='field_a|field_c')
    x = cx.SizedAxis('x', 3)
    inputs = {
        'field_a': cx.wrap(np.array([1, 2, 3]), x),
        'field_b': cx.wrap(np.array([4, 5, 6]), x),
        'field_c': cx.wrap(np.array([7, 8, 9]), x),
    }
    actual = select_transform(inputs)
    expected = {
        'field_a': inputs['field_a'],
        'field_c': inputs['field_c'],
    }
    chex.assert_trees_all_close(actual, expected)

  def test_broadcast(self):
    broadcast_transform = transforms.Broadcast()
    x = cx.SizedAxis('x', 3)
    y = cx.SizedAxis('y', 2)
    inputs = {
        'field_a': cx.wrap(np.array([1, 2, 3]), x),
        'field_b': cx.wrap(np.ones((2, 3)), y, x),
    }
    actual = broadcast_transform(inputs)
    expected = {
        'field_a': cx.wrap(np.array([[1, 2, 3], [1, 2, 3]]), y, x),
        'field_b': cx.wrap(np.ones((2, 3)), y, x),
    }
    chex.assert_trees_all_close(actual, expected)

  def test_shift_and_normalize(self):
    b, x = cx.SizedAxis('batch', 20), cx.SizedAxis('x', 3)
    rng = jax.random.PRNGKey(0)
    data = 0.3 + 0.5 * jax.random.normal(rng, shape=(b.shape + x.shape))
    inputs = {'data': cx.wrap(data, b, x)}
    normalize = transforms.ShiftAndNormalize(
        shift=cx.wrap(np.mean(data)),
        scale=cx.wrap(np.std(data)),
    )
    out = normalize(inputs)
    np.testing.assert_allclose(np.mean(out['data'].data), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(out['data'].data), 1.0, atol=1e-6)
    inverse_normalize = transforms.ShiftAndNormalize(
        shift=cx.wrap(np.mean(data)),
        scale=cx.wrap(np.std(data)),
        reverse=True,
    )
    reconstructed = inverse_normalize(out)
    np.testing.assert_allclose(reconstructed['data'].data, data, atol=1e-6)

  def test_sequential(self):
    x = cx.SizedAxis('x', 3)

    class AddOne(transforms.TransformABC):

      def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
        return {k: v + 1 for k, v in inputs.items()}

    sequential = transforms.Sequential(
        transforms=[transforms.Select(regex_patterns=r'(?!time).*'), AddOne()]
    )
    inputs = {
        'a': cx.wrap(np.array([1, 2, 3]), x),
        'time': cx.wrap(np.pi),
    }
    actual = sequential(inputs)
    expected = {'a': cx.wrap(np.array([2, 3, 4]), x)}
    chex.assert_trees_all_close(actual, expected)

  def test_mask_by_threshold(self):
    x = cx.LabeledAxis('x', np.linspace(0, 1, 10))
    inputs = {
        'mask': x.fields['x'],  # will mask by values of x coordinates.
        'u': cx.wrap(np.ones(x.shape), x),
        'v': cx.wrap(np.arange(x.shape[0]), x),
    }
    with self.subTest('_below'):
      mask = transforms.Mask(
          mask_key='mask',
          compute_mask_method='below',
          apply_mask_method='multiply',
          threshold_value=0.6,
      )
      actual = mask(inputs)
      expected = {
          'u': inputs['u'] * (x.fields['x'] < 0.6).astype(np.float32),
          'v': inputs['v'] * (x.fields['x'] < 0.6).astype(np.float32),
      }
      chex.assert_trees_all_equal(actual, expected)

    with self.subTest('_above'):
      mask = transforms.Mask(
          mask_key='mask',
          compute_mask_method='above',
          apply_mask_method='multiply',
          threshold_value=0.3,
      )
      actual = mask(inputs)
      expected = {
          'u': inputs['u'] * (x.fields['x'] > 0.3).astype(np.float32),
          'v': inputs['v'] * (x.fields['x'] > 0.3).astype(np.float32),
      }
      chex.assert_trees_all_equal(actual, expected)

  def test_mask_explicit_scale(self):
    x = cx.LabeledAxis('x', np.linspace(0, 1, 10))
    inputs = {
        'mask': x.fields['x'] ** 2 < 0.64,
        'u': cx.wrap(np.ones(x.shape), x),
        'v': cx.wrap(np.arange(x.shape[0]), x),
    }
    mask = transforms.Mask(
        mask_key='mask',
        compute_mask_method='take',
        apply_mask_method='multiply',
    )
    actual = mask(inputs)
    expected = {
        'u': inputs['u'] * inputs['mask'],
        'v': inputs['v'] * inputs['mask'],
    }
    chex.assert_trees_all_equal(actual, expected)

  def test_mask_nan_to_0(self):
    x = cx.LabeledAxis('x', np.linspace(0, 1, 4))
    y = cx.SizedAxis('y', 10)
    mask = cx.wrap(np.array([0.1, np.nan, 10.4, np.nan]), x)
    one_hot_mask = cx.wrap(np.array([1.0, np.nan, 1.0, np.nan]), x)
    inputs = {
        'mask': mask,
        'nan_at_mask': one_hot_mask * cx.wrap(np.ones(x.shape + y.shape), x, y),
        'no_nans': cx.wrap(np.arange(x.shape[0]), x),  # no nan in v.
        'all_nans': cx.wrap(np.ones(x.shape) * np.nan, x),
    }
    mask = transforms.Mask(
        mask_key='mask',
        compute_mask_method='isnan',
        apply_mask_method='nan_to_0',
    )
    actual = mask(inputs)
    with self.subTest('nans_are_zeros_under_mask'):
      y_zeros = np.zeros(y.shape)
      np.testing.assert_allclose(actual['nan_at_mask'].data[1, :], y_zeros)
      np.testing.assert_allclose(actual['nan_at_mask'].data[3, :], y_zeros)
      np.testing.assert_allclose(actual['all_nans'].data[1], 0.0)
      np.testing.assert_allclose(actual['all_nans'].data[3], 0.0)
    with self.subTest('non_nans_unaffected'):
      np.testing.assert_allclose(actual['no_nans'].data, inputs['no_nans'].data)
    with self.subTest('same_values_outside_of_mask'):
      y_ones = np.ones(y.shape)
      np.testing.assert_allclose(actual['nan_at_mask'].data[0, :], y_ones)
      np.testing.assert_allclose(actual['all_nans'].data[0], np.nan)

  @parameterized.parameters(
      dict(n_clip=1),
      dict(n_clip=2),
      dict(n_clip=5),
  )
  def test_clip_wavenumbers(self, n_clip: int = 1):
    """Tests that ClipWavenumbers works as expected."""
    grid = coordinates.SphericalHarmonicGrid.T21()
    inputs = {
        'u': cx.wrap(np.ones(grid.shape), grid),
        'v': cx.wrap(np.ones(grid.shape), grid),
    }
    ls = grid.fields['total_wavenumber']
    make_mask = lambda x: (np.arange(x.size) <= (x.max() - n_clip)).astype(int)
    clip_mask = cx.cmap(make_mask)(ls.untag(*ls.axes)).tag(*ls.axes)
    expected = {k: v * clip_mask for k, v in inputs.items()}
    clip_transform = transforms.ClipWavenumbers(
        grid=grid,
        wavenumbers_to_clip=n_clip,
    )
    actual = clip_transform(inputs)
    chex.assert_trees_all_equal(actual, expected)

  def test_streaming_stats_normalization_scalar(self):
    b, x = cx.SizedAxis('batch', 20), cx.SizedAxis('x', 7)
    rng = jax.random.PRNGKey(0)
    inputs = {
        's': cx.wrap(jax.random.normal(rng, shape=(b.shape + x.shape)), b, x),
    }

    feature_shapes = {'s': tuple()}
    feature_axes = tuple()

    streaming_norm_scalar = transforms.StreamingStatsNormalization(
        feature_shapes=feature_shapes,
        feature_axes=feature_axes,
        update_stats=True,
        epsilon=0.0,  # Use zero epsilon to get a tight std check.
    )
    _ = streaming_norm_scalar(inputs)
    streaming_norm_scalar.update_stats = False
    out = streaming_norm_scalar(inputs)['s']
    self.assertEqual(cx.get_coordinate(out), cx.compose_coordinates(b, x))
    np.testing.assert_allclose(np.mean(out.data), 0.0, atol=1e-6)
    np.testing.assert_allclose(out.data.var(ddof=1), 1.0, atol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='scale_10', scale=10.0, atol_identity=1e-1),
      dict(testcase_name='scale_100', scale=100.0, atol_identity=1),
  )
  def test_tanh_clip(self, scale: float, atol_identity: float):
    x_size = 11
    x = cx.SizedAxis('x', x_size)
    inputs = {
        'in_range': cx.wrap(np.linspace(-scale * 0.3, scale * 0.3, x_size), x),
        'out_of_range': cx.wrap(np.linspace(-scale * 2, scale * 2, x_size), x),
    }
    transform_instance = transforms.TanhClip(scale=scale)
    clipped = transform_instance(inputs)

    with self.subTest('valid_range'):
      for k, v in clipped.items():
        np.testing.assert_array_less(v.data, scale, err_msg=f'failed_for_{k=}')
        np.testing.assert_array_less(-v.data, scale, err_msg=f'failed_for_{k=}')

    with self.subTest('near_identity_in_range'):
      np.testing.assert_allclose(
          clipped['in_range'].data, inputs['in_range'].data, atol=atol_identity
      )

  def test_streaming_stats_normalization_1d(self):
    b, x = cx.SizedAxis('batch', 20), cx.SizedAxis('x', 7)
    rng = jax.random.PRNGKey(0)
    inputs = {
        's': cx.wrap(jax.random.normal(rng, shape=(b.shape + x.shape)), b, x),
    }

    feature_shapes = {'s': x.shape}
    feature_axes = tuple([1])

    streaming_norm_scalar = transforms.StreamingStatsNormalization(
        feature_shapes=feature_shapes,
        feature_axes=feature_axes,
        update_stats=True,
        epsilon=0.0,  # Use zero epsilon to get a tight std check.
    )
    _ = streaming_norm_scalar(inputs)
    streaming_norm_scalar.update_stats = False
    out = streaming_norm_scalar(inputs)['s']
    self.assertEqual(cx.get_coordinate(out), cx.compose_coordinates(b, x))
    np.testing.assert_allclose(
        np.mean(out.data, axis=0), np.zeros(x.shape), atol=1e-6
    )
    np.testing.assert_allclose(
        out.data.var(ddof=1, axis=0), np.ones(x.shape), atol=1e-6
    )


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
