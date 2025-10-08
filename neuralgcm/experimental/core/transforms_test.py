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
from neuralgcm.experimental.core import interpolators
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_transforms
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

  def test_sel(self):
    x = cx.LabeledAxis('x', np.array([0.1, 0.5, 1.0, 1.5]))
    inputs = {'field_a': cx.wrap(np.arange(4, dtype=np.float32) * 10, x)}

    with self.subTest('exact_match_no_method'):
      sel_transform = transforms.Sel(sel_arg={'x': 0.5})
      actual = sel_transform(inputs)
      expected = {'field_a': cx.wrap(10.0)}
      chex.assert_trees_all_close(actual, expected)

    with self.subTest('nearest_match'):
      sel_transform = transforms.Sel(sel_arg={'x': 0.55}, method='nearest')
      actual = sel_transform(inputs)
      expected = {'field_a': cx.wrap(10.0)}
      chex.assert_trees_all_close(actual, expected)

    with self.subTest('no_match_no_method_raises'):
      sel_transform = transforms.Sel(sel_arg={'x': 0.55})
      with self.assertRaisesRegex(ValueError, 'No match found'):
        sel_transform(inputs)

    with self.subTest('nearest_with_array_selection_raises'):
      sel_transform = transforms.Sel(
          sel_arg={'x': np.array([0.1, 0.5])}, method='nearest'
      )
      with self.assertRaisesRegex(
          AssertionError, 'selection must be a single value'
      ):
        sel_transform(inputs)

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

  def test_clip_wavenumbers(self):
    """Tests that ClipWavenumbers works as expected."""
    ylm_grid_21 = coordinates.SphericalHarmonicGrid.T21()
    ylm_grid_31 = coordinates.SphericalHarmonicGrid.TL31()
    ylm_dims = 'longitude_wavenumber', 'total_wavenumber'
    inputs = {
        'u': cx.wrap(np.ones(ylm_grid_21.shape), ylm_grid_21),
        'v': cx.wrap(np.ones(ylm_grid_31.shape), ylm_grid_31),
        'skipped': cx.wrap(np.ones(ylm_grid_21.shape), *ylm_dims),
    }
    ls_21 = ylm_grid_21.fields['total_wavenumber']
    ls_31 = ylm_grid_31.fields['total_wavenumber']
    make_mask = lambda x, n: (np.arange(x.size) <= (x.max() - n)).astype(int)
    make_mask = cx.cmap(make_mask)
    clip_mask_21 = make_mask(ls_21.untag(*ls_21.axes), 3).tag(*ls_21.axes)
    clip_mask_31 = make_mask(ls_31.untag(*ls_31.axes), 5).tag(*ls_31.axes)
    expected = {
        'u': inputs['u'] * clip_mask_21,
        'v': inputs['v'] * clip_mask_31,
        'skipped': inputs['skipped'],
    }
    clip_transform = transforms.ClipWavenumbers(
        wavenumbers_for_grid={ylm_grid_21: 3, ylm_grid_31: 5},
        skip_missing=True,
    )
    actual = clip_transform(inputs)
    chex.assert_trees_all_equal(actual, expected)

    with self.subTest('raises_error_if_no_match'):
      clip_transform = transforms.ClipWavenumbers(
          wavenumbers_for_grid={ylm_grid_21: 3, ylm_grid_31: 5},
          skip_missing=False,
      )
      with self.assertRaisesRegex(ValueError, 'No matching grid found'):
        clip_transform(inputs)

  def test_to_modal(self):
    mesh = parallelism.Mesh()
    nodal_grid = coordinates.LonLatGrid.T21()
    inputs = {
        'u': cx.wrap(np.ones(nodal_grid.shape), nodal_grid),
    }
    with self.subTest('fixed_ylm_mapping_cubic'):
      ylm_grid = coordinates.SphericalHarmonicGrid.T21()
      ylm_transform = spherical_transforms.FixedYlmMapping(
          nodal_grid, ylm_grid, mesh, None
      )
      to_modal = transforms.ToModal(ylm_transform)
      actual_out_grid = cx.get_coordinate(to_modal(inputs)['u'])
      expected_ylm_grid = ylm_grid
      self.assertEqual(actual_out_grid, expected_ylm_grid)

    with self.subTest('fixed_ylm_mapping_linear'):
      ylm_grid = coordinates.SphericalHarmonicGrid.TL31()
      ylm_transform = spherical_transforms.FixedYlmMapping(
          nodal_grid, ylm_grid, mesh, None
      )
      to_modal = transforms.ToModal(ylm_transform)
      actual_out_grid = cx.get_coordinate(to_modal(inputs)['u'])
      expected_ylm_grid = ylm_grid
      self.assertEqual(actual_out_grid, expected_ylm_grid)

    with self.subTest('ylm_mapper_cubic'):
      ylm_mapper = spherical_transforms.YlmMapper(
          truncation_rule='cubic', mesh=mesh, partition_schema_key=None
      )
      to_modal = transforms.ToModal(ylm_mapper)
      actual_out_grid = cx.get_coordinate(to_modal(inputs)['u'])
      expected_ylm_grid = coordinates.SphericalHarmonicGrid.T21()
      self.assertEqual(actual_out_grid, expected_ylm_grid)

    with self.subTest('ylm_mapper_linear'):
      ylm_mapper = spherical_transforms.YlmMapper(
          truncation_rule='linear', mesh=mesh, partition_schema_key=None
      )
      to_modal = transforms.ToModal(ylm_mapper)
      actual_out_grid = cx.get_coordinate(to_modal(inputs)['u'])
      expected_ylm_grid = coordinates.SphericalHarmonicGrid.TL31()
      self.assertEqual(actual_out_grid, expected_ylm_grid)

  def test_to_nodal(self):
    mesh = parallelism.Mesh()
    cubic_ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    linear_ylm_grid = coordinates.SphericalHarmonicGrid.TL31()
    cubic_inputs = {
        'u': cx.wrap(np.ones(cubic_ylm_grid.shape), cubic_ylm_grid),
    }
    linear_inputs = {
        'u': cx.wrap(np.ones(linear_ylm_grid.shape), linear_ylm_grid),
    }
    grid = coordinates.LonLatGrid.T21()
    with self.subTest('fixed_ylm_mapping'):
      ylm_transform = spherical_transforms.FixedYlmMapping(
          grid, cubic_ylm_grid, mesh, None
      )
      to_nodal = transforms.ToNodal(ylm_transform)
      actual_out_grid = cx.get_coordinate(to_nodal(cubic_inputs)['u'])
      self.assertEqual(actual_out_grid, grid)
      # transforming from linear inputs.
      ylm_transform = spherical_transforms.FixedYlmMapping(
          grid, linear_ylm_grid, mesh, None
      )
      to_nodal = transforms.ToNodal(ylm_transform)
      actual_out_grid = cx.get_coordinate(to_nodal(linear_inputs)['u'])
      self.assertEqual(actual_out_grid, grid)

    with self.subTest('ylm_mapper'):
      ylm_mapper = spherical_transforms.YlmMapper(
          truncation_rule='cubic', mesh=mesh, partition_schema_key=None
      )
      to_nodal = transforms.ToNodal(ylm_mapper)
      actual_out_grid = cx.get_coordinate(to_nodal(cubic_inputs)['u'])
      self.assertEqual(actual_out_grid, grid)
      # transforming from linear inputs.
      ylm_mapper = spherical_transforms.YlmMapper(
          truncation_rule='linear', mesh=mesh, partition_schema_key=None
      )
      to_nodal = transforms.ToNodal(ylm_mapper)
      actual_out_grid = cx.get_coordinate(to_nodal(linear_inputs)['u'])
      self.assertEqual(actual_out_grid, grid)

  def test_apply_to_keys(self):
    x = cx.SizedAxis('x', 3)
    inputs = {
        'a': cx.wrap(np.array([1.0, 2.0, 3.0]), x),
        'b': cx.wrap(np.array([4.0, 5.0, 6.0]), x),
    }
    shift_and_norm = transforms.ShiftAndNormalize(
        shift=cx.wrap(1.0), scale=cx.wrap(2.0)
    )
    apply_to_a = transforms.ApplyToKeys(transform=shift_and_norm, keys=['a'])
    actual = apply_to_a(inputs)
    expected = {
        'a': cx.wrap(np.array([0.0, 0.5, 1.0]), x),
        'b': inputs['b'],
    }
    chex.assert_trees_all_close(actual, expected)

  def test_apply_over_axis_with_scan(self):
    nodal_grid = coordinates.LonLatGrid.T21()
    modal_grid = coordinates.SphericalHarmonicGrid.T21()
    mesh = parallelism.Mesh()
    ylm_transform = spherical_transforms.FixedYlmMapping(
        nodal_grid, modal_grid, mesh, None
    )
    to_modal = transforms.ToModal(ylm_transform)
    time = cx.SizedAxis('time', 5)
    inputs = {
        'u': cx.wrap(np.ones(time.shape + nodal_grid.shape), time, nodal_grid),
    }
    expected = to_modal(inputs)
    scan_transform = transforms.ApplyOverAxisWithScan(
        transform=to_modal, axis='time'
    )
    actual = scan_transform(inputs)
    chex.assert_trees_all_close(actual, expected, atol=1e-6)

  def test_regrid_conservative(self):
    source_grid = coordinates.LonLatGrid.TL63()
    target_grid = coordinates.LonLatGrid.T21()
    inputs = {'u': cx.wrap(np.ones(source_grid.shape), source_grid)}
    transform = transforms.Regrid(
        regridder=interpolators.ConservativeRegridder(target_grid)
    )
    expected = {'u': cx.wrap(np.ones(target_grid.shape), target_grid)}
    actual = transform(inputs)
    chex.assert_trees_all_close(actual, expected)

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

  def test_scale_to_match_coarse_fields(self):
    hres_grid = coordinates.LonLatGrid.TL63()
    coarse_grid = coordinates.LonLatGrid.T21()
    keys = ['precip', 'temp']
    hres_ones = cx.wrap(np.ones(hres_grid.shape, dtype=np.float32), hres_grid)
    coarse_twos = cx.wrap(
        np.ones(coarse_grid.shape, dtype=np.float32) * 2.0, coarse_grid
    )
    inputs = {
        'hres_precip': hres_ones,
        'hres_temp': hres_ones,
        'coarse_precip': coarse_twos,
        'coarse_temp': coarse_twos,
    }
    raw_hres_transform = transforms.Sequential([
        transforms.Select('hres_.*'),
        transforms.Rename({'hres_precip': 'precip', 'hres_temp': 'temp'}),
    ])
    ref_coarse_transform = transforms.Sequential([
        transforms.Select('coarse_.*'),
        transforms.Rename({'coarse_precip': 'precip', 'coarse_temp': 'temp'}),
    ])

    with self.subTest('conservation_applied'):
      transform = transforms.ScaleToMatchCoarseFields(
          raw_hres_transform=raw_hres_transform,
          ref_coarse_transform=ref_coarse_transform,
          coarse_grid=coarse_grid,
          hres_grid=hres_grid,
          keys=keys,
          epsilon=1e-9,
      )
      outputs = transform(inputs)
      regrid_to_coarse = transforms.Regrid(
          regridder=interpolators.ConservativeRegridder(coarse_grid)
      )
      downscaled_conserved_hres = regrid_to_coarse(outputs)
      expected_coarse = {key: coarse_twos for key in keys}
      chex.assert_trees_all_close(
          downscaled_conserved_hres, expected_coarse, atol=1e-5
      )

    with self.subTest('missing_key_raises'):
      inputs_missing = {
          'hres_precip': hres_ones,
          'hres_temp': hres_ones,
          'coarse_precip': coarse_twos,
          # 'coarse_temp' is missing
      }
      transform = transforms.ScaleToMatchCoarseFields(
          raw_hres_transform=raw_hres_transform,
          ref_coarse_transform=ref_coarse_transform,
          coarse_grid=coarse_grid,
          hres_grid=hres_grid,
          keys=['precip', 'temp'],
          epsilon=1e-9,
      )
      with self.assertRaisesRegex(ValueError, 'Key temp not found'):
        transform(inputs_missing)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
