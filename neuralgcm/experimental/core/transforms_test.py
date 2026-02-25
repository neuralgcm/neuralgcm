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

from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import diagnostics
from neuralgcm.experimental.core import interpolators
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spatial_filters
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import transforms
import numpy as np


class TransformsTest(parameterized.TestCase):
  """Tests that transforms work as expected."""

  def test_select(self):
    select_transform = transforms.Select(regex_patterns='field_a|field_c')
    x = cx.SizedAxis('x', 3)
    inputs = {
        'field_a': cx.field(np.array([1, 2, 3]), x),
        'field_b': cx.field(np.array([4, 5, 6]), x),
        'field_c': cx.field(np.array([7, 8, 9]), x),
    }
    actual = select_transform(inputs)
    expected = {
        'field_a': inputs['field_a'],
        'field_c': inputs['field_c'],
    }
    chex.assert_trees_all_close(actual, expected)

  def test_sel(self):
    x = cx.LabeledAxis('x', np.array([0.1, 0.5, 1.0, 1.5]))
    inputs = {'field_a': cx.field(np.arange(4, dtype=np.float32) * 10, x)}

    with self.subTest('exact_match_no_method'):
      sel_transform = transforms.Sel(sel_arg={'x': 0.5})
      actual = sel_transform(inputs)
      expected = {'field_a': cx.field(10.0)}
      chex.assert_trees_all_close(actual, expected)

    with self.subTest('nearest_match'):
      sel_transform = transforms.Sel(sel_arg={'x': 0.55}, method='nearest')
      actual = sel_transform(inputs)
      expected = {'field_a': cx.field(10.0)}
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
        'field_a': cx.field(np.array([1, 2, 3]), x),
        'field_b': cx.field(np.ones((2, 3)), y, x),
    }
    actual = broadcast_transform(inputs)
    expected = {
        'field_a': cx.field(np.array([[1, 2, 3], [1, 2, 3]]), y, x),
        'field_b': cx.field(np.ones((2, 3)), y, x),
    }
    chex.assert_trees_all_close(actual, expected)

  def test_shift_and_normalize(self):
    b, x = cx.SizedAxis('batch', 20), cx.SizedAxis('x', 3)
    rng = jax.random.PRNGKey(0)
    data = 0.3 + 0.5 * jax.random.normal(rng, shape=(b.shape + x.shape))
    inputs = {'data': cx.field(data, b, x)}
    normalize = transforms.ShiftAndNormalize(
        shift=cx.field(np.mean(data)),
        scale=cx.field(np.std(data)),
    )
    out = normalize(inputs)
    np.testing.assert_allclose(np.mean(out['data'].data), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(out['data'].data), 1.0, atol=1e-6)
    inverse_normalize = transforms.ShiftAndNormalize(
        shift=cx.field(np.mean(data)),
        scale=cx.field(np.std(data)),
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
        'a': cx.field(np.array([1, 2, 3]), x),
        'time': cx.field(np.pi),
    }
    actual = sequential(inputs)
    expected = {'a': cx.field(np.array([2, 3, 4]), x)}
    chex.assert_trees_all_close(actual, expected)

  def test_mean_over_axes(self):
    x, y = cx.SizedAxis('x', 4), cx.SizedAxis('y', 5)
    xy = cx.coords.compose(x, y)
    # Mean of (0+1+2+3)/4 + (0+1+2+3+4)/5 = 1.5 + 2.0 = 3.5
    inputs = {'a': cx.field(np.arange(4)[:, None] + np.arange(5)[None, :], xy)}

    with self.subTest('standard_mean'):
      transform = transforms.MeanOverAxes(dims=('x', 'y'))
      actual = transform(inputs)
      expected = {'a': cx.field(3.5)}
      chex.assert_trees_all_close(actual, expected)

    grid = coordinates.LonLatGrid.T21()
    lats = np.deg2rad(grid.fields['latitude'].data)
    lons = np.deg2rad(grid.fields['longitude'].data)
    # sin(lat) integrates to 0 over the sphere.
    # sin(lon) integrates to 0 over longitude.
    # 3.0 is the constant offset.
    data = 3.0 + np.sin(lats)[None, :] + np.sin(lons)[:, None]
    inputs_sphere = {'u': cx.field(data, grid)}

    with self.subTest('spherical_mean_lat_lon'):
      transform = transforms.MeanOverAxes(dims=('latitude', 'longitude'))
      actual = transform(inputs_sphere)
      expected = {'u': cx.field(3.0)}
      chex.assert_trees_all_close(actual, expected, atol=1e-6)

    with self.subTest('spherical_mean_lon'):
      transform = transforms.MeanOverAxes(dims=('longitude',))
      actual = transform(inputs_sphere)
      # Mean over lon of sin(lon) is 0.
      # Mean over lon of 3.0 + sin(lat) is 3.0 + sin(lat).
      expected_data = 3.0 + np.sin(lats)
      expected = {'u': cx.field(expected_data, grid.axes[1])}
      chex.assert_trees_all_close(actual, expected, atol=1e-6)

    with self.subTest('spherical_mean_lat'):
      transform = transforms.MeanOverAxes(dims=('latitude',))
      actual = transform(inputs_sphere)
      # Mean over lat of sin(lat) is 0.
      # Mean over lat of 3.0 + sin(lon) is 3.0 + sin(lon).
      expected_data = 3.0 + np.sin(lons)
      expected = {'u': cx.field(expected_data, grid.axes[0])}
      chex.assert_trees_all_close(actual, expected, atol=1e-6)

    with self.subTest('spherical_mean_lat_and_pressure'):
      p = cx.SizedAxis('pressure', 2)
      # Create data that varies along pressure: 1.0 at index 0, 2.0 at index 1.
      data_varying = np.ones(p.shape + grid.shape)
      data_varying[1, ...] = 2.0
      inputs_sphere_p = {'u': cx.field(data_varying, p, grid)}

      # Should average over lat (spatial) then mean over pressure (arithmetic).
      # Spatial mean of a constant field is the constant.
      # Mean of 1.0 and 2.0 is 1.5.
      transform = transforms.MeanOverAxes(dims=('latitude', 'pressure'))
      actual = transform(inputs_sphere_p)
      expected = {
          'u': cx.field(np.ones(grid.axes[0].shape) * 1.5, grid.axes[0])
      }
      chex.assert_trees_all_close(actual, expected)

  def test_mask_by_threshold(self):
    x = cx.LabeledAxis('x', np.linspace(0, 1, 10))
    inputs = {
        'mask': x.fields['x'],  # will mask by values of x coordinates.
        'u': cx.field(np.ones(x.shape), x),
        'v': cx.field(np.arange(x.shape[0]), x),
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
        'u': cx.field(np.ones(x.shape), x),
        'v': cx.field(np.arange(x.shape[0]), x),
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
    xy = cx.coords.compose(x, y)
    mask = cx.field(np.array([0.1, np.nan, 10.4, np.nan]), x)
    one_hot_mask = cx.field(np.array([1.0, np.nan, 1.0, np.nan]), x)
    inputs = {
        'mask': mask,
        'nan_at_mask': one_hot_mask * cx.field(np.ones(xy.shape), xy),
        'no_nans': cx.field(np.arange(x.shape[0]), x),  # no nan in v.
        'all_nans': cx.field(np.ones(x.shape) * np.nan, x),
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
        'u': cx.field(np.ones(ylm_grid_21.shape), ylm_grid_21),
        'v': cx.field(np.ones(ylm_grid_31.shape), ylm_grid_31),
        'skipped': cx.field(np.ones(ylm_grid_21.shape), *ylm_dims),
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
      with self.assertRaisesRegex(ValueError, 'No matching grid for'):
        clip_transform(inputs)

  def test_to_modal(self):
    mesh = parallelism.Mesh()
    nodal_grid = coordinates.LonLatGrid.T21()
    inputs = {
        'u': cx.field(np.ones(nodal_grid.shape), nodal_grid),
    }
    with self.subTest('fixed_ylm_mapping_cubic'):
      ylm_grid = coordinates.SphericalHarmonicGrid.T21()
      ylm_map = spherical_harmonics.FixedYlmMapping(
          nodal_grid, ylm_grid, mesh, None
      )
      to_modal = transforms.ToModal(ylm_map)
      actual_out_grid = cx.get_coordinate(to_modal(inputs)['u'])
      expected_ylm_grid = ylm_grid
      self.assertEqual(actual_out_grid, expected_ylm_grid)

    with self.subTest('fixed_ylm_mapping_linear'):
      ylm_grid = coordinates.SphericalHarmonicGrid.TL31()
      ylm_map = spherical_harmonics.FixedYlmMapping(
          nodal_grid, ylm_grid, mesh, None
      )
      to_modal = transforms.ToModal(ylm_map)
      actual_out_grid = cx.get_coordinate(to_modal(inputs)['u'])
      expected_ylm_grid = ylm_grid
      self.assertEqual(actual_out_grid, expected_ylm_grid)

    with self.subTest('ylm_mapper_cubic'):
      ylm_mapper = spherical_harmonics.YlmMapper(
          truncation_rule='cubic', mesh=mesh, partition_schema_key=None
      )
      to_modal = transforms.ToModal(ylm_mapper)
      actual_out_grid = cx.get_coordinate(to_modal(inputs)['u'])
      expected_ylm_grid = coordinates.SphericalHarmonicGrid.T21()
      self.assertEqual(actual_out_grid, expected_ylm_grid)

    with self.subTest('ylm_mapper_linear'):
      ylm_mapper = spherical_harmonics.YlmMapper(
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
        'u': cx.field(np.ones(cubic_ylm_grid.shape), cubic_ylm_grid),
    }
    linear_inputs = {
        'u': cx.field(np.ones(linear_ylm_grid.shape), linear_ylm_grid),
    }
    grid = coordinates.LonLatGrid.T21()
    with self.subTest('fixed_ylm_mapping'):
      ylm_map = spherical_harmonics.FixedYlmMapping(
          grid, cubic_ylm_grid, mesh, None
      )
      to_nodal = transforms.ToNodal(ylm_map)
      actual_out_grid = cx.get_coordinate(to_nodal(cubic_inputs)['u'])
      self.assertEqual(actual_out_grid, grid)
      # transforming from linear inputs.
      ylm_map = spherical_harmonics.FixedYlmMapping(
          grid, linear_ylm_grid, mesh, None
      )
      to_nodal = transforms.ToNodal(ylm_map)
      actual_out_grid = cx.get_coordinate(to_nodal(linear_inputs)['u'])
      self.assertEqual(actual_out_grid, grid)

    with self.subTest('ylm_mapper'):
      ylm_mapper = spherical_harmonics.YlmMapper(
          truncation_rule='cubic', mesh=mesh, partition_schema_key=None
      )
      to_nodal = transforms.ToNodal(ylm_mapper)
      actual_out_grid = cx.get_coordinate(to_nodal(cubic_inputs)['u'])
      self.assertEqual(actual_out_grid, grid)
      # transforming from linear inputs.
      ylm_mapper = spherical_harmonics.YlmMapper(
          truncation_rule='linear', mesh=mesh, partition_schema_key=None
      )
      to_nodal = transforms.ToNodal(ylm_mapper)
      actual_out_grid = cx.get_coordinate(to_nodal(linear_inputs)['u'])
      self.assertEqual(actual_out_grid, grid)

  def test_insert_axis(self):
    x = cx.SizedAxis('x', 3)
    y = cx.SizedAxis('y', 2)
    inputs = {'field_a': cx.field(np.array([1, 2, 3]), x)}

    with self.subTest('insert_by_index'):
      # Insert 'y' at index 0 (before 'x')
      insert_y = transforms.InsertAxis(axis=y, loc=0)
      actual = insert_y(inputs)
      expected = {'field_a': cx.field(np.array([[1, 2, 3], [1, 2, 3]]), y, x)}
      chex.assert_trees_all_close(actual, expected)

    with self.subTest('insert_by_name'):
      # Insert 'y' at loc 'x' (before 'x')
      insert_y = transforms.InsertAxis(axis=y, loc='x')
      actual = insert_y(inputs)
      expected = {'field_a': cx.field(np.array([[1, 2, 3], [1, 2, 3]]), y, x)}
      chex.assert_trees_all_close(actual, expected)

    with self.subTest('insert_at_end'):
      # Insert 'y' at index 1 (after 'x')
      insert_y = transforms.InsertAxis(axis=y, loc=1)
      actual = insert_y(inputs)
      expected = {'field_a': cx.field(np.array([[1, 1], [2, 2], [3, 3]]), x, y)}
      chex.assert_trees_all_close(actual, expected)

    with self.subTest('raises_if_missing'):
      insert_y = transforms.InsertAxis(axis=y, loc='z')
      with self.assertRaisesRegex(ValueError, 'Axis z not present'):
        insert_y(inputs)

  def test_apply_to_keys(self):
    x = cx.SizedAxis('x', 3)
    inputs = {
        'a': cx.field(np.array([1.0, 2.0, 3.0]), x),
        'b': cx.field(np.array([4.0, 5.0, 6.0]), x),
    }
    shift_and_norm = transforms.ShiftAndNormalize(
        shift=cx.field(1.0), scale=cx.field(2.0)
    )
    apply_to_a = transforms.ApplyToKeys(transform=shift_and_norm, keys=['a'])
    actual = apply_to_a(inputs)
    expected = {
        'a': cx.field(np.array([0.0, 0.5, 1.0]), x),
        'b': inputs['b'],
    }
    chex.assert_trees_all_close(actual, expected)

  def test_apply_fn_to_keys(self):
    x = cx.SizedAxis('x', 3)
    inputs = {
        'a': cx.field(np.array([1.0, 2.0, 3.0]), x),
        'b': cx.field(np.array([4.0, 5.0, 6.0]), x),
    }
    double_fn = lambda x: x * 2.0
    apply_to_a = transforms.ApplyFnToKeys(double_fn, ['a'])
    actual = apply_to_a(inputs)
    expected = {'a': cx.field(np.array([2.0, 4.0, 6.0]), x)}
    chex.assert_trees_all_close(actual, expected)

    apply_to_b_keep_a = transforms.ApplyFnToKeys(
        double_fn, ['b'], include_remaining=True
    )
    actual = apply_to_b_keep_a(inputs)
    expected = {
        'a': cx.field(np.array([1.0, 2.0, 3.0]), x),
        'b': cx.field(np.array([8.0, 10.0, 12.0]), x),
    }
    chex.assert_trees_all_close(actual, expected)

  def test_apply_over_axis_with_scan(self):
    nodal_grid = coordinates.LonLatGrid.T21()
    modal_grid = coordinates.SphericalHarmonicGrid.T21()
    mesh = parallelism.Mesh()
    ylm_map = spherical_harmonics.FixedYlmMapping(
        nodal_grid, modal_grid, mesh, None
    )
    to_modal = transforms.ToModal(ylm_map)
    time = cx.SizedAxis('time', 5)
    inputs = {
        'u': cx.field(np.ones(time.shape + nodal_grid.shape), time, nodal_grid),
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
    inputs = {'u': cx.field(np.ones(source_grid.shape), source_grid)}
    transform = transforms.Regrid(
        regridder=interpolators.ConservativeRegridder(target_grid)
    )
    expected = {'u': cx.field(np.ones(target_grid.shape), target_grid)}
    actual = transform(inputs)
    chex.assert_trees_all_close(actual, expected)

  @parameterized.named_parameters(
      dict(testcase_name='scale_10', scale=10.0, atol_identity=1e-1),
      dict(testcase_name='scale_100', scale=100.0, atol_identity=1),
  )
  def test_tanh_clip(self, scale: float, atol_identity: float):
    x_size = 11
    x = cx.SizedAxis('x', x_size)
    inputs = {
        'in_range': cx.field(np.linspace(-scale * 0.3, scale * 0.3, x_size), x),
        'out_of_range': cx.field(np.linspace(-scale * 2, scale * 2, x_size), x),
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

  def test_streaming_stats_norm(self):
    batch = cx.SizedAxis('batch', 20)
    level = cx.SizedAxis('level', 5)
    pressure = cx.SizedAxis('pressure', 7)

    rng = jax.random.PRNGKey(42)
    u_data = jax.random.normal(rng, (batch.size, level.size))
    v_data = jax.random.normal(rng, (batch.size, pressure.size)) + 5.0
    inputs = {
        'u': cx.field(u_data, batch, level),
        'v': cx.field(v_data, batch, pressure),
    }

    with self.subTest('explicit_constructor'):
      norm = transforms.StreamingStatsNorm(
          norm_coords={'u': level, 'v': pressure},
          update_stats=True,
          epsilon=0.0,
      )
      _ = norm(inputs)
      norm.update_stats = False
      outputs = norm(inputs)

      # Verify u statistics (normalized over batch, independent per level)
      u_out = outputs['u'].data
      np.testing.assert_allclose(np.mean(u_out, axis=0), 0.0, atol=1e-5)
      np.testing.assert_allclose(np.var(u_out, axis=0, ddof=1), 1.0, atol=1e-5)

      # Verify v statistics (normalized over batch, independent per pressure)
      v_out = outputs['v'].data
      np.testing.assert_allclose(np.mean(v_out, axis=0), 0.0, atol=1e-5)
      np.testing.assert_allclose(np.var(v_out, axis=0, ddof=1), 1.0, atol=1e-5)

    with self.subTest('for_inputs_struct'):
      norm = transforms.StreamingStatsNorm.for_inputs_struct(
          inputs,
          independent_axes=(level, pressure),
          update_stats=True,
          epsilon=0.0,
      )
      _ = norm(inputs)
      norm.update_stats = False
      outputs = norm(inputs)

      u_out = outputs['u'].data
      np.testing.assert_allclose(np.mean(u_out, axis=0), 0.0, atol=1e-5)
      np.testing.assert_allclose(np.var(u_out, axis=0, ddof=1), 1.0, atol=1e-5)

      v_out = outputs['v'].data
      np.testing.assert_allclose(np.mean(v_out, axis=0), 0.0, atol=1e-5)
      np.testing.assert_allclose(np.var(v_out, axis=0, ddof=1), 1.0, atol=1e-5)

  def test_streaming_stats_norm_with_mask(self):
    rng = np.random.RandomState(42)
    data = rng.normal(size=(10, 4))  # Data with mean 0.
    data[0, 0] = 1e6  # Add an outlier that will be masked out.
    mask_data = np.ones((10, 4), dtype=bool)
    mask_data[0, 0] = False  # Mask must be False to skip data points.
    inputs = {
        'u': cx.field(data, 't', 'x'),
        'outlier_mask': cx.field(mask_data, 't', 'x'),
    }
    make_masks_transform = transforms.Select('outlier_mask')
    norm = transforms.StreamingStatsNorm(
        norm_coords={'u': cx.Scalar()},
        update_stats=True,
        make_masks_transform=make_masks_transform,
        epsilon=0.0,
        skip_unspecified=True,
    )
    outputs = norm(inputs)
    relevant_outputs = outputs['u'].data.ravel()[1:]  # skip the outlier.
    np.testing.assert_allclose(np.mean(relevant_outputs), 0.0, atol=1e-4)
    np.testing.assert_allclose(np.var(relevant_outputs, ddof=1), 1.0, atol=1e-4)

  def test_scale_to_match_coarse_fields(self):
    hres_grid = coordinates.LonLatGrid.TL63()
    coarse_grid = coordinates.LonLatGrid.T21()
    keys = ['precip', 'temp']
    hres_ones = cx.field(np.ones(hres_grid.shape, dtype=np.float32), hres_grid)
    coarse_twos = cx.field(
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

  def test_velocity_div_curl_roundtrip(self):
    mesh = parallelism.Mesh()
    ylm_map = spherical_harmonics.YlmMapper(
        truncation_rule='cubic', mesh=mesh, partition_schema_key=None
    )
    modal_grid = coordinates.SphericalHarmonicGrid.T21()
    nodal_grid = coordinates.LonLatGrid.T21()
    rng = jax.random.PRNGKey(42)
    u_data = jax.random.normal(rng, nodal_grid.shape)
    v_data = jax.random.normal(rng, nodal_grid.shape)
    u = cx.field(u_data, nodal_grid)
    v = cx.field(v_data, nodal_grid)
    nodal_inputs = {'u_component_of_wind': u, 'v_component_of_wind': v}
    to_modal = transforms.ToModal(ylm_map)
    l_ax = modal_grid.axes[1]
    scales = np.exp(-0.5 * np.arange(l_ax.sizes['total_wavenumber']))
    scales = cx.field(scales, l_ax)
    filter_fn = lambda in_dict: {k: v * scales for k, v in in_dict.items()}
    to_nodal = transforms.ToNodal(ylm_map)
    nodal_inputs = to_nodal(filter_fn(to_modal(nodal_inputs)))

    div_curl_to_uv = transforms.VelocityFromModalDivCurl(ylm_map)
    uv_to_div_curl = transforms.DivCurlFromNodalVelocity(ylm_map)

    # test nodal -> modal -> nodal roundtrip
    div_curl_outputs = uv_to_div_curl(nodal_inputs)
    self.assertEqual(
        cx.get_coordinate(div_curl_outputs['divergence']), modal_grid
    )
    self.assertEqual(
        cx.get_coordinate(div_curl_outputs['vorticity']), modal_grid
    )
    uv_recon = div_curl_to_uv(div_curl_outputs)
    chex.assert_trees_all_close(
        uv_recon['u_component_of_wind'],
        nodal_inputs['u_component_of_wind'],
        atol=7.5e-3,
    )
    chex.assert_trees_all_close(
        uv_recon['v_component_of_wind'],
        nodal_inputs['v_component_of_wind'],
        atol=7.5e-3,
    )

  def test_constrain_precipitation_and_evaporation(self):
    x = cx.SizedAxis('x', 4)
    precip_key = 'precipitation'
    evap_key = 'evaporation'

    with self.subTest('diagnose_precip'):
      p_plus_e_val = cx.field(np.array([-2.0, -1.0, 0.5, 1.0]), x)
      evap_in = cx.field(np.array([-3.0, -0.5, -1.0, 0.2]), x)
      p_plus_e_diag = diagnostics.InstantDiagnostic(
          extract=lambda *args, **kwargs: {
              'precipitation_plus_evaporation_rate': p_plus_e_val
          },
          extract_coords={'precipitation_plus_evaporation_rate': x},
      )
      p_plus_e_diag({}, prognostics={})  # call to compute and store values
      inputs = {evap_key: evap_in}
      transform = transforms.ConstrainPrecipitationAndEvaporation(
          p_plus_e_diagnostic=p_plus_e_diag,
          var_to_constrain=evap_key,
          precipitation_key=precip_key,
          evaporation_key=evap_key,
      )
      actual = transform(inputs)
      # in evap case, result should be min(evap_in, min(p_plus_e, 0))
      # constrained_evap=min([-3,-2],[-0.5,-1],[-1,0],[0.2,0])
      expected_evap_val = np.array([-3.0, -1.0, -1.0, 0.0])
      cx.testing.assert_fields_allclose(
          actual[evap_key], cx.field(expected_evap_val, x)
      )
      # diagnosed precip = p_plus_e - constrained_evap
      expected_precip_val = p_plus_e_val.data - expected_evap_val
      cx.testing.assert_fields_allclose(
          actual[precip_key], cx.field(expected_precip_val, x)
      )

    with self.subTest('diagnose_evap'):
      p_plus_e_val = cx.field(np.array([-1.0, 0.5, 2.0, 3.0]), x)
      precip_in = cx.field(np.array([-2.0, -1.0, 1.0, 4.0]), x)
      p_plus_e_diag = diagnostics.InstantDiagnostic(
          extract=lambda *args, **kwargs: {
              'precipitation_plus_evaporation_rate': p_plus_e_val
          },
          extract_coords={'precipitation_plus_evaporation_rate': x},
      )
      p_plus_e_diag({}, prognostics={})  # call to compute and store values
      inputs = {precip_key: precip_in}
      transform = transforms.ConstrainPrecipitationAndEvaporation(
          p_plus_e_diagnostic=p_plus_e_diag,
          var_to_constrain=precip_key,
          precipitation_key=precip_key,
          evaporation_key=evap_key,
      )
      actual = transform(inputs)
      # in precip case, result should be max(precip_in, max(p_plus_e, 0))
      # constrained_precip = max([-2,0],[-1,0.5],[1,2],[4,3]) = [0, 0.5, 2, 4]
      expected_precip_val = np.array([0.0, 0.5, 2.0, 4.0])
      cx.testing.assert_fields_allclose(
          actual[precip_key], cx.field(expected_precip_val, x)
      )
      expected_evap_val = p_plus_e_val.data - expected_precip_val
      cx.testing.assert_fields_allclose(
          actual[evap_key], cx.field(expected_evap_val, x)
      )

  def test_inpaint_mask_for_harmonics(self):
    mesh = parallelism.Mesh()
    grid = coordinates.LonLatGrid.T21()
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    ylm_map = spherical_harmonics.FixedYlmMapping(grid, ylm_grid, mesh, None)

    # Use a middle latitudinal band as a demo mask.
    mask_data = np.zeros(grid.shape, dtype=bool)
    lat_idx = grid.shape[1] // 2
    mask_data[:, lat_idx] = True  # these values will be inpainted.
    mask = cx.field(mask_data, grid)

    # Corrupt the data under the mask
    u_data = np.ones(grid.shape)
    u_corrupted_data = u_data.copy()
    u_corrupted_data[:, lat_idx] = 100.0  # Large outlier to be inpainted.
    u_corrupted = cx.field(u_corrupted_data, grid)
    inputs = {'u': u_corrupted}

    lowpass = spatial_filters.ExponentialModalFilter(
        ylm_map, attenuation=2.0, order=1
    )
    prescirbed_mask = transforms.Prescribed({'u': mask})
    inpaint = transforms.InpaintMaskForHarmonics(
        ylm_map=ylm_map,
        get_masks_transform=prescirbed_mask,
        lowpass_filter=lowpass,
        n_iter=5,
    )
    output = inpaint(inputs)

    valid_mask = ~mask_data
    np.testing.assert_allclose(
        output['u'].data[valid_mask], u_corrupted.data[valid_mask]
    )

    inpainted_values = output['u'].data[mask_data]
    np.testing.assert_array_less(inpainted_values, 50.0)
    np.testing.assert_allclose(inpainted_values, 1.0, atol=1e-5)


class PointNeighborsFromGridTest(parameterized.TestCase):

  def test_nearest_neighbor_on_spherical_grid(self):
    """Verifies that width=1 returns the single nearest value on a TL63 grid."""
    grid = coordinates.LonLatGrid.TL63()
    lons = grid.fields['longitude']
    lats = grid.fields['latitude']
    # Use complex values to encode (longitude, latitude) into a single field.
    field = lons + 1j * lats
    transform = transforms.PointNeighborsFromGrid(width=1)

    with self.subTest('on_grid'):
      lon_idx, lat_idx = 12, 15
      lon_q, lat_q = lons.data[lon_idx], lats.data[lat_idx]
      output = transform({
          'longitude': cx.field(lon_q),
          'latitude': cx.field(lat_q),
          'f': field,
      })
      self.assertEqual(output['f'].data, lon_q + 1j * lat_q)

    with self.subTest('off_grid'):
      lon_idx, lat_idx = 0, 2
      lon_q, lat_q = lons.data[lon_idx], lats.data[lat_idx]
      # delta is smaller than half-spacing, should round to the original.
      lon_off, lat_off = lon_q + 0.1, lat_q + 0.1
      output = transform({
          'longitude': cx.field(lon_off),
          'latitude': cx.field(lat_off),
          'f': field,
      })
      self.assertEqual(output['f'].data, lon_q + 1j * lat_q)

  def test_even_width_patch(self):
    """Verifies even width=2 patch orientation around a query point."""
    lons = np.arange(0, 360, 10)
    lats = np.arange(-90, 100, 10)
    grid = cx.coords.compose(
        cx.LabeledAxis('longitude', lons),
        cx.LabeledAxis('latitude', lats),
    )
    field = cx.field(lons[:, None] + 1j * lats[None, :], grid)
    transform = transforms.PointNeighborsFromGrid(width=2)

    # Query point at (15, 15), which is in the center of the 4 grid points:
    # (10, 10), (10, 20), (20, 10), (20, 20).
    output = transform({
        'longitude': cx.field(15.0),
        'latitude': cx.field(15.0),
        'f': field,
    })

    # Expected patch is 2x2 with these 4 values.
    # Orientation in cx.Field is (longitude, latitude).
    expected = [
        [10 + 10j, 10 + 20j],
        [20 + 10j, 20 + 20j],
    ]
    np.testing.assert_array_equal(output['f'].data, expected)

  def test_odd_width_patch(self):
    """Verifies odd width=3 patch is centered on the nearest neighbor."""
    lons = np.arange(0, 360, 10)
    lats = np.arange(-90, 100, 10)
    grid = cx.coords.compose(
        cx.LabeledAxis('longitude', lons),
        cx.LabeledAxis('latitude', lats),
    )
    field = cx.field(lons[:, None] + 1j * lats[None, :], grid)
    transform = transforms.PointNeighborsFromGrid(width=3)

    # Queries within +\-5 of (20, 20) should return the same patch.
    # Width 3 patch should be centered at (20, 20), i.e., indices [1, 2, 3]
    # for both dims, which corresponds to values [10, 20, 30].
    expected = [
        [10 + 10j, 10 + 20j, 10 + 30j],
        [20 + 10j, 20 + 20j, 20 + 30j],
        [30 + 10j, 30 + 20j, 30 + 30j],
    ]
    for lon in [16, 23]:
      for lat in [17, 22]:
        output = transform({
            'longitude': cx.field(lon),
            'latitude': cx.field(lat),
            'f': field,
        })
        np.testing.assert_array_equal(output['f'].data, expected)

  def test_longitude_wrapping_at_equator(self):
    """Tests longitude wrapping at the central band (equator)."""
    lons = np.arange(0, 360, 60)  # [0, 60, 120, 180, 240, 300]
    lats = np.arange(-20, 30, 10)  # [-20, -10, 0, 10, 20] -> 0 is index 2.
    grid = cx.coords.compose(
        cx.LabeledAxis('longitude', lons),
        cx.LabeledAxis('latitude', lats),
    )
    field = cx.field(lons[:, None] + 1j * lats[None, :], grid)
    transform = transforms.PointNeighborsFromGrid(width=2)

    # Query at 330 (between 300 and 0) and latitude 5 (between 0 and 10).
    output = transform({
        'longitude': cx.field(330.0),
        'latitude': cx.field(5.0),
        'f': field,
    })
    expected = [
        [300 + 0j, 300 + 10j],
        [0 + 0j, 0 + 10j],
    ]
    np.testing.assert_array_equal(output['f'].data, expected)

  def test_multiple_grids_and_batching(self):
    """Tests handling of multiple fields on different grids and batched queries."""
    # Field 1 on a 10-degree grid.
    grid1 = cx.coords.compose(
        cx.LabeledAxis('longitude', np.arange(0, 40, 10)),
        cx.LabeledAxis('latitude', np.arange(0, 40, 10)),
    )
    field1 = cx.field(np.arange(16).reshape(4, 4), grid1)

    # Field 2 on a shifted by 5-degree grid.
    grid2 = cx.coords.compose(
        cx.LabeledAxis('longitude', np.arange(5, 45, 10)),
        cx.LabeledAxis('latitude', np.arange(5, 45, 10)),
    )
    field2 = cx.field(np.arange(16).reshape(4, 4) + 100, grid2)

    # Batched query:
    sparse = cx.SizedAxis('points', 2)
    inputs = {
        'longitude': cx.field(np.array([8.0, 24.0]), sparse),
        'latitude': cx.field(np.array([9.0, 24.0]), sparse),
        'f1': field1,
        'f2': field2,
    }
    transform = transforms.PointNeighborsFromGrid(width=1)
    output = transform(inputs)

    # For point 0: (8, 9).
    # f1: nearest is (10, 10) at index (1, 1) -> 5.
    # f2: nearest is (5, 5) at index (0, 0) -> 100.
    # For point 1: (24, 24).
    # f1: nearest is (20, 20) at index (2, 2) -> 10.
    # f2: nearest is (25, 25) at index (2, 2) -> 110.
    np.testing.assert_array_equal(output['f1'].data, [5, 10])
    np.testing.assert_array_equal(output['f2'].data, [100, 110])

  def test_extra_dimensions(self):
    """Tests that PointNeighborsFromGrid handles extra dimensions correctly."""
    grid = coordinates.LonLatGrid.T21()
    z_axis = cx.SizedAxis('z', 4)
    f1 = cx.field(np.ones(z_axis.shape + grid.shape), z_axis, grid)
    f2 = cx.field(np.ones(grid.shape), grid)

    sparse = cx.SizedAxis('points', 2)
    inputs = {
        'longitude': cx.field(np.array([10.0, 30.0]), sparse),
        'latitude': cx.field(np.array([5.0, 25.0]), sparse),
        'f1': f1,
        'f2': f2,
    }
    transform = transforms.PointNeighborsFromGrid(width=1, include_offset=True)
    output = transform(inputs)

    self.assertEqual(output['f1'].dims, ('points', 'z'))
    self.assertEqual(output['f2'].dims, ('points',))
    self.assertIn('delta_longitude_0', output)
    self.assertIn('delta_latitude_0', output)


#
# Sanitize nans transform test with demo transforms.
#


class SimpleScale(transforms.TransformABC):
  """Simple stateless transform that scales input."""

  def __init__(self, scale):
    self.scale = nnx.Param(scale)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    s = self.scale.get_value()
    return {k: v * s for k, v in inputs.items()}


class IncrementStateAndAdd(transforms.TransformABC):
  """Stateful transform that increments a counter and adds it to input."""

  def __init__(self, init_value: float = 1.0):
    self.a = nnx.Param(init_value)

  def __call__(self, inputs):
    to_add = self.a.get_value()
    self.a.set_value(to_add + 1)
    return {k: v + to_add for k, v in inputs.items()}


class MultiplyStateAndAdd(transforms.TransformABC):
  """Stateful transform that multiplies state and adds it to input."""

  def __init__(self, init_value: float = 1.0, scale: float = 2.0):
    self.a = nnx.Param(init_value)
    self.scale = scale

  def __call__(self, inputs):
    val = self.a.get_value()
    self.a.set_value(val * self.scale)
    return {k: v + val for k, v in inputs.items()}


def mean_squared_a_err(transform, inputs, targets):
  """Computes MSE loss plus state change to track gradients through state."""
  init_param_sum = sum(jax.tree.leaves(nnx.state(transform)))
  prediction = transform(inputs)
  post_param_sum = sum(jax.tree.leaves(nnx.state(transform)))
  a_squared_errors = cx.cpmap(jnp.square)(prediction['a'] - targets['a'])
  # Use nanmean to handle NaNs in forward pass output if any
  return jnp.nanmean(a_squared_errors.data) + post_param_sum - init_param_sum


class SanitizedNanGradTransformTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='stateless_simple',
          transform_factory=lambda: transforms.Sequential([
              SimpleScale(2.0),
              transforms.ApplyFnToKeys(cx.cpmap(jnp.square), ['a'], True),
          ]),
          inputs={'a': cx.field(np.array([0.0, 1.0, 2.0]), 'x')},
      ),
      dict(
          testcase_name='stateful_increment',
          transform_factory=lambda: transforms.Sequential([
              SimpleScale(2.0),
              IncrementStateAndAdd(0.0),
          ]),
          inputs={'a': cx.field(np.array([0.0, 1.0, 2.0]), 'x')},
      ),
      dict(
          testcase_name='stateful_multiply',
          transform_factory=lambda: transforms.Sequential([
              SimpleScale(2.0),
              MultiplyStateAndAdd(1.0, 3.0),
          ]),
          inputs={'a': cx.field(np.array([0.0, 1.0, 2.0]), 'x')},
      ),
      dict(
          testcase_name='stateful_multiply_chained',
          transform_factory=lambda: transforms.Sequential([
              SimpleScale(2.0),
              MultiplyStateAndAdd(1.0, 3.0),
              MultiplyStateAndAdd(1.0, 3.0),
          ]),
          inputs={'a': cx.field(np.array([0.0, 1.0, 2.0]), 'x')},
      ),
  )
  def test_equivalence_no_nans(
      self,
      transform_factory: Callable[[], transforms.TransformABC],
      inputs: dict[str, cx.Field],
  ):
    """Verifies sanitized matches original for clean inputs."""
    # Compute targets using throwaway instance to avoid state mutation
    targets = {'a': transform_factory()(inputs)['a'] + 0.5}

    # Reference run
    original = transform_factory()
    out_orig, grads_orig = nnx.value_and_grad(mean_squared_a_err, argnums=0)(
        original, inputs, targets
    )

    # Sanitized run
    sanitized = transforms.SanitizeNanGradTransform(transform_factory())
    out_san, grads_san = nnx.value_and_grad(mean_squared_a_err, argnums=0)(
        sanitized, inputs, targets
    )

    np.testing.assert_allclose(out_san, out_orig, err_msg='output mismatch')
    chex.assert_trees_all_close(grads_san['transform'], grads_orig)

  @parameterized.named_parameters(
      dict(
          testcase_name='stateless_simple_nan',
          transform_factory=lambda: transforms.Sequential([
              SimpleScale(2.0),
              transforms.ApplyFnToKeys(cx.cpmap(jnp.square), ['a'], True),
          ]),
          inputs={'a': cx.field(np.array([0.0, np.nan, 2.0]), 'x')},
          inputs_filtered={'a': cx.field(np.array([0.0, 2.0]), 'x')},
      ),
      dict(
          testcase_name='stateful_multiply_nan',
          transform_factory=lambda: transforms.Sequential([
              SimpleScale(2.0),
              MultiplyStateAndAdd(1.0, 3.0),
          ]),
          inputs={'a': cx.field(np.array([np.nan, 1.0, 2.0]), 'x')},
          inputs_filtered={'a': cx.field(np.array([1.0, 2.0]), 'x')},
      ),
  )
  def test_gradients_with_nans(
      self,
      transform_factory: Callable[[], transforms.TransformABC],
      inputs: dict[str, cx.Field],
      inputs_filtered: dict[str, cx.Field],
  ):
    """Verifies gradients match reference run on valid data subset."""
    targets_full = {'a': transform_factory()(inputs)['a'] + 0.5}  # with Nans.
    targets_filtered = {'a': transform_factory()(inputs_filtered)['a'] + 0.5}

    # Reference run using filtered data without nans.
    original = transform_factory()
    out_orig, grads_orig = nnx.value_and_grad(mean_squared_a_err, argnums=0)(
        original, inputs_filtered, targets_filtered
    )

    # Sanitized Run (Full with NaNs), should match reference.
    sanitized = transforms.SanitizeNanGradTransform(transform_factory())
    out_san, grads_san = nnx.value_and_grad(mean_squared_a_err, argnums=0)(
        sanitized, inputs, targets_full
    )

    # Check finite sanitized gradients.
    grads_san_inner = grads_san['transform']
    leaves_san = jax.tree.leaves(grads_san_inner)
    self.assertTrue(
        all(np.all(np.isfinite(l)) for l in leaves_san),
        msg=f'Gradients contain NaNs: {leaves_san}',
    )

    np.testing.assert_allclose(out_san, out_orig, err_msg='output mismatch')
    chex.assert_trees_all_close(grads_san_inner, grads_orig)


class NestedTransformTest(parameterized.TestCase):

  def test_single_transform(self):
    """Tests backward compatibility with single transform."""
    transform = transforms.NestedTransform(transforms.Identity())
    inputs = {'a': {'x': cx.field(1.0)}, 'b': {'y': cx.field(2.0)}}
    actual = transform(inputs)
    chex.assert_trees_all_close(actual, inputs)

  def test_dict_transform_specific_keys(self):
    """Tests using a dictionary of transforms."""
    scale_2 = transforms.Scale(cx.field(2.0))
    scale_3 = transforms.Scale(cx.field(3.0))
    transform = transforms.NestedTransform({'a': scale_2, 'b': scale_3})
    inputs = {
        'a': {'val': cx.field(np.array([1.0]), 'x')},
        'b': {'val': cx.field(np.array([1.0]), 'x')},
    }
    actual = transform(inputs)
    expected = {
        'a': {'val': cx.field(np.array([2.0]), 'x')},
        'b': {'val': cx.field(np.array([3.0]), 'x')},
    }
    chex.assert_trees_all_close(actual, expected)

  def test_dict_transform_with_ellipsis(self):
    """Tests using Ellipsis as default transform."""
    scale_2 = transforms.Scale(cx.field(2.0))
    scale_10 = transforms.Scale(cx.field(10.0))
    transform = transforms.NestedTransform({'a': scale_2, ...: scale_10})
    inputs = {
        'a': {'val': cx.field(np.array([1.0]), 'x')},
        'b': {'val': cx.field(np.array([1.0]), 'x')},
        'c': {'val': cx.field(np.array([1.0]), 'x')},
    }
    actual = transform(inputs)
    expected = {
        'a': {'val': cx.field(np.array([2.0]), 'x')},
        'b': {'val': cx.field(np.array([10.0]), 'x')},
        'c': {'val': cx.field(np.array([10.0]), 'x')},
    }
    chex.assert_trees_all_close(actual, expected)

  def test_dict_transform_missing_key_raises(self):
    """Tests ValueError is raised if a key is missing and no default is set."""
    scale_2 = transforms.Scale(cx.field(2.0))
    transform = transforms.NestedTransform({'a': scale_2})
    inputs = {
        'a': {'val': cx.field(np.array([1.0]), 'x')},
        'missing_key': {'val': cx.field(np.array([1.0]), 'x')},
    }
    with self.assertRaisesRegex(
        ValueError, "No default or key-specific transform for k='missing_key'"
    ):
      transform(inputs)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
