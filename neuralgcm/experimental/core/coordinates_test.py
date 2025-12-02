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
import collections
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from coordax import testing as coordax_testing
from dinosaur import sigma_coordinates
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import xarray_utils
import numpy as np


class CoordinatesTest(parameterized.TestCase):
  """Tests that coordinate have expected shapes and dims."""

  @parameterized.named_parameters(
      dict(
          testcase_name='spherical_harmonic',
          coords=coordinates.SphericalHarmonicGrid.TL31(),
          expected_dims=('longitude_wavenumber', 'total_wavenumber'),
          expected_shape=(64, 33),
      ),
      dict(
          testcase_name='lon_lat',
          coords=coordinates.LonLatGrid.T21(),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(64, 32),
      ),
      dict(
          testcase_name='product_of_levels',
          coords=cx.compose_coordinates(
              coordinates.SigmaLevels.equidistant(4),
              coordinates.PressureLevels([50, 100, 200, 800, 1000]),
              coordinates.HybridLevels.with_n_levels(7),
              coordinates.LayerLevels(3),
          ),
          expected_dims=('sigma', 'pressure', 'hybrid', 'layer_index'),
          expected_shape=(4, 5, 7, 3),
      ),
      dict(
          testcase_name='sigma_and_sigma_boundaries',
          coords=cx.compose_coordinates(
              coordinates.SigmaLevels.equidistant(4),
              coordinates.SigmaBoundaries.equidistant(4),
          ),
          expected_dims=('sigma', 'sigma_boundaries'),
          expected_shape=(4, 5),
      ),
      dict(
          testcase_name='sigma_spherical_harmonic_product',
          coords=cx.compose_coordinates(
              coordinates.SigmaLevels.equidistant(4),
              coordinates.SphericalHarmonicGrid.T21(),
          ),
          expected_dims=('sigma', 'longitude_wavenumber', 'total_wavenumber'),
          expected_shape=(4, 44, 23),
      ),
      dict(
          testcase_name='dinosaur_primitive_equation_coords',
          coords=coordinates.DinosaurCoordinates(
              horizontal=coordinates.SphericalHarmonicGrid.T21(),
              vertical=coordinates.SigmaLevels.equidistant(4),
          ),
          expected_dims=('sigma', 'longitude_wavenumber', 'total_wavenumber'),
          expected_shape=(4, 44, 23),
      ),
      dict(
          testcase_name='batched_trajectory',
          coords=cx.compose_coordinates(
              cx.SizedAxis('batch', 7),
              coordinates.TimeDelta(np.arange(5) * np.timedelta64(1, 'h')),
              coordinates.PressureLevels([50, 200, 800, 1000]),
              coordinates.LonLatGrid.T21(),
          ),
          expected_dims=(
              'batch',
              'timedelta',
              'pressure',
              'longitude',
              'latitude',
          ),
          expected_shape=(7, 5, 4, 64, 32),
          expected_field_transform=lambda f: f.untag('batch').tag('batch'),
      ),
      dict(
          testcase_name='coordinate_shard_none',
          coords=parallelism.CoordinateShard(
              coordinate=coordinates.LonLatGrid.T42(),
              spmd_mesh_shape=collections.OrderedDict(x=2, y=1, z=2),
              dimension_partitions={'longitude': None, 'latitude': None},
          ),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(128, 64),  # unchanged.
          supports_xarray_roundtrip=False,
      ),
      dict(
          testcase_name='coordinate_shard_longitude',
          coords=parallelism.CoordinateShard(
              coordinate=coordinates.LonLatGrid.T42(),
              spmd_mesh_shape=collections.OrderedDict(x=2, y=1, z=2),
              dimension_partitions={'longitude': ('x', 'z'), 'latitude': None},
          ),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(32, 64),  # unchanged.
          supports_xarray_roundtrip=False,
      ),
      dict(
          testcase_name='coordinate_shard_longitude_and_latitude',
          coords=parallelism.CoordinateShard(
              coordinate=coordinates.LonLatGrid.T42(),
              spmd_mesh_shape=collections.OrderedDict(x=2, y=4, z=2),
              dimension_partitions={'longitude': 'x', 'latitude': ('y', 'z')},
          ),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(64, 8),  # unchanged.
          supports_xarray_roundtrip=False,
      ),
  )
  def test_coordinates(
      self,
      coords: cx.Coordinate,
      expected_dims: tuple[str, ...],
      expected_shape: tuple[int, ...],
      expected_field_transform: Callable[[cx.Field], cx.Field] = lambda x: x,
      supports_xarray_roundtrip: bool = True,
  ):
    """Tests that coordinates are pytrees and have expected shape and dims."""
    with self.subTest('pytree_roundtrip'):
      leaves, tree_def = jax.tree.flatten(coords)
      reconstructed = jax.tree.unflatten(tree_def, leaves)
      self.assertEqual(reconstructed, coords)

    with self.subTest('dims'):
      self.assertEqual(coords.dims, expected_dims)

    with self.subTest('shape'):
      self.assertEqual(coords.shape, expected_shape)

    if supports_xarray_roundtrip:
      with self.subTest('xarray_roundtrip'):
        field = cx.wrap(np.zeros(coords.shape), coords)
        data_array = field.to_xarray()
        reconstructed = xarray_utils.field_from_xarray(data_array)
        expected = expected_field_transform(field)
        coordax_testing.assert_fields_equal(reconstructed, expected)


class CoordinatesMethodsTest(parameterized.TestCase):
  """Tests methods of coordinate objects."""

  @parameterized.named_parameters(
      dict(
          testcase_name='sigma_axis_minus_3',
          shape=(4, 5, 3),
          sigma_axis=-3,
      ),
      dict(
          testcase_name='sigma_axis_1',
          shape=(5, 4, 3),
          sigma_axis=1,
      ),
  )
  def test_sigma_level_integrate(self, shape, sigma_axis):
    sigma_coord = coordinates.SigmaLevels.equidistant(shape[sigma_axis])
    pos_sigma_axis = sigma_axis if sigma_axis >= 0 else sigma_axis + len(shape)
    coords = cx.compose_coordinates(*[
        sigma_coord if i == pos_sigma_axis else cx.SizedAxis(f'ax{i}', shape[i])
        for i in range(len(shape))
    ])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    field = cx.wrap(data, coords)
    integrated_field = sigma_coord.integrate(field)
    expected_data = sigma_coordinates.sigma_integral(
        data,
        sigma_coord.sigma_levels,
        axis=sigma_axis,
        keepdims=False,
    )
    np.testing.assert_allclose(integrated_field.data, expected_data, atol=1e-6)
    expected_dims = tuple(d for d in coords.dims if d not in sigma_coord.dims)
    self.assertEqual(integrated_field.dims, expected_dims)

  @parameterized.named_parameters(
      dict(
          testcase_name='sigma_axis_minus_3',
          shape=(4, 5, 3),
          sigma_axis=-3,
      ),
      dict(
          testcase_name='sigma_axis_1',
          shape=(5, 4, 3),
          sigma_axis=1,
      ),
  )
  def test_sigma_level_integrate_cumulative(self, shape, sigma_axis):
    sigma_coord = coordinates.SigmaLevels.equidistant(shape[sigma_axis])
    pos_sigma_axis = sigma_axis if sigma_axis >= 0 else sigma_axis + len(shape)
    coords = cx.compose_coordinates(*[
        sigma_coord if i == pos_sigma_axis else cx.SizedAxis(f'ax{i}', shape[i])
        for i in range(len(shape))
    ])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    field = cx.wrap(data, coords)
    integrated_field = sigma_coord.integrate_cumulative(field)
    expected_data = sigma_coordinates.cumulative_sigma_integral(
        data,
        sigma_coord.sigma_levels,
        axis=sigma_axis,
    )
    np.testing.assert_allclose(integrated_field.data, expected_data, atol=1e-6)
    self.assertEqual(integrated_field.dims, coords.dims)

  def test_lon_lat_grid_integrate(self):
    grid = coordinates.LonLatGrid.T21()
    field = cx.wrap(np.ones(grid.shape), grid)
    radius = 123.4
    integral = grid.integrate(field, radius=radius)
    np.testing.assert_allclose(integral.data, 4 * np.pi * radius**2, rtol=1e-5)

  def test_lon_lat_grid_partial_integrate(self):
    n_lon, n_lat = 64, 32
    grid = coordinates.LonLatGrid(
        longitude_nodes=n_lon,
        latitude_nodes=n_lat,
        lon_lat_padding=(8, 4),
    )
    # data that is 1 everywhere except 2 on first half of longitudes
    data = np.ones(grid.shape)
    data[: (n_lon // 2), :] = 2
    field = cx.wrap(data, grid)

    field_lat = grid.integrate(field, dims='longitude')
    self.assertEqual(field_lat.coordinate, cx.SelectedAxis(grid, axis=1))

    field_lon = grid.integrate(field, dims='latitude')
    self.assertEqual(field_lon.coordinate, cx.SelectedAxis(grid, axis=0))

    scalar = grid.integrate(field)
    self.assertEqual(scalar.coordinate, cx.Scalar())

    sclar_from_lon = grid.integrate(field_lon, dims='longitude')
    sclar_from_lat = grid.integrate(field_lat, dims='latitude')
    cx.testing.assert_fields_allclose(sclar_from_lon, scalar)
    cx.testing.assert_fields_allclose(sclar_from_lat, scalar)

  @parameterized.named_parameters(
      dict(testcase_name='float', c=2.5),
      dict(testcase_name='array', c=np.array(2.5)),
      dict(
          testcase_name='field_with_named_axes',
          c=cx.wrap(np.eye(3), cx.SizedAxis('a', 3), cx.SizedAxis('b', 3)),
      ),
  )
  def test_spherical_harmonic_grid_add_constant(self, c):
    """Tests that `add_constant` is consistent with nodal addition."""
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    a, b = cx.SizedAxis('a', 3), cx.SizedAxis('b', 3)
    levels = coordinates.SigmaLevels.equidistant(4)
    coords = cx.compose_coordinates(a, levels, b, ylm_grid)
    x_data = np.zeros(coords.shape)
    rng = np.random.RandomState(4)
    x_data[:, :, :, 0, 0] = 0.13
    x_data[:, :, :, 2:4, 2:6] = rng.uniform(size=(2, 4))
    x = cx.wrap(x_data, coords)
    grid = coordinates.LonLatGrid.T21()
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=grid,
        ylm_grid=ylm_grid,
        mesh=parallelism.Mesh(),
        partition_schema_key=None,
    )
    expected = ylm_map.to_modal(ylm_map.to_nodal(x) + c)
    actual = ylm_grid.add_constant(x, c)
    coordax_testing.assert_fields_allclose(actual, expected, atol=1e-5)

  def test_spherical_harmonic_grid_add_constant_raises_with_positional_axes(
      self,
  ):
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    x = cx.wrap(np.zeros(ylm_grid.shape), ylm_grid)
    c = cx.wrap(np.arange(3.0))
    with self.assertRaisesRegex(
        ValueError, 'Adding non-scalar constants without axes is not supported'
    ):
      ylm_grid.add_constant(x, c)

  def test_spherical_harmonic_grid_add_constant_raises_with_conflicting_axes(
      self,
  ):
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    x = cx.wrap(np.zeros(ylm_grid.shape), ylm_grid)
    conflicting_axis = cx.SizedAxis('total_wavenumber', ylm_grid.shape[-1])
    c = cx.wrap(np.zeros(ylm_grid.shape[-1]), conflicting_axis)
    with self.assertRaisesRegex(
        ValueError, 'cannot have any of the dimensions'
    ):
      ylm_grid.add_constant(x, c)

  def test_spherical_harmonic_grid_add_constant_raises_with_new_axes(self):
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    levels = coordinates.SigmaLevels.equidistant(4)
    x = cx.wrap(np.zeros(levels.shape + ylm_grid.shape), levels, ylm_grid)
    c = cx.wrap(np.arange(5), cx.SizedAxis('new_dim', 5))
    with self.assertRaisesRegex(
        ValueError, 'Introduction of new axes via add_constant is not supported'
    ):
      ylm_grid.add_constant(x, c)


if __name__ == '__main__':
  absltest.main()
