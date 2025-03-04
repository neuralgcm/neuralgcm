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
"""Tests utilities for converting between xarray and coordax objects."""

import collections

from absl.testing import absltest
from absl.testing import parameterized
import jax
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import jax_datetime as jdt
from neuralgcm.experimental import xarray_utils
import numpy as np
import xarray


class ReadFieldsFromXarrayTest(parameterized.TestCase):
  """Tests utilities for reading fields from xarray configured via specs."""

  def setUp(self):
    super().setUp()
    grid = coordinates.LonLatGrid.TL31()
    levels = coordinates.PressureLevels.with_era5_levels()
    timedelta_values = np.arange(3, dtype='timedelta64[D]')
    timedelta = coordinates.TimeDelta(timedelta_values)

    volume_variables = ['geopotential', 'temperature']
    surface_variables = ['sst', '2m_temperature']

    ones_like = lambda coord: cx.wrap(np.ones(coord.shape), coord)
    volume_coord = cx.compose_coordinates(timedelta, levels, grid)
    surface_coord = cx.compose_coordinates(timedelta, grid)
    volume_fields = {k: ones_like(volume_coord) for k in volume_variables}
    surface_fields = {k: ones_like(surface_coord) for k in surface_variables}
    t0 = np.datetime64('2024-01-01')
    other_fields = {
        'time': cx.wrap(jdt.to_datetime(t0 + timedelta_values), timedelta),
        'global_scalar': cx.wrap(np.linspace(0, np.pi, 3), timedelta),
    }
    volume_vars = {k: v.to_xarray() for k, v in volume_fields.items()}
    surface_vars = {k: v.to_xarray() for k, v in surface_fields.items()}
    other_vars = {k: v.to_xarray() for k, v in other_fields.items()}
    self.mock_ds = xarray.Dataset(volume_vars | surface_vars | other_vars)
    self.grid = grid
    self.levels = levels
    self.timedelta = timedelta

  def assert_data_and_specs_keys_match(
      self,
      actual: dict[str, dict[str, cx.Field]],
      specs: dict[str, dict[str, cx.Coordinate]],
  ):
    """Tests that actual data and specs have matching keys."""
    self.assertSameElements(actual.keys(), specs.keys())
    for k in actual.keys():
      self.assertSameElements(actual[k].keys(), specs[k].keys())

  def test_read_fields_from_xarray_expected_structure(self):
    """Tests that read_fields_from_xarray returns a structure matching specs."""
    coords = cx.compose_coordinates(self.levels, self.grid)
    input_specs = {
        'era5': {
            'geopotential': coords,
            'temperature': coords,
            'sst': self.grid,
            'time': cx.Scalar(),
        },
        'era5:surface': {'2m_temperature': self.grid},
        'era5:g': {
            'global_scalar': cx.Scalar(),
            'time': cx.Scalar(),
        },
    }
    actual = xarray_utils.read_fields_from_xarray(self.mock_ds, input_specs)
    self.assert_data_and_specs_keys_match(actual, input_specs)

  def test_read_sharded_fields_from_xarray(self):
    """Tests that read_sharded_fields handles different shard sizes."""
    coords = cx.compose_coordinates(self.levels, self.grid)
    input_specs = {
        'era5': {
            'temperature': coords,
            'time': cx.Scalar(),
        },
        'era5:other': {
            '2m_temperature': self.grid,
            'global_scalar': cx.Scalar(),
            'time': cx.Scalar(),
        },
    }
    mesh_shape = collections.OrderedDict([('x', 2), ('y', 2)])
    with self.subTest('single_shard'):
      field_partition = {}
      actual = xarray_utils.read_sharded_fields_from_xarray(
          self.mock_ds, input_specs, mesh_shape, field_partition
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      coord_shard = coordinates.CoordinateShard(
          coords, mesh_shape, field_partition
      )
      expected_coord = cx.compose_coordinates(self.timedelta, coord_shard)
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64, 32))

    with self.subTest('two_longitude_shards'):
      field_partition = {'longitude': 'x'}
      actual = xarray_utils.read_sharded_fields_from_xarray(
          self.mock_ds.isel(longitude=slice(0, 32)),
          input_specs,
          mesh_shape,
          field_partition,
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      coord_shard = coordinates.CoordinateShard(
          coords, mesh_shape, field_partition
      )
      expected_coord = cx.compose_coordinates(self.timedelta, coord_shard)
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64 // 2, 32))

    with self.subTest('lon_lat_shards'):
      field_partition = {'longitude': 'x', 'latitude': 'y'}
      actual = xarray_utils.read_sharded_fields_from_xarray(
          self.mock_ds.isel(longitude=slice(0, 32), latitude=slice(0, 16)),
          input_specs,
          mesh_shape,
          field_partition,
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      coord_shard = coordinates.CoordinateShard(
          coords, mesh_shape, field_partition
      )
      expected_coord = cx.compose_coordinates(self.timedelta, coord_shard)
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64 // 2, 32 // 2))

    with self.subTest('four_longitude_shards'):
      field_partition = {'longitude': ('x', 'y')}
      actual = xarray_utils.read_sharded_fields_from_xarray(
          self.mock_ds.isel(longitude=slice(0, 16)),
          input_specs,
          mesh_shape,
          field_partition,
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      coord_shard = coordinates.CoordinateShard(
          coords, mesh_shape, field_partition
      )
      expected_coord = cx.compose_coordinates(self.timedelta, coord_shard)
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64 // 4, 32))


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
