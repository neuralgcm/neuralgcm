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
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import jax
import jax_datetime as jdt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import data_specs
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import xarray_utils
import numpy as np
import xarray


def _maybe_isel(ds: xarray.Dataset, **indexers: slice):
  if any(k in ds.dims for k in indexers):
    return ds.isel(**indexers)
  else:
    return ds


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class CustomCoord(cx.Coordinate):
  """Dummy custom coordinate class to test utilities against."""

  sz: int

  @property
  def dims(self):
    return ('custom_dim',)

  @property
  def shape(self) -> tuple[int, ...]:
    return (self.sz,)

  @property
  def fields(self):
    return {
        'pi': cx.wrap((np.pi * np.ones(self.sz)) ** np.arange(self.sz), self)
    }

  @classmethod
  def from_xarray(cls, dims: tuple[str, ...], coords: xarray.Coordinates):
    dim = dims[0]
    return cls(sz=coords[dim].size)


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
    other_fields = {
        'global_scalar': cx.wrap(np.linspace(0, np.pi, 3), timedelta)
    }

    t0 = np.datetime64('2024-01-01')
    time = cx.wrap(jdt.to_datetime(t0 + timedelta_values), timedelta)
    volume_fields['time'] = time
    surface_fields['time'] = time
    other_fields['time'] = time

    volume_vars = {k: v.to_xarray() for k, v in volume_fields.items()}
    surface_vars = {k: v.to_xarray() for k, v in surface_fields.items()}
    other_vars = {k: v.to_xarray() for k, v in other_fields.items()}

    self.mock_data = {
        'era5': xarray.Dataset(volume_vars),
        'era5:surface': xarray.Dataset(surface_vars),
        'era5:other': xarray.Dataset(other_vars),
    }
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
            'time': cx.Scalar(),
        },
        'era5:surface': {
            '2m_temperature': self.grid,
            'sst': self.grid,
            'time': cx.Scalar(),
        },
        'era5:other': {
            'global_scalar': cx.Scalar(),
            'time': cx.Scalar(),
        },
    }
    actual = xarray_utils.read_fields_from_xarray(self.mock_data, input_specs)
    self.assert_data_and_specs_keys_match(actual, input_specs)

  def test_read_fields_from_xarray_new_coord_type(self):
    """Tests that read_fields_from_xarray works with new coordinate types."""
    custom_axis = CustomCoord(sz=3)
    coords = cx.compose_coordinates(custom_axis, self.grid)
    ones_like = lambda coord: cx.wrap(np.ones(coord.shape), coord)
    a_b_vars = {'a': ones_like(coords), 'b': ones_like(custom_axis)}
    c_d_vars = {'c': ones_like(cx.Scalar()), 'd': ones_like(custom_axis)}
    e_f_vars = {'e': ones_like(coords), 'f': ones_like(cx.Scalar())}
    mock_data = {
        'ab': xarray.Dataset({k: v.to_xarray() for k, v in a_b_vars.items()}),
        'cd': xarray.Dataset({k: v.to_xarray() for k, v in c_d_vars.items()}),
        'ef': xarray.Dataset({k: v.to_xarray() for k, v in e_f_vars.items()}),
    }
    input_specs = {
        'ab': {'a': coords, 'b': CustomCoord(sz=3)},
        'cd': {'c': cx.Scalar(), 'd': CustomCoord(sz=3)},
        'ef': {'e': coords, 'f': cx.Scalar()},
    }
    actual = xarray_utils.read_fields_from_xarray(mock_data, input_specs)
    self.assert_data_and_specs_keys_match(actual, input_specs)

  def test_read_sharded_fields_from_xarray(self):
    """Tests that read_sharded_fields handles different shard sizes."""
    coords = cx.compose_coordinates(self.levels, self.grid)
    input_specs = {
        'era5': {
            'temperature': coords,
            'time': cx.Scalar(),
        },
        'era5:surface': {
            '2m_temperature': self.grid,
            'time': cx.Scalar(),
        },
        'era5:other': {
            'global_scalar': cx.Scalar(),
            'time': cx.Scalar(),
        },
    }
    mesh_shape = collections.OrderedDict([('x', 2), ('y', 2)])
    with self.subTest('single_shard'):
      field_partition = {}
      actual = xarray_utils.read_sharded_fields_from_xarray(
          self.mock_data, input_specs, mesh_shape, field_partition
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      coord_shard = parallelism.CoordinateShard(
          coords, mesh_shape, field_partition
      )
      expected_coord = cx.compose_coordinates(self.timedelta, coord_shard)
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64, 32))

    with self.subTest('two_longitude_shards'):
      field_partition = {'longitude': 'x'}
      actual = xarray_utils.read_sharded_fields_from_xarray(
          {
              k: _maybe_isel(v, longitude=slice(0, 32))
              for k, v in self.mock_data.items()
          },
          input_specs,
          mesh_shape,
          field_partition,
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      coord_shard = parallelism.CoordinateShard(
          coords, mesh_shape, field_partition
      )
      expected_coord = cx.compose_coordinates(self.timedelta, coord_shard)
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64 // 2, 32))

    with self.subTest('lon_lat_shards'):
      field_partition = {'longitude': 'x', 'latitude': 'y'}
      actual = xarray_utils.read_sharded_fields_from_xarray(
          {
              k: _maybe_isel(v, longitude=slice(0, 32), latitude=slice(0, 16))
              for k, v in self.mock_data.items()
          },
          input_specs,
          mesh_shape,
          field_partition,
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      coord_shard = parallelism.CoordinateShard(
          coords, mesh_shape, field_partition
      )
      expected_coord = cx.compose_coordinates(self.timedelta, coord_shard)
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64 // 2, 32 // 2))

    with self.subTest('four_longitude_shards'):
      field_partition = {'longitude': ('x', 'y')}
      actual = xarray_utils.read_sharded_fields_from_xarray(
          {
              k: _maybe_isel(v, longitude=slice(0, 16))
              for k, v in self.mock_data.items()
          },
          input_specs,
          mesh_shape,
          field_partition,
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      coord_shard = parallelism.CoordinateShard(
          coords, mesh_shape, field_partition
      )
      expected_coord = cx.compose_coordinates(self.timedelta, coord_shard)
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64 // 4, 32))


class ReadFromXarrayTest(parameterized.TestCase):

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
    other_fields = {
        'global_scalar': cx.wrap(np.linspace(0, np.pi, 3), timedelta)
    }

    t0 = np.datetime64('2024-01-01')
    time = cx.wrap(jdt.to_datetime(t0 + timedelta_values), timedelta)
    volume_fields['time'] = time
    surface_fields['time'] = time
    other_fields['time'] = time

    volume_vars = {k: v.to_xarray() for k, v in volume_fields.items()}
    surface_vars = {k: v.to_xarray() for k, v in surface_fields.items()}
    other_vars = {k: v.to_xarray() for k, v in other_fields.items()}

    self.mock_data = {
        'era5': xarray.Dataset(volume_vars),
        'era5:surface': xarray.Dataset(surface_vars),
        'era5:other': xarray.Dataset(other_vars),
    }
    self.grid = grid
    self.levels = levels
    self.timedelta = timedelta

  def assert_data_and_specs_keys_match(
      self,
      actual: dict[str, dict[str, cx.Field]],
      specs: dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]],
  ):
    """Tests that actual data and specs have matching keys."""
    self.assertSameElements(actual.keys(), specs.keys())
    for k in actual.keys():
      self.assertSameElements(actual[k].keys(), specs[k].keys())

  def test_read_from_xarray_with_basic_inputs_spec(self):
    t = coordinates.TimeDelta(np.arange(3) * np.timedelta64(1, 'h'))
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    tx = cx.compose_coordinates(t, x)
    rng = np.random.RandomState(42)
    fields = {'u': cx.wrap(rng.randn(*tx.shape), tx)}
    nested_data = xarray_utils.nested_fields_to_xarray({'data': fields})
    inputs_spec = {'data': {'u': tx}}

    read_data = xarray_utils.read_from_xarray(nested_data, inputs_spec)
    cx.testing.assert_fields_equal(read_data['data']['u'], fields['u'])

  def test_read_from_xarray_with_optional_spec(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    y = cx.LabeledAxis('y', np.linspace(0, np.e, num=5))
    rng = np.random.RandomState(42)
    fields = {'u': cx.wrap(rng.randn(*x.shape), x)}
    nested_data = xarray_utils.nested_fields_to_xarray({'data': fields})
    inputs_spec = {
        'data': {
            'u': x,
            'v': data_specs.OptionalSpec(spec=y),
        }
    }
    read_data = xarray_utils.read_from_xarray(nested_data, inputs_spec)
    cx.testing.assert_fields_equal(read_data['data']['u'], fields['u'])
    self.assertNotIn('v', read_data['data'])  # OptionalSpec is not present.

  def test_read_from_xarray_with_missing_non_optional_variable_raises(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    y = cx.LabeledAxis('y', np.linspace(0, np.e, num=5))
    rng = np.random.RandomState(42)
    fields = {'u': cx.wrap(rng.randn(*x.shape), x)}
    nested_data = xarray_utils.nested_fields_to_xarray({'data': fields})
    inputs_spec = {'data': {'u': x, 'v': y}}
    with self.assertRaisesRegex(
        ValueError, "Specs for 'data' contains missing_vars=\\['v'\\]"
    ):
      xarray_utils.read_from_xarray(nested_data, inputs_spec)

  def test_read_from_xarray_with_incompatible_coordinate_raises(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    y = cx.LabeledAxis('y', np.linspace(0, np.e, num=5))
    rng = np.random.RandomState(42)
    fields = {'u': cx.wrap(rng.randn(*x.shape), x)}
    nested_data = xarray_utils.nested_fields_to_xarray({'data': fields})
    inputs_spec = {'data': {'u': y}}
    with self.assertRaisesRegex(ValueError, '.* have different dims'):
      xarray_utils.read_from_xarray(nested_data, inputs_spec)

  def test_read_from_xarray_new_coord_type(self):
    """Tests that read_fields_from_xarray works with new coordinate types."""
    custom_axis = CustomCoord(sz=3)
    coords = cx.compose_coordinates(custom_axis, self.grid)
    ones_like = lambda coord: cx.wrap(np.ones(coord.shape), coord)
    a_b_vars = {'a': ones_like(coords), 'b': ones_like(custom_axis)}
    c_d_vars = {'c': ones_like(cx.Scalar()), 'd': ones_like(custom_axis)}
    e_f_vars = {'e': ones_like(coords), 'f': ones_like(cx.Scalar())}
    mock_data = {
        'ab': xarray.Dataset({k: v.to_xarray() for k, v in a_b_vars.items()}),
        'cd': xarray.Dataset({k: v.to_xarray() for k, v in c_d_vars.items()}),
        'ef': xarray.Dataset({k: v.to_xarray() for k, v in e_f_vars.items()}),
    }
    input_specs = {
        'ab': {'a': coords, 'b': CustomCoord(sz=3)},
        'cd': {'c': cx.Scalar(), 'd': data_specs.CoordSpec(CustomCoord(sz=3))},
        'ef': {'e': coords, 'f': cx.Scalar()},
    }
    actual = xarray_utils.read_from_xarray(mock_data, input_specs)
    self.assert_data_and_specs_keys_match(actual, input_specs)

  def test_read_sharded_from_xarray(self):
    """Tests that read_sharded_from_xarray handles different shard sizes."""
    coords = cx.compose_coordinates(self.levels, self.grid)
    input_specs = {
        'era5': {
            'temperature': coords,
            'time': cx.Scalar(),
        },
        'era5:surface': {
            '2m_temperature': self.grid,
            'time': cx.Scalar(),
        },
        'era5:other': {
            'global_scalar': cx.Scalar(),
            'time': cx.Scalar(),
        },
    }
    input_specs = jax.tree.map(
        data_specs.CoordSpec.with_any_timedelta,
        input_specs,
        is_leaf=lambda x: not isinstance(x, dict),
    )
    mesh_shape = collections.OrderedDict([('x', 2), ('y', 2)])

    def _with_shard_axes(coord, mesh_shape, partition):
      axes = [
          parallelism.CoordinateShard(ax, mesh_shape, partition)
          for ax in coord.axes
      ]
      return cx.compose_coordinates(*axes)

    with self.subTest('single_shard'):
      field_partition = {}
      actual = xarray_utils.read_sharded_from_xarray(
          self.mock_data, input_specs, mesh_shape, field_partition
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      expected_coord = _with_shard_axes(
          cx.compose_coordinates(self.timedelta, coords),
          mesh_shape,
          field_partition,
      )
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64, 32))

    with self.subTest('two_longitude_shards'):
      field_partition = {'longitude': 'x'}
      actual = xarray_utils.read_sharded_from_xarray(
          {
              k: _maybe_isel(v, longitude=slice(0, 32))
              for k, v in self.mock_data.items()
          },
          input_specs,
          mesh_shape,
          field_partition,
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      expected_coord = _with_shard_axes(
          cx.compose_coordinates(self.timedelta, coords),
          mesh_shape,
          field_partition,
      )
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64 // 2, 32))

    with self.subTest('lon_lat_shards'):
      field_partition = {'longitude': 'x', 'latitude': 'y'}
      actual = xarray_utils.read_sharded_from_xarray(
          {
              k: _maybe_isel(v, longitude=slice(0, 32), latitude=slice(0, 16))
              for k, v in self.mock_data.items()
          },
          input_specs,
          mesh_shape,
          field_partition,
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      expected_coord = _with_shard_axes(
          cx.compose_coordinates(self.timedelta, coords),
          mesh_shape,
          field_partition,
      )
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64 // 2, 32 // 2))

    with self.subTest('four_longitude_shards'):
      field_partition = {'longitude': ('x', 'y')}
      actual = xarray_utils.read_sharded_from_xarray(
          {
              k: _maybe_isel(v, longitude=slice(0, 16))
              for k, v in self.mock_data.items()
          },
          input_specs,
          mesh_shape,
          field_partition,
      )
      self.assert_data_and_specs_keys_match(actual, input_specs)
      expected_coord = _with_shard_axes(
          cx.compose_coordinates(self.timedelta, coords),
          mesh_shape,
          field_partition,
      )
      actual_coord = cx.get_coordinate(actual['era5']['temperature'])
      self.assertEqual(actual_coord, expected_coord)
      self.assertEqual(actual_coord.shape, (3, 37, 64 // 4, 32))


class ValidateXarrayInputsTest(parameterized.TestCase):

  def test_validate_xarray_inputs_with_coordinate(self):
    t = coordinates.TimeDelta(np.arange(3) * np.timedelta64(1, 'h'))
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    y = cx.LabeledAxis('y', np.linspace(0, np.e, num=5))
    tx = cx.compose_coordinates(t, x)
    ty = cx.compose_coordinates(t, y)
    rng = np.random.RandomState(42)
    fields = {
        'u': cx.wrap(rng.randn(*tx.shape), tx),
        'v': cx.wrap(rng.randn(*ty.shape), ty),
    }
    inputs = xarray_utils.nested_fields_to_xarray({'data_key': fields})

    inputs_spec = {'data_key': {'u': tx, 'v': ty}}
    xarray_utils.validate_xarray_inputs(inputs, inputs_spec)

  def test_validate_xarray_inputs_with_incompatible_coordinate_raises(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    wrong_x = cx.LabeledAxis('x', np.linspace(0, np.e, num=4))
    rng = np.random.RandomState(42)
    fields = {'u': cx.wrap(rng.randn(*x.shape), x)}
    inputs = xarray_utils.nested_fields_to_xarray({'data_key': fields})

    inputs_spec = {'data_key': {'u': wrong_x}}
    with self.assertRaisesRegex(
        ValueError, 'Coordinate axis .* for dimension x does not match'
    ):
      xarray_utils.validate_xarray_inputs(inputs, inputs_spec)

  def test_validate_xarray_inputs_with_missing_dataset_raises(self):
    x = cx.LabeledAxis('x', np.linspace(0, np.pi, num=4))
    inputs = {}  # Empty inputs.
    inputs_spec = {'data_key': {'u': x}}
    with self.assertRaisesRegex(ValueError, 'Data key data_key is missing'):
      xarray_utils.validate_xarray_inputs(inputs, inputs_spec)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
