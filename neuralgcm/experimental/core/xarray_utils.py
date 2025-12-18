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

"""Utilities for converting between xarray and DataObservation objects."""

import collections
from typing import TypeAlias, TypeVar, cast

import coordax as cx
from dinosaur import xarray_utils as dino_xarray_utils
import jax
import jax_datetime as jdt
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import data_specs
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import scales
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import xarray


CoordSpec: TypeAlias = data_specs.CoordSpec
OptionalSpec: TypeAlias = data_specs.OptionalSpec

verify_grid_consistency = dino_xarray_utils.verify_grid_consistency


def xarray_nondimensionalize(
    ds: xarray.Dataset | xarray.DataArray,
    sim_units: units.SimUnits,
) -> xarray.Dataset | xarray.DataArray:
  return xarray.apply_ufunc(sim_units.nondimensionalize, ds)


def get_longitude_latitude_names(ds: xarray.Dataset) -> tuple[str, str]:
  """Infers names used for longitude and latitude in the dataset `ds`."""
  if 'lon' in ds.dims and 'lat' in ds.dims:
    return ('lon', 'lat')
  if 'longitude' in ds.dims and 'latitude' in ds.dims:
    return ('longitude', 'latitude')
  raise ValueError(f'No `lon/lat`|`longitude/latitude` in {ds.coords.keys()=}')


def attach_data_units(
    array: xarray.DataArray,
    default_units: typing.Quantity = typing.units.dimensionless,
) -> xarray.DataArray:
  """Attaches units to `array` based on `attrs.units` or `default_units`."""
  attrs = dict(array.attrs)
  unit = attrs.pop('units', None)
  if unit is not None:
    data = units.parse_units(unit) * array.data
  else:
    data = default_units * array.data
  return xarray.DataArray(data, array.coords, array.dims, attrs=attrs)


def nodal_orography_from_ds(ds: xarray.Dataset) -> xarray.DataArray:
  """Returns orography in nodal representation from `ds`."""
  orography_key = dino_xarray_utils.OROGRAPHY
  if orography_key not in ds:
    ds[orography_key] = (
        ds[dino_xarray_utils.GEOPOTENTIAL_AT_SURFACE_KEY]
        / scales.GRAVITY_ACCELERATION.magnitude
    )
  lon_lat_order = get_longitude_latitude_names(ds)
  orography = attach_data_units(ds[orography_key], typing.units.meter)
  return orography.transpose(*lon_lat_order)


def swap_time_to_timedelta(ds: xarray.Dataset) -> xarray.Dataset:
  """Converts an xarray dataset with a time axis to a timedelta axis."""
  ds = ds.assign_coords(timedelta=ds.time - ds.time[0])
  ds = ds.swap_dims({'time': 'timedelta'})
  return ds


DatasetOrNestedDataset = TypeVar(
    'DatasetOrNestedDataset', xarray.Dataset, dict[str, xarray.Dataset]
)


def ensure_timedelta_axis(ds: DatasetOrNestedDataset) -> DatasetOrNestedDataset:
  """Ensures an xarray dataset has a timedelta axis."""
  if not isinstance(ds, xarray.Dataset):
    return jax.tree.map(
        ensure_timedelta_axis,
        ds,
        is_leaf=lambda x: isinstance(x, xarray.Dataset),
    )
  if 'time' in ds.dims:
    ds = swap_time_to_timedelta(ds)
  if 'time' in ds.coords:
    ds = ds.reset_coords('time')
  return ds


def time_field_from_xarray_coords(
    coords: xarray.Coordinates,
) -> cx.Field:
  """Returns a coordax.Field object for the time coordinate in `coords`."""
  if 'time' not in coords:
    raise ValueError(f'No `time` in {coords.keys()=}')
  timedelta = coordinates.TimeDelta.from_xarray(('timedelta',), coords)
  time = coords['time']
  return cx.field(jdt.to_datetime(time.data), timedelta)


def xarray_to_fields(
    ds: xarray.Dataset,
    additional_coord_types: tuple[cx.Coordinate, ...] = (),
) -> dict[str, cx.Field]:
  """Converts an xarray dataset to a dictionary of coordax.Field objects.

  Args:
    ds: dataset to convert.
    additional_coord_types: additional coordinate types to use when inferring
      coordinates from the dataset.

  Returns:
    A dictionary mapping variable names to coordax fields.
  """
  variables = cast(dict[str, xarray.DataArray], dict(ds))
  return {
      k: field_from_xarray(v, additional_coord_types)
      for k, v in variables.items()
  }


def xarray_to_fields_with_time(
    ds: xarray.Dataset,
    additional_coord_types: tuple[cx.Coordinate, ...] = (),
) -> dict[str, cx.Field]:
  """Same as xarray_to_fields, but with required time in `ds` or `ds.coords`."""
  fields = xarray_to_fields(ds, additional_coord_types)
  if 'time' not in fields:
    time = time_field_from_xarray_coords(ds.coords)
    fields['time'] = time
  return fields


def field_from_xarray(
    data_array: xarray.DataArray,
    additional_coord_types: tuple[cx.Coordinate, ...] = (),
) -> cx.Field:
  """Converts an xarray.DataArray to a Field using NeuralGCM coordinates."""
  coord_types = (
      coordinates.TimeDelta,
      coordinates.LonLatGrid,
      coordinates.SphericalHarmonicGrid,
      coordinates.HybridLevels,
      coordinates.PressureLevels,
      coordinates.SigmaLevels,
      coordinates.SigmaBoundaries,
      coordinates.LayerLevels,
      cx.DummyAxis,
  )
  return cx.Field.from_xarray(data_array, coord_types + additional_coord_types)


def validate_xarray_inputs(
    inputs: dict[str, xarray.Dataset],
    in_spec: dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]],
):
  """Validates that `inputs` from xarray satisfy expectations of `in_spec`."""
  coords = {}
  for data_key, dataset in inputs.items():
    if data_key not in in_spec:
      continue
    coords[data_key] = {}
    dataset_spec = in_spec[data_key]
    for k, var_spec in dataset_spec.items():
      if k in dataset:
        spec, _ = data_specs.unwrap_optional(var_spec)
        c_types = data_specs.get_coord_types(spec)
        coords[data_key][k] = cx.coords.from_xarray(dataset[k], c_types)
  data_specs.validate_inputs(coords, in_spec)


def read_from_xarray(
    nested_data: dict[str, xarray.Dataset],
    in_spec: dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]],
    strict_matches: bool = True,
) -> typing.InputFields:
  """Reads `nested_data` from xarray according to `in_spec`."""

  def _get_field(da: xarray.DataArray, spec: cx.Coordinate | CoordSpec):
    coord_types = data_specs.get_coord_types(spec)
    if not strict_matches:
      coord_types += (cx.LabeledAxis,)
    return field_from_xarray(da, coord_types)

  result = {}
  for data_key, data_spec in in_spec.items():
    if data_key not in nested_data:
      raise ValueError(
          f'Missing dataset for source {data_key!r} in '
          f'nested_data. Available keys: {list(nested_data.keys())}'
      )
    dataset = nested_data[data_key]
    specs, fields, missing_vars = {}, {}, []
    for k, v in data_spec.items():
      spec, is_optional = data_specs.unwrap_optional(v)
      if k in dataset:
        specs[k] = spec if isinstance(spec, CoordSpec) else CoordSpec(spec)
        fields[k] = _get_field(dataset[k], spec)
      elif not is_optional:
        missing_vars.append(k)

    if missing_vars:
      raise ValueError(
          f'Specs for {data_key!r} contains {missing_vars=} that are '
          f'not in the corresponding dataset with keys {list(dataset.keys())}'
      )

    result[data_key] = {}
    for k, v in fields.items():
      spec = specs[k]
      target_coord = data_specs.finalize_spec(spec, v.coordinate)
      if not set(target_coord.dims).issubset(set(v.coordinate.dims)):
        raise ValueError(
            f'Dimensions for coordinate {target_coord} for {data_key}.{k} are '
            f'not in {v.coordinate}'
        )
      if strict_matches:
        v = v.untag(target_coord).tag(target_coord)
      else:
        v = v.untag(*target_coord.dims).tag(target_coord)
      result[data_key][k] = v
  return result


def read_sharded_from_xarray(
    nested_data: dict[str, xarray.Dataset],
    in_spec: dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]],
    mesh_shape: collections.OrderedDict[str, int],
    dim_partitions: parallelism.DimPartitions,
) -> dict[str, dict[str, cx.Field]]:
  """Returns a `specs`-like structure of coordax.Fields from a `dataset` shard.

  This is a helpful function for annotating coordax.Field with full coordinates
  while reading shards of dataset in a distributed setting. By providing the
  mesh shape and how different dimensions are partitioned we can include full
  coordinate information by tagging the data with CoordinateShard objects. This
  can later be dropped once the data is converted to jax arrays and sharded
  across devices.

  Args:
    nested_data: dict of xarray datasets from which to read data.
    in_spec: nested dictionary that associates variables with coordinates.
    mesh_shape: shape of the sharding mesh indicating number of devices in each
      axis.
    dim_partitions: mapping from dimension names to labels of device axes that
      the dimension is partitioned across.

  Returns:
    A dictionary of dictionaries of coordax.Fields tagged with CoordinateShard
    coordinates.
  """

  def _wrap_axis(ax: cx.Coordinate) -> cx.Coordinate:
    return parallelism.CoordinateShard(ax, mesh_shape, dim_partitions)

  def wrap_coordinate_shard(coord_or_spec):
    if isinstance(coord_or_spec, data_specs.OptionalSpec):
      return data_specs.OptionalSpec(wrap_coordinate_shard(coord_or_spec.spec))
    elif isinstance(coord_or_spec, data_specs.CoordSpec):
      coord = coord_or_spec.coord
      coord = cx.coords.compose(*[_wrap_axis(ax) for ax in coord.axes])
      exact = data_specs.AxisMatchRules.EXACT
      replaced = data_specs.AxisMatchRules.REPLACED
      replace_dict = {exact: replaced}
      exact_to_replaced = lambda v: replace_dict.get(v, v)
      new_dim_match_rules = {
          dim: exact_to_replaced(coord_or_spec.dim_match_rules.get(dim, exact))
          for dim in coord_or_spec.coord.dims
      }
      return data_specs.CoordSpec(coord, new_dim_match_rules)
    elif isinstance(coord_or_spec, cx.Coordinate):
      return wrap_coordinate_shard(data_specs.CoordSpec(coord_or_spec))
    else:
      raise ValueError(f'Unsupported coord or spec type: {coord_or_spec}')

  if_leaf = lambda x: not isinstance(x, dict)  # in_spec must be a nested dict.
  shard_specs = jax.tree.map(wrap_coordinate_shard, in_spec, is_leaf=if_leaf)
  return read_from_xarray(nested_data, shard_specs, strict_matches=False)


def fields_to_xarray(
    fields: dict[str, cx.Field],
) -> xarray.Dataset:
  """Converts a coordax.Field dictionary to an xarray dataset."""
  ds = xarray.Dataset({k: v.to_xarray() for k, v in fields.items()})
  return ds


def nested_fields_to_xarray(
    nested_fields: dict[str, dict[str, cx.Field]],
) -> dict[str, xarray.Dataset]:
  """Converts a two level nested coordax.Field dictionary to xarray."""
  # TODO(dkochkov): Consider switching to xarray.DataTree.
  return {k: fields_to_xarray(v) for k, v in nested_fields.items()}


def model_inputs_from_xarray(
    nested_data: dict[str, xarray.Dataset],
    model: api.InferenceModel | api.Model,
) -> dict[str, dict[str, cx.Field]]:
  """Returns subset of `nested_data` supported by the model for assimilation."""
  nested_data = ensure_timedelta_axis(nested_data)
  return read_from_xarray(
      nested_data=nested_data,
      in_spec=model.inputs_spec,
      strict_matches=False,
  )


def model_dynamic_inputs_from_xarray(
    nested_data: dict[str, xarray.Dataset],
    model: api.InferenceModel | api.Model,
) -> dict[str, dict[str, cx.Field]]:
  """Returns subset of `nested_data` required by the model."""
  nested_data = ensure_timedelta_axis(nested_data)
  return read_from_xarray(
      nested_data=nested_data,
      in_spec=model.dynamic_inputs_spec,
      strict_matches=False,
  )
