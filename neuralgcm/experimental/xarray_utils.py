# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for converting between xarray and DataObservation objects."""

import collections
import functools
import operator
from typing import cast

from dinosaur import spherical_harmonic
from dinosaur import xarray_utils as dino_xarray_utils
import jax
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import parallelism
from neuralgcm.experimental import scales
from neuralgcm.experimental import typing
from neuralgcm.experimental import units
import neuralgcm.experimental.jax_datetime as jdt
import xarray


verify_grid_consistency = dino_xarray_utils.verify_grid_consistency


def xarray_nondimensionalize(
    ds: xarray.Dataset | xarray.DataArray,
    sim_units: units.SimUnits,
) -> xarray.Dataset | xarray.DataArray:
  return xarray.apply_ufunc(sim_units.nondimensionalize, ds)


# TODO(shoyer): drop this in favor of coordax.Field.from_xarray
def coordinates_from_dataset(
    ds: xarray.Dataset,
    spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
) -> coordinates.DinosaurCoordinates:
  """Infers coordinates from `ds`."""
  # TODO(dkochkov): Generalize this to take a role of composing coordinates
  # from a collection of candidate coordax.Coordinate instances.
  dino_coords = dino_xarray_utils.coordinate_system_from_dataset(
      ds,
      truncation=dino_xarray_utils.LINEAR,
      spherical_harmonics_impl=spherical_harmonics_impl,
  )
  return coordinates.DinosaurCoordinates.from_dinosaur_coords(dino_coords)


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


def xarray_to_fields_with_time(
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
  fields = {
      k: coordinates.field_from_xarray(v, additional_coord_types)
      for k, v in variables.items()
  }
  time = variables.pop('time', None)
  # TODO(dkochkov): consider breaking this into a separate function.
  if time is None:
    # Fall back to getting 'time' from coordinates
    time = ds.coords['time']
    timedelta = coordinates.TimeDelta.from_xarray(('timedelta',), ds.coords)
    fields['time'] = cx.wrap(jdt.to_datetime(time.data), timedelta)
  return fields


def read_fields_from_xarray(
    dataset: xarray.Dataset,
    input_specs: dict[str, dict[str, cx.Coordinate]],
    strict_matches: bool = True,
) -> dict[str, dict[str, cx.Field]]:
  """Returns a `specs`-like structure of coordax.Fields from a `dataset`.

  Args:
    dataset: xarray dataset to read data from.
    input_specs: nested dictionary that associates variable with coordinates.
    strict_matches: whether to require exact coordinate matches.

  Returns:
    A dictionary of dictionaries of coordax.Fields.
  """
  # TODO(dkochkov): Generalize this to work with xarray.DataTree and a
  # a dict of arrays to be compatible with the new xreader.
  requested_var_names = functools.reduce(
      operator.or_, [d.keys() for d in input_specs.values()], set()
  )
  if not all(k in dataset for k in requested_var_names):
    missing_vars = set(requested_var_names).difference(set(dataset))
    raise ValueError(f'specs contains {missing_vars=} that are not in dataset')
  dataset = dataset[requested_var_names]
  variables = cast(dict[str, xarray.DataArray], dict(dataset))
  is_coordinate = lambda x: isinstance(x, cx.Coordinate)
  input_spec_coords = jax.tree.leaves(input_specs, is_leaf=is_coordinate)
  additional_coord_types = [type(x) for x in input_spec_coords]
  if not strict_matches:
    # including labeled axis ensures that we can match coordinates that do not
    # support from_xarray reconstruction. LabelAxis will be retagged with the
    # coordinate specified in the specs.
    additional_coord_types += [cx.LabeledAxis]
  fields = {
      k: coordinates.field_from_xarray(v, tuple(additional_coord_types))
      for k, v in variables.items()
  }
  result = {}
  for observation_key, observation_specs in input_specs.items():
    result[observation_key] = {}
    for k, v in observation_specs.items():
      if strict_matches:
        result[observation_key][k] = fields[k].untag(v).tag(v)
      else:
        result[observation_key][k] = fields[k].untag(*v.dims).tag(v)
  return result


def read_sharded_fields_from_xarray(
    dataset: xarray.Dataset,
    input_specs: dict[str, dict[str, cx.Coordinate]],
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
    dataset: xarray dataset to read data from.
    input_specs: nested dictionary that associates variable with coordinates.
    mesh_shape: shape of the sharding mesh indicating number of devices in each
      axis.
    dim_partitions: mapping from dimension names to labels of device axes that
      the dimension is partitioned across.

  Returns:
    A dictionary of dictionaries of coordax.Fields tagged with CoordinateShard
    coordinates.
  """

  def wrap_coordinate_shard(coord: cx.Coordinate) -> cx.Coordinate:
    return coordinates.CoordinateShard(coord, mesh_shape, dim_partitions)

  is_coordinate = lambda x: isinstance(x, cx.Coordinate)
  shard_specs = jax.tree.map(
      wrap_coordinate_shard, input_specs, is_leaf=is_coordinate
  )
  return read_fields_from_xarray(dataset, shard_specs, strict_matches=False)


def fields_to_xarray(
    fields: dict[str, cx.Field],
) -> xarray.Dataset:
  """Converts a coordax.Field dictionary to an xarray dataset."""
  ds = xarray.Dataset({k: v.to_xarray() for k, v in fields.items()})
  return ds
