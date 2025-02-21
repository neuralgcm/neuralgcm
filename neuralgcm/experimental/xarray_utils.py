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

from typing import cast

from dinosaur import spherical_harmonic
from dinosaur import xarray_utils as dino_xarray_utils
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import coordinates
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


def fields_to_xarray(
    fields: dict[str, cx.Field],
) -> xarray.Dataset:
  """Converts a coordax.Field dictionary to an xarray dataset."""
  ds = xarray.Dataset({k: v.to_xarray() for k, v in fields.items()})
  return ds
