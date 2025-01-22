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
"""Functionality for converting between coordax.Field and Xarray objects."""
from collections import abc
import typing
from typing import Callable, Type

from neuralgcm.experimental.coordax import core
import xarray


def field_to_data_array(field: core.Field) -> xarray.DataArray:
  """Converts a coordax.Field to an xarray.DataArray.

  Args:
    field: coordax.Field to convert into an xarray.DataArray.

  Returns:
    An xarray.DataArray object with the same data as the input coordax.Field.
    This DataArray will still be wrapping a jax.Array and have operations
    implemented on jax.Array objects using the Python Array API interface.
  """
  if not all(isinstance(dim, str) for dim in field.dims):
    raise ValueError(
        'can only convert Field objects with fully named dimensions to Xarray '
        f'objects, got dimensions {field.dims!r}'
    )

  coords = {}
  field_dims = set(field.dims)
  for coord in field.coords.values():
    for name, coord_field in coord.fields.items():
      if set(coord_field.dims) <= field_dims:
        # xarray.DataArray coordinate dimensions must be a subset of the
        # dimensions of the associated DataArray, which is not necessarily a
        # constraint for coordax.Field.
        variable = xarray.Variable(coord_field.dims, coord_field.data)
        if name in coords and not variable.identical(coords[name]):
          raise ValueError(
              f'inconsistent coordinate fields for {name!r}:\n'
              f'{variable}\nvs\n{coords[name]}'
          )
        coords[name] = variable

  return xarray.DataArray(data=field.data, dims=field.dims, coords=coords)


DEFAULT_COORD_TYPES = (core.LabeledAxis, core.NamedAxis)


def data_array_to_field(
    data_array: xarray.DataArray,
    coord_types: abc.Sequence[Type[core.Coordinate]] = DEFAULT_COORD_TYPES,
) -> core.Field:
  """Converts an xarray.DataArray to a coordax.Field.

  Args:
    data_array: xarray.DataArray to convert into a Field.
    coord_types: sequence of coordax.Coordinate subclasses with
      `maybe_from_xarray` methods defined. The first coordinate class that
      returns a coordinate object (indicating a match) will be used. By default,
      coordinates are constructed out of generic coordax.LabeledAxis and
      coordax.NamedAxis objects.

  Returns:
    A coordax.Field object with the same data as the input xarray.DataArray.
  """
  field = core.wrap(data_array.data)
  dims = data_array.dims
  coords = []

  if not all(isinstance(dim, str) for dim in dims):
    raise TypeError(
        'can only convert DataArray objects with string dimensions to Field'
    )
  dims = typing.cast(tuple[str, ...], dims)

  if not coord_types:
    raise ValueError('coord_types must be non-empty')

  def get_next_match():
    reasons = []
    for coord_type in coord_types:
      try:
        return coord_type.from_xarray(dims, data_array.coords)
      except Exception as e:  # pylint: disable=broad-exception-caught
        coord_name = coord_type.__module__ + '.' + coord_type.__name__
        reasons.append(f'{coord_name}: {e}')

    reasons_str = '\n'.join(reasons)
    raise ValueError(
        'failed to convert xarray.DataArray to coordax.Field, because no '
        f'coordinate type matched the dimensions starting with {dims}:\n'
        f'{data_array}\n\n'
        f'Reasons why coordinate matching failed:\n{reasons_str}'
    )

  while dims:
    coord = get_next_match()
    coords.append(coord)
    assert coord.ndim > 0  # dimensions will shrink by at least one
    dims = dims[coord.ndim :]

  return field.tag(*coords)
