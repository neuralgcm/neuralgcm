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

"""Modules that interpolate data from one coordinate system to another."""

import dataclasses
from typing import Protocol, overload

import coordax as cx
from dinosaur import horizontal_interpolation
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import typing


class BaseRegridder(Protocol):
  """Base class for regridders."""

  @overload
  def __call__(self, inputs: cx.Field) -> cx.Field:
    ...

  @overload
  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    ...

  def __call__(
      self, inputs: cx.Field | dict[str, cx.Field]
  ) -> cx.Field | dict[str, cx.Field]:
    ...


@dataclasses.dataclass(frozen=True)
class SpectralRegridder:
  """Regrid between spherical harmonic grids with truncation or zero padding."""

  target_coords: coordinates.SphericalHarmonicGrid

  def truncate_to_target_wavenumbers(
      self,
      field: cx.Field,
  ) -> cx.Field:
    """Interpolates to lower resolution spherical harmonic by truncation."""
    # TODO(dkochkov) consider using coordinate values to inform truncation.
    target_lon_wavenumbers = self.target_coords.sizes['longitude_wavenumber']
    target_total_wavenumbers = self.target_coords.sizes['total_wavenumber']
    lon_slice = slice(0, target_lon_wavenumbers)
    total_slice = slice(0, target_total_wavenumbers)
    field = field.untag('longitude_wavenumber', 'total_wavenumber')
    result = cx.cmap(lambda x: x[lon_slice, total_slice], field.named_axes)(
        field
    )
    return result.tag(self.target_coords)

  def pad_to_target_wavenumbers(
      self,
      field: cx.Field,
  ) -> cx.Field:
    """Interpolates to higher resolution spherical harmonic by zero-padding."""
    # TODO(dkochkov) use `sizes` on coords to carry shape of dims info.
    input_lon_k = field.coord_fields['longitude_wavenumber'].shape[0]
    input_total_k = field.coord_fields['total_wavenumber'].shape[0]
    target_lon_k = self.target_coords.sizes['longitude_wavenumber']
    target_total_k = self.target_coords.sizes['total_wavenumber']
    pad_lon = (0, target_lon_k - input_lon_k)
    pad_total = (0, target_total_k - input_total_k)
    pad_fn = lambda x: jnp.pad(x, pad_width=(pad_lon, pad_total))
    field = field.untag('longitude_wavenumber', 'total_wavenumber')
    result = cx.cmap(pad_fn, field.named_axes)(field)
    return result.tag(self.target_coords)

  def interpolate_field(self, field: cx.Field) -> cx.Field:
    """Interpolates a single field."""
    # TODO(dkochkov) Check that inputs.coords includes SphericalHarmonicGrid.
    input_lon_k = field.coord_fields['longitude_wavenumber'].shape[0]
    input_total_k = field.coord_fields['total_wavenumber'].shape[0]
    target_lon_k = self.target_coords.sizes['longitude_wavenumber']
    target_total_k = self.target_coords.sizes['total_wavenumber']
    if (input_total_k < target_total_k) and (input_lon_k <= target_lon_k):
      return self.pad_to_target_wavenumbers(field)
    elif (input_total_k >= target_total_k) and (input_lon_k >= target_lon_k):
      return self.truncate_to_target_wavenumbers(field)
    else:
      raise ValueError(
          'Incompatible horizontal coordinates with shapes '
          f'{field.dims=} with {field.shape=}, '
          f'{self.target_coords.dims=} with {self.target_coords.shape=}, '
      )

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    """Interpolates fields in `inputs`."""
    is_field = lambda x: isinstance(x, cx.Field)
    return jax.tree.map(
        self.interpolate_field,
        inputs,
        is_leaf=is_field,
    )


@dataclasses.dataclass
class ConservativeRegridder:
  """Regrids fields between two LonLatGrids using conservative interpolation."""

  target_grid: coordinates.LonLatGrid
  skipna: bool = False

  def lat_weights(self, source_grid: coordinates.LonLatGrid):
    """Conservative regridding weights for the latitude dimension."""
    s_lat = jnp.deg2rad(source_grid.fields['latitude'].data)
    t_lat = jnp.deg2rad(self.target_grid.fields['latitude'].data)
    return horizontal_interpolation.conservative_latitude_weights(s_lat, t_lat)

  def lon_weights(self, source_grid: coordinates.LonLatGrid):
    """Conservative regridding weights for the longitude dimension."""
    s_lon = jnp.deg2rad(source_grid.fields['longitude'].data)
    t_lon = jnp.deg2rad(self.target_grid.fields['longitude'].data)
    return horizontal_interpolation.conservative_longitude_weights(s_lon, t_lon)

  def _regrid_2d(
      self, array: jax.Array, source_grid: coordinates.LonLatGrid
  ) -> jax.Array:
    """Applies conservative regridding to a 2D array."""

    def _mean(data: jax.Array) -> jax.Array:
      return jnp.einsum(
          'ab,cd,bd->ac',
          self.lon_weights(source_grid),
          self.lat_weights(source_grid),
          data,
          precision='float32',
      )

    not_nulls = jnp.logical_not(jnp.isnan(array))
    mean = _mean(jnp.where(not_nulls, array, 0))
    not_null_fraction = _mean(not_nulls)

    if self.skipna:
      return mean / not_null_fraction  # intended NaN if not_null_fraction == 0
    else:
      # If not_null_fraction is not close to 1, it means some source cells
      # were NaN. In this case, the target cell becomes NaN.
      return jnp.where(
          jnp.isclose(not_null_fraction, 1.0, rtol=1e-3),
          mean / not_null_fraction,
          jnp.nan,
      )

  @overload
  def __call__(self, inputs: cx.Field) -> cx.Field:
    ...

  @overload
  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    ...

  def __call__(
      self, inputs: cx.Field | dict[str, cx.Field]
  ) -> cx.Field | dict[str, cx.Field]:
    """Regrids a dictionary of fields to the target grid."""
    if isinstance(inputs, dict):
      return {k: self.__call__(v) for k, v in inputs.items()}

    x = inputs
    lon_lat_dims = ('longitude', 'latitude')
    source_grid = cx.coords.compose(*[x.axes.get(d) for d in lon_lat_dims])
    x = inputs.untag(source_grid)
    regrid_fn = cx.cmap(self._regrid_2d, x.named_axes)
    x = regrid_fn(x, source_grid)
    return x.tag(self.target_grid)
