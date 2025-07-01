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

import coordax as cx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import typing


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
