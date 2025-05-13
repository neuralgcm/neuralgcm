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
"""Defines objects that transform between nodal and modal grids."""

import dataclasses
from typing import overload

from dinosaur import spherical_harmonic
import jax
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import typing


# TODO(dkochkov): Consider dropping nodal_grid and modal_grid properties and
# instead have them as arguments when grids contain explicit padding details.


@dataclasses.dataclass(frozen=True)
class SphericalHarmonicsTransform:
  """Spherical harmonic transform specified by grids and parallelism mesh."""

  lon_lat_grid: coordinates.LonLatGrid
  ylm_grid: coordinates.SphericalHarmonicGrid
  mesh: parallelism.Mesh
  partition_schema_key: parallelism.Schema | None
  level_key: str = 'level'
  longitude_key: str = 'longitude'
  latitude_key: str = 'latitude'
  radius: float = 1.0

  @property
  def dinosaur_spmd_mesh(self) -> jax.sharding.Mesh | None:
    """Returns the SPMD mesh transformed to Dinosaur convention."""
    dims_to_axes = {
        self.level_key: 'z',
        self.longitude_key: 'x',
        self.latitude_key: 'y',
    }
    return self.mesh.rearrange_spmd_mesh(
        self.partition_schema_key, dims_to_axes
    )

  @property
  def dinosaur_grid(self) -> spherical_harmonic.Grid:
    method = coordinates.SPHERICAL_HARMONICS_METHODS[
        self.ylm_grid.spherical_harmonics_method
    ]
    return spherical_harmonic.Grid(
        longitude_wavenumbers=self.ylm_grid.longitude_wavenumbers,
        total_wavenumbers=self.ylm_grid.total_wavenumbers,
        longitude_nodes=self.lon_lat_grid.longitude_nodes,
        latitude_nodes=self.lon_lat_grid.latitude_nodes,
        longitude_offset=self.lon_lat_grid.longitude_offset,
        latitude_spacing=self.lon_lat_grid.latitude_spacing,
        radius=self.radius,
        spherical_harmonics_impl=method,
        spmd_mesh=self.dinosaur_spmd_mesh,
    )

  @property
  def nodal_grid(self) -> coordinates.LonLatGrid:
    return coordinates.LonLatGrid.from_dinosaur_grid(self.dinosaur_grid)

  @property
  def modal_grid(self) -> coordinates.SphericalHarmonicGrid:
    return coordinates.SphericalHarmonicGrid.from_dinosaur_grid(
        self.dinosaur_grid
    )

  def to_modal_array(self, x: typing.Array) -> typing.Array:
    """Converts a nodal array to a modal array."""
    return self.dinosaur_grid.to_modal(x)

  def to_nodal_array(self, x: typing.Array) -> typing.Array:
    """Converts a modal array to a nodal array."""
    return self.dinosaur_grid.to_nodal(x)

  @overload
  def to_modal(self, x: cx.Field) -> cx.Field:
    ...

  @overload
  def to_modal(self, x: dict[str, cx.Field]) -> dict[str, cx.Field]:
    ...

  def to_modal(
      self, x: cx.Field | dict[str, cx.Field]
  ) -> cx.Field | dict[str, cx.Field]:
    """Converts a nodal field(s) to a modal field(s)."""
    if isinstance(x, dict):
      return {k: self.to_modal(v) for k, v in x.items()}
    modal = cx.cmap(self.to_modal_array)(x.untag(self.nodal_grid))
    return modal.tag(self.modal_grid)

  @overload
  def to_nodal(self, x: cx.Field) -> cx.Field:
    ...

  @overload
  def to_nodal(self, x: dict[str, cx.Field]) -> dict[str, cx.Field]:
    ...

  def to_nodal(
      self, x: cx.Field | dict[str, cx.Field]
  ) -> cx.Field | dict[str, cx.Field]:
    """Converts a modal field(s) to a nodal field(s)."""
    if isinstance(x, dict):
      return {k: self.to_nodal(v) for k, v in x.items()}
    nodal = cx.cmap(self.to_nodal_array)(x.untag(self.modal_grid))
    return nodal.tag(self.nodal_grid)
