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
from typing import Literal, overload

from dinosaur import spherical_harmonic
import jax
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import typing


# TODO(dkochkov): Consider dropping nodal_grid and modal_grid properties and
# instead have them as arguments when grids contain explicit padding details.


SphericalHarmonicMethods = Literal['fast', 'real']
TruncationRules = Literal['linear', 'cubic']
Grid = spherical_harmonic.Grid
FastSphericalHarmonics = spherical_harmonic.FastSphericalHarmonics
RealSphericalHarmonics = spherical_harmonic.RealSphericalHarmonics

# fmt: off
cubic_dino_grid_constructors = [
    Grid.T21, Grid.T31, Grid.T42, Grid.T85, Grid.T106, Grid.T119, Grid.T170,
    Grid.T213, Grid.T340, Grid.T425
]
linear_dino_grid_constructors = [
    Grid.TL31, Grid.TL47, Grid.TL63, Grid.TL95, Grid.TL127, Grid.TL159,
    Grid.TL179, Grid.TL255, Grid.TL639, Grid.TL1279
]
# fmt: on

# Cubic shape-to-cls dicts.
FAST_CUBIC_NODAL_SHAPE_TO_GRID = {
    grid(spherical_harmonics_impl=FastSphericalHarmonics).nodal_shape: grid
    for grid in cubic_dino_grid_constructors
}
REAL_CUBIC_NODAL_SHAPE_TO_GRID = {
    grid(spherical_harmonics_impl=RealSphericalHarmonics).nodal_shape: grid
    for grid in cubic_dino_grid_constructors
}
FAST_CUBIC_MODAL_SHAPE_TO_GRID = {
    grid(spherical_harmonics_impl=FastSphericalHarmonics).modal_shape: grid
    for grid in cubic_dino_grid_constructors
}
REAL_CUBIC_MODAL_SHAPE_TO_GRID = {
    grid(spherical_harmonics_impl=RealSphericalHarmonics).modal_shape: grid
    for grid in cubic_dino_grid_constructors
}
# Linear shape-to-cls dicts.
FAST_LINEAR_NODAL_SHAPE_TO_GRID = {
    grid(spherical_harmonics_impl=FastSphericalHarmonics).nodal_shape: grid
    for grid in linear_dino_grid_constructors
}
REAL_LINEAR_NODAL_SHAPE_TO_GRID = {
    grid(spherical_harmonics_impl=RealSphericalHarmonics).nodal_shape: grid
    for grid in linear_dino_grid_constructors
}
FAST_LINEAR_MODAL_SHAPE_TO_GRID = {
    grid(spherical_harmonics_impl=FastSphericalHarmonics).modal_shape: grid
    for grid in linear_dino_grid_constructors
}
REAL_LINEAR_MODAL_SHAPE_TO_GRID = {
    grid(spherical_harmonics_impl=RealSphericalHarmonics).modal_shape: grid
    for grid in linear_dino_grid_constructors
}
NODAL_SHAPE_TO_GRID = {
    'fast': {
        'linear': FAST_LINEAR_NODAL_SHAPE_TO_GRID,
        'cubic': FAST_CUBIC_NODAL_SHAPE_TO_GRID,
    },
    'real': {
        'linear': REAL_LINEAR_NODAL_SHAPE_TO_GRID,
        'cubic': REAL_CUBIC_NODAL_SHAPE_TO_GRID,
    },
}
MODAL_SHAPE_TO_GRID = {
    'fast': {
        'linear': FAST_LINEAR_MODAL_SHAPE_TO_GRID,
        'cubic': FAST_CUBIC_MODAL_SHAPE_TO_GRID,
    },
    'real': {
        'linear': REAL_LINEAR_MODAL_SHAPE_TO_GRID,
        'cubic': REAL_CUBIC_MODAL_SHAPE_TO_GRID,
    },
}


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
    x = x.untag(self.nodal_grid)
    modal = cx.cmap(self.to_modal_array, out_axes=x.named_axes)(x)
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
    x = x.untag(self.modal_grid)
    nodal = cx.cmap(self.to_nodal_array, out_axes=x.named_axes)(x)
    return nodal.tag(self.nodal_grid)


@dataclasses.dataclass(frozen=True, kw_only=True)
class YlmMapper:
  """Family of spherical harmonic transforms specified by truncation rule.

  This class provides a default way of specifying a collection of spherical
  harmonics transforms specified by a truncation rule, spherical harmonic
  implementation method and relevant parallelism mesh needed to infer paddding.

  Attributes:
    truncation_rule: The truncation rule used to match nodal and modal grids.
    spherical_harmonics_method: Name of the spherical harmonics representations
      implementation. Must be one of `fast` or `real`.
    partition_schema_key: The key specifying the partition schema in the mesh.
    mesh: The parallelism mesh used for sharding.
    level_key: The dimension name to be used as levels in dinosaur mesh.
    longitude_key: The dimension name to be used as longitudes in dinosaur mesh.
    latitude_key: The dimension name to be used as latitudes in dinosaur mesh.
    radius: The radius of the sphere.
  """

  truncation_rule: TruncationRules = 'cubic'
  spherical_harmonics_method: SphericalHarmonicMethods = 'fast'
  partition_schema_key: parallelism.Schema | None
  mesh: parallelism.Mesh = dataclasses.field(kw_only=True)
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

  def dinosaur_grid(
      self,
      grid: coordinates.LonLatGrid | coordinates.SphericalHarmonicGrid,
  ) -> spherical_harmonic.Grid:
    """Returns a dinosaur grid for `coord` based on the truncation rule."""
    dino_mesh = self.dinosaur_spmd_mesh
    if isinstance(grid, coordinates.LonLatGrid):
      shape = tuple(s - p for s, p in zip(grid.shape, grid.lon_lat_padding))
      constructors = NODAL_SHAPE_TO_GRID[self.spherical_harmonics_method]
      grid_constructor = constructors[self.truncation_rule][shape]
    elif isinstance(grid, coordinates.SphericalHarmonicGrid):
      shape = tuple(s - p for s, p in zip(grid.shape, grid.wavenumber_padding))
      constructors = MODAL_SHAPE_TO_GRID[self.spherical_harmonics_method]
      grid_constructor = constructors[self.truncation_rule][shape]
    else:
      raise ValueError(
          f'Unsupported {type(grid)=}, expected LonLatGrid or'
          ' SphericalHarmonicGrid.'
      )
    method = coordinates.SPHERICAL_HARMONICS_METHODS[
        self.spherical_harmonics_method
    ]
    return grid_constructor(
        spmd_mesh=dino_mesh, spherical_harmonics_impl=method
    )

  def modal_grid(
      self, grid: coordinates.LonLatGrid
  ) -> coordinates.SphericalHarmonicGrid:
    dino_grid = self.dinosaur_grid(grid)
    return coordinates.SphericalHarmonicGrid.from_dinosaur_grid(dino_grid)

  def nodal_grid(
      self, ylm_grid: coordinates.SphericalHarmonicGrid
  ) -> coordinates.LonLatGrid:
    dino_grid = self.dinosaur_grid(ylm_grid)
    return coordinates.LonLatGrid.from_dinosaur_grid(dino_grid)

  def ylm_transform(
      self, grid: coordinates.SphericalHarmonicGrid | coordinates.LonLatGrid
  ) -> SphericalHarmonicsTransform:
    if isinstance(grid, coordinates.SphericalHarmonicGrid):
      ylm_grid = grid
      nodal_grid = self.nodal_grid(ylm_grid)
    elif isinstance(grid, coordinates.LonLatGrid):
      nodal_grid = grid
      ylm_grid = self.modal_grid(grid)
    else:
      raise ValueError(f'Unsupported {type(grid)=}')
    return SphericalHarmonicsTransform(
        nodal_grid,
        ylm_grid,
        mesh=self.mesh,
        partition_schema_key=self.partition_schema_key,
        level_key=self.level_key,
        longitude_key=self.longitude_key,
        latitude_key=self.latitude_key,
        radius=self.radius,
    )

  @overload
  def to_modal(self, x: cx.Field) -> cx.Field:
    ...

  @overload
  def to_modal(self, x: dict[str, cx.Field]) -> dict[str, cx.Field]:
    ...

  def to_modal(
      self, x: cx.Field | dict[str, cx.Field]
  ) -> cx.Field | dict[str, cx.Field]:
    """Converts `x` to a modal coordinates."""
    if isinstance(x, dict):
      return {k: self.to_modal(v) for k, v in x.items()}

    grid = cx.compose_coordinates(x.axes['longitude'], x.axes['latitude'])
    return self.ylm_transform(grid).to_modal(x)

  @overload
  def to_nodal(self, x: cx.Field) -> cx.Field:
    ...

  @overload
  def to_nodal(self, x: dict[str, cx.Field]) -> dict[str, cx.Field]:
    ...

  def to_nodal(
      self, x: cx.Field | dict[str, cx.Field]
  ) -> cx.Field | dict[str, cx.Field]:
    """Converts `x` to a nodal coordinates."""
    if isinstance(x, dict):
      return {k: self.to_nodal(v) for k, v in x.items()}

    ylm_grid = cx.compose_coordinates(
        x.axes['longitude_wavenumber'], x.axes['total_wavenumber']
    )
    return self.ylm_transform(ylm_grid).to_nodal(x)
