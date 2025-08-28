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
import functools
from typing import Literal, overload

import coordax as cx
from dinosaur import spherical_harmonic
import jax
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
class FixedYlmMapping:
  """Fixed spherical harmonic transform specified by grids."""

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

  def to_modal_array(self, x: typing.cx.Field) -> typing.cx.Field:
    """Converts a nodal array to a modal array."""
    return self.dinosaur_grid.to_modal(x)

  def to_nodal_array(self, x: typing.cx.Field) -> typing.cx.Field:
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

  def cos_lat(self, x: cx.Field) -> cx.Field:
    """Returns the cos(lat) values verified to be present in `x`."""
    if not all(ax in x.axes.values() for ax in self.nodal_grid.axes):
      raise ValueError(f'{self.nodal_grid=} was not found on {x=}')
    return self.nodal_grid.cos_lat

  def laplacian(self, x: cx.Field) -> cx.Field:
    """Returns the Laplacian of `x`."""
    x = x.untag(self.modal_grid)
    y = cx.cmap(self.dinosaur_grid.laplacian, out_axes=x.named_axes)(x)
    return y.tag(self.modal_grid)

  def inverse_laplacian(self, x: cx.Field) -> cx.Field:
    """Returns the inverse Laplacian of `x`."""
    x = x.untag(self.modal_grid)
    y = cx.cmap(self.dinosaur_grid.inverse_laplacian, out_axes=x.named_axes)(x)
    return y.tag(self.modal_grid)

  def d_dlon(self, x: cx.Field) -> cx.Field:
    """Returns the longitudinal derivative of `x`."""
    x = x.untag(self.modal_grid)
    result = cx.cmap(self.dinosaur_grid.d_dlon, out_axes=x.named_axes)(x)
    return result.tag(self.modal_grid)

  def cos_lat_d_dlat(self, x: cx.Field) -> cx.Field:
    """Returns the cos(lat)-weighted latitudinal derivative of `x`."""
    x = x.untag(self.modal_grid)
    y = cx.cmap(self.dinosaur_grid.cos_lat_d_dlat, out_axes=x.named_axes)(x)
    return y.tag(self.modal_grid)

  def sec_lat_d_dlat_cos2(self, x: cx.Field) -> cx.Field:
    """Returns `secθ ∂/∂θ(cos²θ x)`."""
    x = x.untag(self.modal_grid)
    y = cx.cmap(self.dinosaur_grid.sec_lat_d_dlat_cos2, out_axes=x.named_axes)(
        x
    )
    return y.tag(self.modal_grid)

  def clip_wavenumbers(self, x: cx.Field, n: int = 1) -> cx.Field:
    """Zeros out the highest `n` total wavenumbers."""
    return self.modal_grid.clip_wavenumbers(x, n)

  def cos_lat_grad(
      self, x: cx.Field, clip: bool = True
  ) -> tuple[cx.Field, cx.Field]:
    """Returns the cos(lat) gradient of `x`."""
    x = x.untag(self.modal_grid)
    grad_fn = lambda x: self.dinosaur_grid.cos_lat_grad(x, clip=clip)
    u, v = cx.cmap(grad_fn, out_axes=x.named_axes)(x)
    return u.tag(self.modal_grid), v.tag(self.modal_grid)

  def k_cross(self, u: cx.Field, v: cx.Field) -> tuple[cx.Field, cx.Field]:
    """Returns the k-cross of `(u, v)`."""
    u = u.untag(self.modal_grid)
    v = v.untag(self.modal_grid)
    if u.named_axes != v.named_axes:
      raise ValueError(f'Axes mismatch: {u.named_axes=} vs {v.named_axes=}')
    out_axes = u.named_axes
    k_cross_fn = lambda u, v: self.dinosaur_grid.k_cross((u, v))
    out_u, out_v = cx.cmap(k_cross_fn, out_axes=out_axes)(u, v)
    return out_u.tag(self.modal_grid), out_v.tag(self.modal_grid)

  def div_cos_lat(
      self, u: cx.Field, v: cx.Field, clip: bool = True
  ) -> cx.Field:
    """Returns the cos(lat)-weighted divergence of `(u, v)`."""
    u = u.untag(self.modal_grid)
    v = v.untag(self.modal_grid)
    if u.named_axes != v.named_axes:
      raise ValueError(f'Axes mismatch: {u.named_axes=} vs {v.named_axes=}')
    out_axes = u.named_axes
    div_fn = lambda u, v: self.dinosaur_grid.div_cos_lat((u, v), clip=clip)
    result = cx.cmap(div_fn, out_axes=out_axes)(u, v)
    return result.tag(self.modal_grid)

  def curl_cos_lat(
      self, u: cx.Field, v: cx.Field, clip: bool = True
  ) -> cx.Field:
    """Returns the cos(lat)-weighted curl of `(u, v)`."""
    u = u.untag(self.modal_grid)
    v = v.untag(self.modal_grid)
    if u.named_axes != v.named_axes:
      raise ValueError(f'Axes mismatch: {u.named_axes=} vs {v.named_axes=}')
    out_axes = u.named_axes
    curl_fn = lambda u, v: self.dinosaur_grid.curl_cos_lat((u, v), clip=clip)
    result = cx.cmap(curl_fn, out_axes=out_axes)(u, v)
    return result.tag(self.modal_grid)

  def integrate(self, x: cx.Field) -> cx.Field:
    """Returns the integral of `x` over the sphere."""
    x = x.untag(self.nodal_grid)
    return cx.cmap(self.dinosaur_grid.integrate, out_axes=x.named_axes)(x)


# TODO(dkochkov): Remove this alias once all new models have been updated to use
# the updated name.
SphericalHarmonicsTransform = FixedYlmMapping


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
  ) -> FixedYlmMapping:
    if isinstance(grid, coordinates.SphericalHarmonicGrid):
      ylm_grid = grid
      nodal_grid = self.nodal_grid(ylm_grid)
    elif isinstance(grid, coordinates.LonLatGrid):
      nodal_grid = grid
      ylm_grid = self.modal_grid(grid)
    else:
      raise ValueError(f'Unsupported {type(grid)=}')
    return FixedYlmMapping(
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

    return self.ylm_transform(self._extract_nodal_grid(x)).to_modal(x)

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

    return self.ylm_transform(self._extract_modal_grid(x)).to_nodal(x)

  def _extract_nodal_grid(self, x: cx.Field) -> coordinates.LonLatGrid:
    """Returns LonLatGrid from coordinates of `x`."""
    if 'longitude' in x.axes and 'latitude' in x.axes:
      return cx.compose_coordinates(x.axes['longitude'], x.axes['latitude'])
    else:
      raise ValueError(f'No LonLatGrid in {x.axes=}')

  def _extract_modal_grid(
      self, x: cx.Field
  ) -> coordinates.SphericalHarmonicGrid:
    """Returns SphericalHarmonicGrid from coordinates of `x`."""
    if 'longitude_wavenumber' in x.axes and 'total_wavenumber' in x.axes:
      return cx.compose_coordinates(
          x.axes['longitude_wavenumber'], x.axes['total_wavenumber']
      )
    else:
      raise ValueError(f'No SphericalHarmonicGrid in {x.axes=}')

  def cos_lat(self, x: cx.Field) -> cx.Field:
    """Returns the cos(lat) values verified to be present in `x`."""
    return self._extract_nodal_grid(x).cos_lat

  def laplacian(self, x: cx.Field) -> cx.Field:
    """Returns the Laplacian of `x`."""
    ylm_mapping = self.ylm_transform(self._extract_modal_grid(x))
    return ylm_mapping.laplacian(x)

  def inverse_laplacian(self, x: cx.Field) -> cx.Field:
    """Returns the inverse Laplacian of `x`."""
    ylm_mapping = self.ylm_transform(self._extract_modal_grid(x))
    return ylm_mapping.inverse_laplacian(x)

  def d_dlon(self, x: cx.Field) -> cx.Field:
    """Returns the longitudinal derivative of `x`."""
    ylm_mapping = self.ylm_transform(self._extract_modal_grid(x))
    return ylm_mapping.d_dlon(x)

  def cos_lat_d_dlat(self, x: cx.Field) -> cx.Field:
    """Returns the cos(lat)-weighted latitudinal derivative of `x`."""
    ylm_mapping = self.ylm_transform(self._extract_modal_grid(x))
    return ylm_mapping.cos_lat_d_dlat(x)

  def sec_lat_d_dlat_cos2(self, x: cx.Field) -> cx.Field:
    """Returns `secθ ∂/∂θ(cos²θ x)`."""
    ylm_mapping = self.ylm_transform(self._extract_modal_grid(x))
    return ylm_mapping.sec_lat_d_dlat_cos2(x)

  def clip_wavenumbers(self, x: cx.Field, n: int = 1) -> cx.Field:
    """Zeros out the highest `n` total wavenumbers."""
    ylm_mapping = self.ylm_transform(self._extract_modal_grid(x))
    return ylm_mapping.clip_wavenumbers(x, n)

  def cos_lat_grad(
      self, x: cx.Field, clip: bool = True
  ) -> tuple[cx.Field, cx.Field]:
    """Returns the cos(lat) gradient of `x`."""
    ylm_mapping = self.ylm_transform(self._extract_modal_grid(x))
    return ylm_mapping.cos_lat_grad(x, clip=clip)

  def k_cross(self, u: cx.Field, v: cx.Field) -> tuple[cx.Field, cx.Field]:
    """Returns the k-cross of `(u, v)`."""
    ylm_mapping = self.ylm_transform(self._extract_modal_grid(u))
    return ylm_mapping.k_cross(u, v)

  def div_cos_lat(
      self, u: cx.Field, v: cx.Field, clip: bool = True
  ) -> cx.Field:
    """Returns the cos(lat)-weighted divergence of `(u, v)`."""
    ylm_mapping = self.ylm_transform(self._extract_modal_grid(u))
    return ylm_mapping.div_cos_lat(u, v, clip=clip)

  def curl_cos_lat(
      self, u: cx.Field, v: cx.Field, clip: bool = True
  ) -> cx.Field:
    """Returns the cos(lat)-weighted curl of `(u, v)`."""
    ylm_mapping = self.ylm_transform(self._extract_modal_grid(u))
    return ylm_mapping.curl_cos_lat(u, v, clip=clip)

  def integrate(self, x: cx.Field) -> cx.Field:
    """Returns the integral of `x` over the sphere."""
    ylm_mapping = self.ylm_transform(self._extract_nodal_grid(x))
    return ylm_mapping.integrate(x)


@jax.named_call
def get_cos_lat_vector(
    vorticity: cx.Field,
    divergence: cx.Field,
    ylm_map: FixedYlmMapping | YlmMapper,
    clip: bool = True,
) -> tuple[cx.Field, cx.Field]:
  """Computes `v cosθ`, where θ denotes latitude."""
  stream_function = ylm_map.inverse_laplacian(vorticity)
  velocity_potential = ylm_map.inverse_laplacian(divergence)
  return tuple(
      x + y
      for x, y in zip(
          ylm_map.cos_lat_grad(velocity_potential, clip=clip),
          ylm_map.k_cross(*ylm_map.cos_lat_grad(stream_function, clip=clip)),
      )
  )


@functools.partial(jax.jit, static_argnames=('ylm_map', 'clip'))
def uv_nodal_to_vor_div_modal(
    u_nodal: cx.Field,
    v_nodal: cx.Field,
    ylm_map: FixedYlmMapping | YlmMapper,
    clip: bool = True,
) -> tuple[cx.Field, cx.Field]:
  """Converts nodal `u, v` vectors to a modal `vort, div` representation."""
  u_over_cos_lat = ylm_map.to_modal(u_nodal / ylm_map.cos_lat(u_nodal))
  v_over_cos_lat = ylm_map.to_modal(v_nodal / ylm_map.cos_lat(v_nodal))
  vorticity = ylm_map.curl_cos_lat(u_over_cos_lat, v_over_cos_lat, clip=clip)
  divergence = ylm_map.div_cos_lat(u_over_cos_lat, v_over_cos_lat, clip=clip)
  return vorticity, divergence


@functools.partial(jax.jit, static_argnames=('ylm_map', 'clip'))
def vor_div_to_uv_nodal(
    vorticity: cx.Field,
    divergence: cx.Field,
    ylm_map: FixedYlmMapping | YlmMapper,
    clip: bool = True,
) -> tuple[cx.Field, cx.Field]:
  """Converts modal `vorticity, divergence` to a nodal `u, v` representation."""
  u_cos_lat, v_cos_lat = get_cos_lat_vector(
      vorticity, divergence, ylm_map, clip=clip
  )
  u_cos_lat = ylm_map.to_nodal(u_cos_lat)
  v_cos_lat = ylm_map.to_nodal(v_cos_lat)
  return (
      u_cos_lat / ylm_map.cos_lat(u_cos_lat),
      v_cos_lat / ylm_map.cos_lat(v_cos_lat),
  )
