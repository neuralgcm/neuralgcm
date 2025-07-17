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

"""Coordinate systems that describe how data & model states are discretized."""

from __future__ import annotations

import dataclasses
import datetime
from typing import Any, Iterable, Literal, Self, TYPE_CHECKING, cast

import coordax as cx
from dinosaur import coordinate_systems as dinosaur_coordinates
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
import jax
import jax.numpy as jnp
import numpy as np
import treescope
import xarray


if TYPE_CHECKING:
  # import only under TYPE_CHECKING to avoid circular dependency
  # pylint: disable=g-bad-import-order
  # TODO(dkochkov): consider moving UnshardedCoordinate to coordax.
  from neuralgcm.experimental.core import parallelism


SphericalHarmonicsImpl = spherical_harmonic.SphericalHarmonicsImpl
RealSphericalHarmonics = spherical_harmonic.RealSphericalHarmonics
FastSphericalHarmonics = spherical_harmonic.FastSphericalHarmonics
P = jax.sharding.PartitionSpec


SphericalHarmonicsMethodNames = Literal['real', 'fast']
SPHERICAL_HARMONICS_METHODS = {
    'real': RealSphericalHarmonics,
    'fast': FastSphericalHarmonics,
}


def _in_treescope_abbreviation_mode() -> bool:
  """Returns True if treescope.abbreviation is set by context or globally."""
  return treescope.abbreviation_threshold.get() is not None


@dataclasses.dataclass(frozen=True)
class ArrayKey:
  """Wrapper for a numpy array to make it hashable."""

  value: np.ndarray

  def __eq__(self, other):
    return (
        isinstance(self, ArrayKey)
        and self.value.dtype == other.value.dtype
        and self.value.shape == other.value.shape
        and (self.value == other.value).all()
    )

  def __hash__(self) -> int:
    return hash((self.value.shape, self.value.tobytes()))


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class TimeDelta(cx.Coordinate):
  """Coordinates that discretize data along static relative time."""

  deltas: np.ndarray

  def __post_init__(self):
    deltas = np.asarray(self.deltas)
    if not np.issubdtype(deltas.dtype, np.timedelta64):
      raise ValueError(f'deltas must be a timedelta array, got {deltas.dtype=}')
    object.__setattr__(self, 'deltas', deltas.astype('timedelta64[s]'))

  @property
  def dims(self):
    return ('timedelta',)

  @property
  def shape(self):
    return self.deltas.shape

  @property
  def fields(self):
    return {'timedelta': cx.wrap(self.deltas, self)}

  def to_xarray(self) -> dict[str, xarray.Variable]:
    variables = super().to_xarray()
    variables['timedelta'] = xarray.Variable(('timedelta',), self.deltas)
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'timedelta':
      return cx.NoCoordinateMatch(f'dimension {dim!r} != "timedelta"')
    if 'timedelta' not in coords:
      return cx.NoCoordinateMatch('no associated coordinate for timedelta')

    data = coords['timedelta'].data
    if data.ndim != 1:
      return cx.NoCoordinateMatch('timedelta coordinate is not a 1D array')

    if not np.issubdtype(data.dtype, np.timedelta64):
      return cx.NoCoordinateMatch(
          f'data must be a timedelta array, got {data.dtype=}'
      )

    return cls(deltas=data)

  def _components(self):
    return (ArrayKey(self.deltas),)

  def __eq__(self, other):
    return (
        isinstance(other, TimeDelta)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  def __getitem__(self, key: slice) -> Self:
    return type(self)(self.deltas[key])

  def __repr__(self):
    if _in_treescope_abbreviation_mode():
      return treescope.render_to_text(self)
    else:
      with treescope.abbreviation_threshold.set_scoped(1):
        with treescope.using_expansion_strategy(9, 80):
          return treescope.render_to_text(self)

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    """Treescope handler for Field."""
    to_str = lambda x: str(datetime.timedelta(seconds=int(x.astype(int))))
    dts = np.apply_along_axis(to_str, axis=1, arr=self.deltas[:, np.newaxis])
    if dts.size < 6:
      deltas = '[' + ', '.join([str(x) for x in dts]) + ']'
    else:
      deltas = '[' + ', '.join([str(x) for x in dts[:2]])
      deltas += ', ..., '
      deltas += ', '.join([str(x) for x in dts[-2:]]) + ']'
    heading = f'<{type(self).__name__}'
    return treescope.rendering_parts.siblings(
        heading, treescope.rendering_parts.text(deltas), '>'
    )


#
# Grid-like and spherical harmonic coordinate systems
#


def _mesh_to_dinosaur_spmd_mesh(
    dims: tuple[str, ...],
    mesh: parallelism.Mesh | None = None,
    partition_schema_key: str | None = None,
) -> jax.sharding.Mesh | None:
  """Returns spmd_mesh in dinosaur.spherical_harmonic.Grid format."""
  if mesh is not None:
    dim_to_axes = {d: ax for d, ax in zip(dims, ['x', 'y'])}
    # TODO(dkochkov): modify dinosaur.spherical_harmonic.Grid to not
    # assume level dimension in implementation details.
    dim_to_axes['level'] = 'z'
    return mesh.rearrange_spmd_mesh(partition_schema_key, dim_to_axes)
  return None


# TODO(dkochkov) Consider leaving out spherical_harmonics_impl from repr.
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LonLatGrid(cx.Coordinate):
  """Coordinates that discretize data as point values on lon-lat grid."""

  longitude_nodes: int
  latitude_nodes: int
  latitude_spacing: str = 'gauss'
  longitude_offset: float = 0.0
  radius: float = 1.0
  lon_lat_padding: tuple[int, int] = (0, 0)

  @property
  def _ylm_grid(self) -> spherical_harmonic.Grid:
    """Unpadded spherical harmonic grid with matching nodal values."""
    return spherical_harmonic.Grid(
        longitude_wavenumbers=0,
        total_wavenumbers=0,
        longitude_nodes=self.longitude_nodes,
        latitude_nodes=self.latitude_nodes,
        latitude_spacing=self.latitude_spacing,
        longitude_offset=self.longitude_offset,
        radius=self.radius,
    )

  @property
  def dims(self):
    return ('longitude', 'latitude')

  @property
  def shape(self):
    unpadded_shape = (self.longitude_nodes, self.latitude_nodes)
    return tuple(x + y for x, y in zip(unpadded_shape, self.lon_lat_padding))

  @property
  def fields(self):
    lon_pad, lat_pad = self.lon_lat_padding
    lons = jnp.rad2deg(jnp.pad(self._ylm_grid.longitudes, (0, lon_pad)))
    lats = jnp.rad2deg(jnp.pad(self._ylm_grid.latitudes, (0, lat_pad)))
    return {
        'longitude': cx.wrap(lons, cx.SelectedAxis(self, axis=0)),
        'latitude': cx.wrap(lats, cx.SelectedAxis(self, axis=1)),
    }

  @classmethod
  def from_dinosaur_grid(
      cls,
      ylm_grid: spherical_harmonic.Grid,
  ):
    return cls(
        longitude_nodes=ylm_grid.longitude_nodes,
        latitude_nodes=ylm_grid.latitude_nodes,
        latitude_spacing=ylm_grid.latitude_spacing,
        longitude_offset=ylm_grid.longitude_offset,
        radius=ylm_grid.radius,
        lon_lat_padding=ylm_grid.nodal_padding,
    )

  @classmethod
  def construct(
      cls,
      gaussian_nodes: int,
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      radius: float = 1.0,
      mesh: parallelism.Mesh | None = None,
      partition_schema_key: str | None = None,
  ) -> LonLatGrid:
    """Constructs a `LonLatGrid` with specified number of latitude nodes.

    Args:
      gaussian_nodes: number of nodes between the equator and a pole.
      latitude_spacing: either 'gauss' or 'equiangular'. This determines the
        spacing of grid points in the latitudinal (north-south) direction.
      longitude_offset: the value of the first longitude node, in radians.
      radius: radius of the sphere.
      mesh: optional Mesh that specifies necessary grid padding.
      partition_schema_key: key indicating a partition schema on `mesh` to infer
        padding details. Used only if an appropriate `mesh` is passed in.

    Returns:
      Constructed LonLatGrid object.
    """
    dims = ('longitude', 'latitude')
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=0,
        total_wavenumbers=0,
        longitude_nodes=(4 * gaussian_nodes),
        latitude_nodes=(2 * gaussian_nodes),
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        radius=radius,
        spherical_harmonics_impl=FastSphericalHarmonics,
        spmd_mesh=_mesh_to_dinosaur_spmd_mesh(dims, mesh, partition_schema_key),
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  # The factory methods below return "standard" grids that appear in the
  # literature. See, e.g. https://doi.org/10.5194/tc-12-1499-2018 and
  # https://www.ecmwf.int/en/forecasts/documentation-and-support/data-spatial-coordinate-systems

  # The number in these names correspond to the maximum resolved wavenumber,
  # which is one less than the number of wavenumbers used in the Grid
  # constructor. An additional total wavenumber is added because the top
  # wavenumber is clipped from the initial state and each calculation of
  # explicit tendencies.

  # The names for these factory methods (including capilatization) are
  # standard in the literature.
  # pylint:disable=invalid-name

  # T* grids can model quadratic terms without aliasing, because the maximum
  # total wavenumber is <= 2/3 of the number of latitudinal nodes. ECMWF
  # sometimes calls these "TQ" (truncated quadratic) grids.

  @classmethod
  def T21(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=16, **kwargs)

  @classmethod
  def T31(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=24, **kwargs)

  @classmethod
  def T42(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=32, **kwargs)

  @classmethod
  def T85(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=64, **kwargs)

  @classmethod
  def T106(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=80, **kwargs)

  @classmethod
  def T119(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=90, **kwargs)

  @classmethod
  def T170(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=128, **kwargs)

  @classmethod
  def T213(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=160, **kwargs)

  @classmethod
  def T340(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=256, **kwargs)

  @classmethod
  def T425(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=320, **kwargs)

  # TL* grids do not truncate any frequencies, and hence can only model linear
  # terms exactly. ECMWF used "TL" (truncated linear) grids for semi-Lagrangian
  # advection (which eliminates quadratic terms) up to 2016, which it switched
  # to "cubic" grids for resolutions above TL1279:
  # https://www.ecmwf.int/sites/default/files/elibrary/2016/17262-new-grid-ifs.pdf

  @classmethod
  def TL31(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=16, **kwargs)

  @classmethod
  def TL47(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=24, **kwargs)

  @classmethod
  def TL63(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=32, **kwargs)

  @classmethod
  def TL95(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=48, **kwargs)

  @classmethod
  def TL127(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=64, **kwargs)

  @classmethod
  def TL159(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=80, **kwargs)

  @classmethod
  def TL179(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=90, **kwargs)

  @classmethod
  def TL255(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=128, **kwargs)

  @classmethod
  def TL639(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=320, **kwargs)

  @classmethod
  def TL1279(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=640, **kwargs)

  def to_xarray(self) -> dict[str, xarray.Variable]:
    variables = super().to_xarray()
    metadata = dict(
        radius=self.radius,
        lon_lat_padding=self.lon_lat_padding,
    )
    variables['longitude'].attrs = metadata
    variables['latitude'].attrs = metadata
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.NoCoordinateMatch:
    if dims[:2] != ('longitude', 'latitude'):
      return cx.NoCoordinateMatch(
          "leading dimensions are not ('longitude', 'latitude')"
      )

    if coords['longitude'].dims != ('longitude',):
      return cx.NoCoordinateMatch('longitude is not a 1D coordinate')

    if coords['latitude'].dims != ('latitude',):
      return cx.NoCoordinateMatch('latitude is not a 1D coordinate')

    lon = coords['longitude'].data
    lat = coords['latitude'].data

    if lon.max() < 2 * np.pi:
      return cx.NoCoordinateMatch(
          f'expected longitude values in degrees, got {lon}'
      )

    if np.allclose(np.diff(lat), lat[1] - lat[0]):
      if np.isclose(max(lat), 90.0):
        latitude_spacing = 'equiangular_with_poles'
      else:
        latitude_spacing = 'equiangular'
    else:
      latitude_spacing = 'gauss'

    longitude_offset = float(np.deg2rad(coords['longitude'].data[0]))
    longitude_nodes = coords.sizes['longitude']
    latitude_nodes = coords.sizes['latitude']

    radius = float(coords['longitude'].attrs.get('radius', 1.0))
    lon_lat_padding = coords['longitude'].attrs.get('lon_lat_padding', (0, 0))
    result = cls(
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        radius=radius,
        lon_lat_padding=lon_lat_padding,
    )
    result_lat = np.rad2deg(result._ylm_grid.latitudes)
    if not np.allclose(result_lat, lat, atol=1e-3):
      return cx.NoCoordinateMatch(
          f'inferred latitudes with spacing={latitude_spacing!r} do not '
          f' match coordinate data: {result_lat} vs {lat}'
      )

    result_lon = np.rad2deg(result._ylm_grid.longitudes)
    if not np.allclose(result_lon, lon, atol=1e-3):
      return cx.NoCoordinateMatch(
          'inferred longitudes do not match coordinate data:'
          f' {result_lon} vs {lon}'
      )

    return result


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SphericalHarmonicGrid(cx.Coordinate):
  """Coordinates that discretize data as spherical harmonic coefficients."""

  longitude_wavenumbers: int
  total_wavenumbers: int
  radius: float = 1.0
  spherical_harmonics_method: SphericalHarmonicsMethodNames = 'fast'
  wavenumber_padding: tuple[int, int] = (0, 0)

  @property
  def _ylm_grid(self) -> spherical_harmonic.Grid:
    method = SPHERICAL_HARMONICS_METHODS[self.spherical_harmonics_method]
    return spherical_harmonic.Grid(
        longitude_wavenumbers=self.longitude_wavenumbers,
        total_wavenumbers=self.total_wavenumbers,
        longitude_nodes=0,
        latitude_nodes=0,
        latitude_spacing='gauss',
        longitude_offset=0.0,
        radius=self.radius,
        spherical_harmonics_impl=method,
    )

  @property
  def dims(self):
    return ('longitude_wavenumber', 'total_wavenumber')

  @property
  def shape(self) -> tuple[int, ...]:
    unpadded_shape = self._ylm_grid.modal_shape
    return tuple(x + y for x, y in zip(unpadded_shape, self.wavenumber_padding))

  @property
  def fields(self):
    unpadded_ms, unpadded_ls = self._ylm_grid.modal_axes
    m_pad, l_pad = self.wavenumber_padding
    ms = jnp.pad(unpadded_ms, (0, m_pad))
    ls = jnp.pad(unpadded_ls, (0, l_pad))
    axes_fields = {
        k: cx.wrap(v, cx.SelectedAxis(self, i))
        for i, (k, v) in enumerate(zip(self.dims, [ms, ls]))
    }
    unpadded_mask = self._ylm_grid.mask
    mask = jnp.pad(unpadded_mask, ((0, m_pad), (0, l_pad)))
    mask_field = cx.wrap(mask, self)
    return axes_fields | {'mask': mask_field}

  def clip_wavenumbers(self, inputs: Any, n: int) -> Any:
    if n <= 0:
      raise ValueError(f'`n` must be >= 0; got {n}.')

    def clip(x):
      # Multiplication by the mask is significantly faster than directly using
      # `x.at[..., -n:].set(0)`
      num_zeros = n + self.wavenumber_padding[-1]
      mask = jnp.ones(self.shape[-1], x.dtype).at[-num_zeros:].set(0)
      return x * mask

    return jax.tree.map(clip, inputs)

  @classmethod
  def from_dinosaur_grid(
      cls,
      ylm_grid: spherical_harmonic.Grid,
  ):
    cls_to_method = {v: k for k, v in SPHERICAL_HARMONICS_METHODS.items()}
    method_name = cls_to_method[ylm_grid.spherical_harmonics_impl]
    method_name = cast(SphericalHarmonicsMethodNames, method_name)
    return cls(
        longitude_wavenumbers=ylm_grid.longitude_wavenumbers,
        total_wavenumbers=ylm_grid.total_wavenumbers,
        radius=ylm_grid.radius,
        spherical_harmonics_method=method_name,
        wavenumber_padding=ylm_grid.modal_padding,
    )

  @classmethod
  def with_wavenumbers(
      cls,
      longitude_wavenumbers: int,
      spherical_harmonics_method: SphericalHarmonicsMethodNames = 'fast',
      radius: float = 1.0,
      mesh: parallelism.Mesh | None = None,
      partition_schema_key: str | None = None,
  ) -> SphericalHarmonicGrid:
    """Constructs a `SphericalHarmonicGrid` by specifying only wavenumbers."""
    dims = ('longitude_wavenumber', 'total_wavenumber')
    method = SPHERICAL_HARMONICS_METHODS[spherical_harmonics_method]
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=longitude_wavenumbers + 1,
        longitude_nodes=0,
        latitude_nodes=0,
        spherical_harmonics_impl=method,
        radius=radius,
        spmd_mesh=_mesh_to_dinosaur_spmd_mesh(dims, mesh, partition_schema_key),
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  @classmethod
  def construct(
      cls,
      max_wavenumber: int,
      radius: float = 1.0,
      spherical_harmonics_method: SphericalHarmonicsMethodNames = 'fast',
      mesh: parallelism.Mesh | None = None,
      partition_schema_key: str | None = None,
  ) -> SphericalHarmonicGrid:
    """Constructs a `SphericalHarmonicGrid` with max_wavenumber.

    Args:
      max_wavenumber: maximum wavenumber to resolve.
      radius: radius of the sphere.
      spherical_harmonics_method: name of the Yₗᵐ implementation to use.
      mesh: optional Mesh that specifies necessary grid padding.
      partition_schema_key: key indicating a partition schema on `mesh` to infer
        padding details. Used only if an appropriate `mesh` is passed in.

    Returns:
      Constructed SphericalHarmonicGrid object.
    """
    dims = ('longitude_wavenumber', 'total_wavenumber')
    method = SPHERICAL_HARMONICS_METHODS[spherical_harmonics_method]
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=max_wavenumber + 1,
        total_wavenumbers=max_wavenumber + 2,
        longitude_nodes=0,
        latitude_nodes=0,
        spherical_harmonics_impl=method,
        radius=radius,
        spmd_mesh=_mesh_to_dinosaur_spmd_mesh(dims, mesh, partition_schema_key),
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  # The factory methods below return "standard" grids that appear in the
  # literature. See, e.g. https://doi.org/10.5194/tc-12-1499-2018 and
  # https://www.ecmwf.int/en/forecasts/documentation-and-support/data-spatial-coordinate-systems

  # The number in these names correspond to the maximum resolved wavenumber,
  # which is one less than the number of wavenumbers used in the Grid
  # constructor. An additional total wavenumber is added because the top
  # wavenumber is clipped from the initial state and each calculation of
  # explicit tendencies.

  # The names for these factory methods (including capilatization) are
  # standard in the literature.
  # pylint:disable=invalid-name

  # T* grids can model quadratic terms without aliasing, because the maximum
  # total wavenumber is <= 2/3 of the number of latitudinal nodes. ECMWF
  # sometimes calls these "TQ" (truncated quadratic) grids.

  @classmethod
  def T21(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=21, **kwargs)

  @classmethod
  def T31(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=31, **kwargs)

  @classmethod
  def T42(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=42, **kwargs)

  @classmethod
  def T85(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=85, **kwargs)

  @classmethod
  def T106(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=106, **kwargs)

  @classmethod
  def T119(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=119, **kwargs)

  @classmethod
  def T170(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=170, **kwargs)

  @classmethod
  def T213(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=213, **kwargs)

  @classmethod
  def T340(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=340, **kwargs)

  @classmethod
  def T425(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=425, **kwargs)

  # TL* grids do not truncate any frequencies, and hence can only model linear
  # terms exactly. ECMWF used "TL" (truncated linear) grids for semi-Lagrangian
  # advection (which eliminates quadratic terms) up to 2016, which it switched
  # to "cubic" grids for resolutions above TL1279:
  # https://www.ecmwf.int/sites/default/files/elibrary/2016/17262-new-grid-ifs.pdf

  @classmethod
  def TL31(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=31, **kwargs)

  @classmethod
  def TL47(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=47, **kwargs)

  @classmethod
  def TL63(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=63, **kwargs)

  @classmethod
  def TL95(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=95, **kwargs)

  @classmethod
  def TL127(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=127, **kwargs)

  @classmethod
  def TL159(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=159, **kwargs)

  @classmethod
  def TL179(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=179, **kwargs)

  @classmethod
  def TL255(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=255, **kwargs)

  @classmethod
  def TL639(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=639, **kwargs)

  @classmethod
  def TL1279(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=1279, **kwargs)

  def to_xarray(self) -> dict[str, xarray.Variable]:
    variables = super().to_xarray()
    metadata = dict(
        radius=self.radius,
        wavenumber_padding=self.wavenumber_padding,
        spherical_harmonics_method=self.spherical_harmonics_method,
    )
    variables['longitude_wavenumber'].attrs = metadata
    variables['total_wavenumber'].attrs = metadata
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.NoCoordinateMatch:

    if dims[:2] != ('longitude_wavenumber', 'total_wavenumber'):
      return cx.NoCoordinateMatch(
          "leading dimensions are not ('longitude_wavenumber',"
          " 'total_wavenumber')"
      )

    if coords['longitude_wavenumber'].dims != ('longitude_wavenumber',):
      return cx.NoCoordinateMatch('longitude_wavenumber is not a 1D coordinate')

    if coords['total_wavenumber'].dims != ('total_wavenumber',):
      return cx.NoCoordinateMatch('total_wavenumber is not a 1D coordinate')

    longitude_wavenumbers = (coords.sizes['longitude_wavenumber'] + 1) // 2
    radius = float(coords['longitude_wavenumber'].attrs.get('radius', 1.0))
    wavenumber_padding = coords['longitude_wavenumber'].attrs.get(
        'wavenumber_padding', (0, 0)
    )
    spherical_harmonics_method = coords['longitude_wavenumber'].attrs.get(
        'spherical_harmonics_method', 'fast'
    )
    candidate = cls(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=coords.sizes['total_wavenumber'],
        radius=radius,
        spherical_harmonics_method=spherical_harmonics_method,
        wavenumber_padding=wavenumber_padding,
    )

    expected = candidate.fields['longitude_wavenumber'].data
    got = coords['longitude_wavenumber'].data
    if not np.array_equal(expected, got):
      return cx.NoCoordinateMatch(
          'inferred longitude wavenumbers do not match coordinate data:'
          f' {expected} vs {got}. Perhaps you attempted to restore coordinate '
          ' data from FastSphericalHarmonics, which does not support '
          'restoration?'
      )

    expected = candidate.fields['total_wavenumber'].data
    got = coords['total_wavenumber'].data
    if not np.array_equal(expected, got):
      return cx.NoCoordinateMatch(
          f'inferred total wavenumbers do not match coordinate data: {expected}'
          f' vs {got}. Perhaps you attempted to restore coordinate '
          ' data from FastSphericalHarmonics, which does not support '
          'restoration?'
      )

    return candidate


#
# Vertical level coordinates
#


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SigmaLevels(cx.Coordinate):
  """Coordinates that discretize data as fraction of the surface pressure."""

  boundaries: np.ndarray
  sigma_levels: sigma_coordinates.SigmaCoordinates = dataclasses.field(
      init=False, repr=False, compare=False
  )

  def __init__(self, boundaries: Iterable[float] | np.ndarray):
    boundaries = np.asarray(boundaries, np.float32)
    object.__setattr__(self, 'boundaries', boundaries)
    self.__post_init__()

  def __post_init__(self):
    sigma_levels = sigma_coordinates.SigmaCoordinates(
        boundaries=self.boundaries
    )
    object.__setattr__(self, 'sigma_levels', sigma_levels)

  @property
  def dims(self):
    return ('sigma',)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.sigma_levels.centers.shape

  @property
  def fields(self):
    return {'sigma': cx.wrap(self.sigma_levels.centers, self)}

  @property
  def centers(self):
    return self.sigma_levels.centers

  def asdict(self) -> dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def _components(self):
    return (ArrayKey(self.boundaries),)

  def __eq__(self, other):
    return (
        isinstance(other, SigmaLevels)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  @classmethod
  def from_dinosaur_sigma_levels(
      cls,
      sigma_levels: sigma_coordinates.SigmaCoordinates,
  ):
    return cls(boundaries=sigma_levels.boundaries)

  @classmethod
  def equidistant(
      cls,
      layers: int,
  ) -> SigmaLevels:
    sigma_levels = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    boundaries = sigma_levels.boundaries
    return cls(boundaries=boundaries)

  @classmethod
  def from_centers(cls, centers: np.ndarray) -> Self:
    sigma_levels = sigma_coordinates.SigmaCoordinates.from_centers(centers)
    boundaries = sigma_levels.boundaries
    return cls(boundaries=boundaries)

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'sigma':
      return cx.NoCoordinateMatch(f'dimension {dim!r} != "sigma"')

    if coords['sigma'].ndim != 1:
      return cx.NoCoordinateMatch('sigma coordinate is not a 1D array')

    centers = coords['sigma'].data
    candidate = cls.from_centers(centers)
    actual_centers = candidate.sigma_levels.centers
    if not np.array_equal(actual_centers, centers):
      return cx.NoCoordinateMatch(
          'inferred sigma boundaries do not exactly match coordinate data:'
          f' {actual_centers} vs {centers}.'
      )
    return candidate


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class PressureLevels(cx.Coordinate):
  """Coordinates that discretize data per pressure levels."""

  centers: np.ndarray
  pressure_levels: vertical_interpolation.PressureCoordinates = (
      dataclasses.field(init=False, repr=False, compare=False)
  )

  def __init__(self, centers: Iterable[float] | np.ndarray):
    centers = np.asarray(centers, dtype=np.float32)
    object.__setattr__(self, 'centers', centers)
    self.__post_init__()

  def __post_init__(self):
    pressure_levels = vertical_interpolation.PressureCoordinates(
        centers=self.centers
    )
    object.__setattr__(self, 'pressure_levels', pressure_levels)

  @property
  def dims(self):
    return ('pressure',)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.centers.shape

  @property
  def fields(self):
    return {'pressure': cx.wrap(self.centers, self)}

  def asdict(self) -> dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def _components(self):
    return (ArrayKey(self.centers),)

  def __eq__(self, other):
    return (
        isinstance(other, PressureLevels)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  @classmethod
  def from_dinosaur_pressure_levels(
      cls,
      pressure_levels: vertical_interpolation.PressureCoordinates,
  ):
    return cls(centers=pressure_levels.centers)

  @classmethod
  def with_era5_levels(cls):
    """PressureLevels with standard 37 ERA5 pressure levels."""
    return cls(
        centers=[
            1,
            2,
            3,
            5,
            7,
            10,
            20,
            30,
            50,
            70,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            775,
            800,
            825,
            850,
            875,
            900,
            925,
            950,
            975,
            1000,
        ]
    )

  @classmethod
  def with_13_era5_levels(cls):
    """PressureLevels with commonly used subset of 13 ERA5 pressure levels."""
    centers = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    return cls(centers=centers)

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.NoCoordinateMatch:
    dim = dims[0]
    if dim not in {'level', 'pressure'}:
      return cx.NoCoordinateMatch(
          f'dimension {dim!r} is not "pressure" or "level"'
      )
    if coords[dim].ndim != 1:
      return cx.NoCoordinateMatch('pressure coordinate is not a 1D array')
    centers = coords[dim].data
    if not 0 < centers[0] < 100:
      return cx.NoCoordinateMatch(
          f'pressure levels must start between 0 and 100, got: {centers}'
      )
    if not 900 < centers[-1] < 1025:
      return cx.NoCoordinateMatch(
          f'pressure levels must end between 900 and 1025, got: {centers}'
      )
    return cls(centers=centers)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LayerLevels(cx.Coordinate):
  """Coordinates that discretize data by index of unstructured layer."""

  n_layers: int
  name: str = 'layer_index'

  @property
  def dims(self):
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return (self.n_layers,)

  @property
  def fields(self):
    return {self.name: cx.wrap(np.arange(self.n_layers), self)}

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'layer_index':
      return cx.NoCoordinateMatch(f'dimension {dim!r} != "layer_index"')

    if coords['layer_index'].ndim != 1:
      return cx.NoCoordinateMatch('layer_index coordinate is not a 1D array')

    n_layers = coords.sizes['layer_index']
    got = coords['layer_index'].data
    if not np.array_equal(got, np.arange(n_layers)):
      return cx.NoCoordinateMatch(
          f'unexpected layer_index coordinate is not sequential integers: {got}'
      )
    return cls(n_layers=n_layers)


#
# Wrapper coordinates
#
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class CoordinateWithPadding(cx.Coordinate):
  """Coordinate wrapper that represents a coordinate with padding."""

  coordinate: cx.Coordinate
  pad_sizes: dict[str, tuple[int, int]]

  def __post_init__(self):
    if set(self.coordinate.dims) != set(self.pad_sizes.keys()):
      raise ValueError(
          f'{self.pad_sizes=} are expected to specify pads for each dimension '
          f'in {self.coordinate=}'
      )

  @property
  def dims(self):
    return tuple(f'padded_{d}' for d in self.coordinate.dims)

  @property
  def shape(self) -> tuple[int, ...]:
    dims_and_shapes = zip(self.coordinate.dims, self.coordinate.shape)
    return tuple(s + sum(self.pad_sizes[d]) for d, s in dims_and_shapes)

  @property
  def fields(self):
    return {}


#
# Solver-specific coordinate combinations
#


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class DinosaurCoordinates(cx.CartesianProduct):
  """Coordinate that is product of horizontal & vertical coorinates.

  This combined coordinate object is useful for compactly keeping track of the
  full coordinate system of the Dinosaur dynamic core or pressure-level
  representation of the spherical shell data.
  """

  coordinates: tuple[cx.Coordinate, ...] = dataclasses.field(init=False)
  horizontal: LonLatGrid | SphericalHarmonicGrid = dataclasses.field()
  vertical: SigmaLevels | PressureLevels | LayerLevels = dataclasses.field()
  dycore_partition_spec: jax.sharding.PartitionSpec = P('z', 'x', 'y')
  physics_partition_spec: jax.sharding.PartitionSpec = P(None, ('x', 'z'), 'y')

  def __init__(
      self,
      horizontal,
      vertical,
      dycore_partition_spec: jax.sharding.PartitionSpec = P('z', 'x', 'y'),
      physics_partition_spec: jax.sharding.PartitionSpec = P(
          None, ('x', 'z'), 'y'
      ),
  ):
    super().__init__(coordinates=(vertical, horizontal))
    object.__setattr__(self, 'horizontal', horizontal)
    object.__setattr__(self, 'vertical', vertical)
    object.__setattr__(self, 'dycore_partition_spec', dycore_partition_spec)
    object.__setattr__(self, 'physics_partition_spec', physics_partition_spec)

  @property
  def dims(self):
    return self.vertical.dims + self.horizontal.dims

  @property
  def shape(self) -> tuple[int, ...]:
    return self.vertical.shape + self.horizontal.shape

  @property
  def fields(self):
    return self.vertical.fields | self.horizontal.fields

  @classmethod
  def from_dinosaur_coords(
      cls,
      coords: dinosaur_coordinates.CoordinateSystem,
  ):
    """Constructs instance from coordinates in Dinosaur package."""
    horizontal = LonLatGrid.from_dinosaur_grid(coords.horizontal)
    if isinstance(coords.vertical, sigma_coordinates.SigmaCoordinates):
      vertical = SigmaLevels.from_dinosaur_sigma_levels(coords.vertical)
    elif isinstance(
        coords.vertical, vertical_interpolation.PressureCoordinates
    ):
      vertical = PressureLevels.from_dinosaur_pressure_levels(coords.vertical)
    else:
      raise ValueError(f'Unsupported vertical {coords.vertical=}')
    return cls(horizontal=horizontal, vertical=vertical)


#
# Helper functions.
#
# TODO(dkochkov) Refactor/remove helpers below to coordax.coords.


def field_from_xarray(
    data_array: xarray.DataArray,
    additional_coord_types: tuple[cx.Coordinate, ...] = (),
) -> cx.Field:
  """Converts an xarray.DataArray to a Field using NeuralGCM coordinates."""
  # TODO(shoyer): add DinosaurCoordinates into this list?
  coord_types = (
      TimeDelta,
      LonLatGrid,
      SphericalHarmonicGrid,
      PressureLevels,
      SigmaLevels,
      LayerLevels,
      cx.DummyAxis,
  )
  return cx.Field.from_xarray(data_array, coord_types + additional_coord_types)


def consistent_coords(*inputs) -> cx.Coordinate:
  """Returns the unique coordinate, or raises a ValueError."""
  if not all(cx.is_field(f) for f in inputs):
    raise TypeError(f'inputs are not all fields: {inputs}')
  # TODO(dkochkov): extract dimensions from named_dims. named_shape.keys() are
  # keys of a dict and hence don't have a guaranteed order.
  dim_names = {tuple(f.named_shape.keys()) for f in inputs}
  if len(dim_names) != 1:
    raise ValueError(f'Found non-unique {dim_names=} in inputs.')
  (dim_names,) = dim_names
  coords = {
      cx.compose_coordinates(*[f.coords[k] for k in dim_names]) for f in inputs
  }
  if len(coords) != 1:
    raise ValueError(f'Found non-unique {coords=} in inputs.')
  (coords,) = coords
  return coords


def split_field_attrs(pytree):
  """Splits pytree of `Field` into data, sim_time and spec."""
  is_field = lambda x: isinstance(x, cx.Field)
  fields, treedef = jax.tree.flatten(pytree, is_leaf=is_field)
  data = jax.tree.unflatten(treedef, [x.data for x in fields])
  coords = consistent_coords(*fields)
  return data, coords
