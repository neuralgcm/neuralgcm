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

import collections
import dataclasses
import math
from typing import Any, Iterable, Self

from dinosaur import coordinate_systems as dinosaur_coordinates
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
import numpy as np
import xarray


SphericalHarmonicsImpl = spherical_harmonic.SphericalHarmonicsImpl
RealSphericalHarmonics = spherical_harmonic.RealSphericalHarmonics
FastSphericalHarmonics = spherical_harmonic.FastSphericalHarmonics
P = jax.sharding.PartitionSpec


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
    # TODO(shoyer: support jax_datetime.Timedelta arrays inside coordax.Field
    return {}

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
      return cx.NoCoordinateMatch(f"dimension {dim!r} != 'timedelta'")
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


#
# Grid-like and spherical harmonic coordinate systems
#


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
  spherical_harmonics_impl: SphericalHarmonicsImpl = dataclasses.field(
      default=FastSphericalHarmonics, kw_only=True
  )
  longitude_wavenumbers: int = dataclasses.field(default=0, kw_only=True)
  total_wavenumbers: int = dataclasses.field(default=0, kw_only=True)
  spmd_mesh: jax.sharding.Mesh | None = dataclasses.field(
      default=None, kw_only=True
  )
  _ylm_grid: spherical_harmonic.Grid = dataclasses.field(
      init=False, repr=False, compare=False
  )

  def __post_init__(self):
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=self.longitude_wavenumbers,
        total_wavenumbers=self.total_wavenumbers,
        longitude_nodes=self.longitude_nodes,
        latitude_nodes=self.latitude_nodes,
        latitude_spacing=self.latitude_spacing,
        longitude_offset=self.longitude_offset,
        radius=self.radius,
        spherical_harmonics_impl=self.spherical_harmonics_impl,
        spmd_mesh=self.spmd_mesh,
    )
    object.__setattr__(self, '_ylm_grid', ylm_grid)

  @property
  def ylm_grid(self) -> spherical_harmonic.Grid:
    return self._ylm_grid

  @property
  def dims(self):
    return ('longitude', 'latitude')

  @property
  def shape(self):
    return self.ylm_grid.nodal_shape

  @property
  def fields(self):
    lons = jnp.rad2deg(self.ylm_grid.longitudes)
    lats = jnp.rad2deg(self.ylm_grid.latitudes)
    return {
        'longitude': cx.wrap(lons, cx.SelectedAxis(self, axis=0)),
        'latitude': cx.wrap(lats, cx.SelectedAxis(self, axis=1)),
    }

  def to_spherical_harmonic_grid(
      self,
      longitude_wavenumbers: int | None = None,
      total_wavenumbers: int | None = None,
      spherical_harmonics_impl: SphericalHarmonicsImpl | None = None,
  ):
    """Constructs a `SphericalHarmonicGrid` from the `LonLatGrid`."""
    if longitude_wavenumbers is None:
      longitude_wavenumbers = self.longitude_wavenumbers
    if total_wavenumbers is None:
      total_wavenumbers = self.total_wavenumbers
    if spherical_harmonics_impl is None:
      spherical_harmonics_impl = self.spherical_harmonics_impl
    return SphericalHarmonicGrid(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=total_wavenumbers,
        longitude_offset=self.longitude_offset,
        radius=self.radius,
        spherical_harmonics_impl=spherical_harmonics_impl,
        longitude_nodes=self.longitude_nodes,
        latitude_nodes=self.latitude_nodes,
        latitude_spacing=self.latitude_spacing,
        spmd_mesh=self.spmd_mesh,
    )

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
        total_wavenumbers=ylm_grid.total_wavenumbers,
        longitude_wavenumbers=ylm_grid.longitude_wavenumbers,
        spherical_harmonics_impl=ylm_grid.spherical_harmonics_impl,
        spmd_mesh=ylm_grid.spmd_mesh,
    )

  @classmethod
  def construct(
      cls,
      max_wavenumber: int,
      gaussian_nodes: int,
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      radius: float = 1.0,
      spherical_harmonics_impl: SphericalHarmonicsImpl = FastSphericalHarmonics,
      spmd_mesh: jax.sharding.Mesh | None = None,
  ) -> LonLatGrid:
    """Constructs a `LonLatGrid` compatible with max_wavenumber and nodes.

    Args:
      max_wavenumber: maximum wavenumber to resolve.
      gaussian_nodes: number of nodes on the Gaussian grid between the equator
        and a pole.
      latitude_spacing: either 'gauss' or 'equiangular'. This determines the
        spacing of nodal grid points in the latitudinal (north-south) direction.
      longitude_offset: the value of the first longitude node, in radians.
      radius: radius of the sphere.
      spherical_harmonics_impl: class providing an implementation of spherical
        harmonics.
      spmd_mesh: optional SPMD mesh that informps how the grid is padded to have
        equal number of elements in each shard.

    Returns:
      Constructed LonLatGrid object.
    """
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=max_wavenumber + 1,
        total_wavenumbers=max_wavenumber + 2,
        longitude_nodes=4 * gaussian_nodes,
        latitude_nodes=2 * gaussian_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=spherical_harmonics_impl,
        radius=radius,
        spmd_mesh=spmd_mesh,
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
    return cls.construct(max_wavenumber=21, gaussian_nodes=16, **kwargs)

  @classmethod
  def T31(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=24, **kwargs)

  @classmethod
  def T42(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=42, gaussian_nodes=32, **kwargs)

  @classmethod
  def T85(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=85, gaussian_nodes=64, **kwargs)

  @classmethod
  def T106(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=106, gaussian_nodes=80, **kwargs)

  @classmethod
  def T119(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=119, gaussian_nodes=90, **kwargs)

  @classmethod
  def T170(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=170, gaussian_nodes=128, **kwargs)

  @classmethod
  def T213(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=213, gaussian_nodes=160, **kwargs)

  @classmethod
  def T340(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=340, gaussian_nodes=256, **kwargs)

  @classmethod
  def T425(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=425, gaussian_nodes=320, **kwargs)

  # TL* grids do not truncate any frequencies, and hence can only model linear
  # terms exactly. ECMWF used "TL" (truncated linear) grids for semi-Lagrangian
  # advection (which eliminates quadratic terms) up to 2016, which it switched
  # to "cubic" grids for resolutions above TL1279:
  # https://www.ecmwf.int/sites/default/files/elibrary/2016/17262-new-grid-ifs.pdf

  @classmethod
  def TL31(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=16, **kwargs)

  @classmethod
  def TL47(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=47, gaussian_nodes=24, **kwargs)

  @classmethod
  def TL63(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=63, gaussian_nodes=32, **kwargs)

  @classmethod
  def TL95(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=95, gaussian_nodes=48, **kwargs)

  @classmethod
  def TL127(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=127, gaussian_nodes=64, **kwargs)

  @classmethod
  def TL159(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=159, gaussian_nodes=80, **kwargs)

  @classmethod
  def TL179(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=179, gaussian_nodes=90, **kwargs)

  @classmethod
  def TL255(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=255, gaussian_nodes=128, **kwargs)

  @classmethod
  def TL639(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=639, gaussian_nodes=320, **kwargs)

  @classmethod
  def TL1279(cls, **kwargs) -> LonLatGrid:
    return cls.construct(max_wavenumber=1279, gaussian_nodes=640, **kwargs)

  def to_xarray(self) -> dict[str, xarray.Variable]:
    if self.spmd_mesh is not None:
      raise ValueError(
          'Conversion of LonLatGrid with SPMD mesh to xarray.DataArray is not'
          ' supported.'
      )
    variables = super().to_xarray()
    metadata = dict(
        total_wavenumbers=self.total_wavenumbers,
        longitude_wavenumbers=self.longitude_wavenumbers,
        radius=self.radius,
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
    total_wavenumbers = int(
        coords['longitude'].attrs.get('total_wavenumbers', 0)
    )
    longitude_wavenumbers = int(
        coords['longitude'].attrs.get('longitude_wavenumbers', 0)
    )
    result = cls(
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        radius=radius,
        total_wavenumbers=total_wavenumbers,
        longitude_wavenumbers=longitude_wavenumbers,
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
  longitude_offset: float = 0.0
  radius: float = 1.0
  spherical_harmonics_impl: SphericalHarmonicsImpl = dataclasses.field(
      default=FastSphericalHarmonics, kw_only=True
  )
  longitude_nodes: int = dataclasses.field(default=0, kw_only=True)
  latitude_nodes: int = dataclasses.field(default=0, kw_only=True)
  latitude_spacing: str = dataclasses.field(
      default='gauss', repr=False, kw_only=True
  )
  spmd_mesh: jax.sharding.Mesh | None = dataclasses.field(
      default=None, kw_only=True
  )
  _ylm_grid: spherical_harmonic.Grid = dataclasses.field(
      init=False, repr=False, compare=False
  )

  def __post_init__(self):
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=self.longitude_wavenumbers,
        total_wavenumbers=self.total_wavenumbers,
        longitude_offset=self.longitude_offset,
        radius=self.radius,
        spherical_harmonics_impl=self.spherical_harmonics_impl,
        longitude_nodes=self.longitude_nodes,
        latitude_nodes=self.latitude_nodes,
        latitude_spacing=self.latitude_spacing,
        spmd_mesh=self.spmd_mesh,
    )
    object.__setattr__(self, '_ylm_grid', ylm_grid)

  @property
  def ylm_grid(self) -> spherical_harmonic.Grid:
    return self._ylm_grid

  @property
  def dims(self):
    return ('longitude_wavenumber', 'total_wavenumber')

  @property
  def shape(self) -> tuple[int, ...]:
    return self.ylm_grid.modal_shape

  @property
  def fields(self):
    return {
        k: cx.wrap(v, cx.SelectedAxis(self, i))
        for i, (k, v) in enumerate(zip(self.dims, self.ylm_grid.modal_axes))
    }

  def to_lon_lat_grid(
      self,
      longitude_nodes: int | None = None,
      latitude_nodes: int | None = None,
      latitude_spacing: str | None = None,
  ):
    """Constructs a `LonLatGrid` from the `SphericalHarmonicGrid`."""
    if longitude_nodes is None:
      longitude_nodes = self.longitude_nodes
    if latitude_nodes is None:
      latitude_nodes = self.latitude_nodes
    if latitude_spacing is None:
      latitude_spacing = self.latitude_spacing
    return LonLatGrid(
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=self.longitude_offset,
        radius=self.radius,
        spherical_harmonics_impl=self.spherical_harmonics_impl,
        longitude_wavenumbers=self.longitude_wavenumbers,
        total_wavenumbers=self.total_wavenumbers,
        spmd_mesh=self.spmd_mesh,
    )

  @classmethod
  def from_dinosaur_grid(
      cls,
      ylm_grid: spherical_harmonic.Grid,
  ):
    return cls(
        longitude_wavenumbers=ylm_grid.longitude_wavenumbers,
        total_wavenumbers=ylm_grid.total_wavenumbers,
        longitude_offset=ylm_grid.longitude_offset,
        radius=ylm_grid.radius,
        spherical_harmonics_impl=ylm_grid.spherical_harmonics_impl,
        longitude_nodes=ylm_grid.longitude_nodes,
        latitude_nodes=ylm_grid.latitude_nodes,
        latitude_spacing=ylm_grid.latitude_spacing,
        spmd_mesh=ylm_grid.spmd_mesh,
    )

  @classmethod
  def with_wavenumbers(
      cls,
      longitude_wavenumbers: int,
      dealiasing: str = 'quadratic',
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      spherical_harmonics_impl: SphericalHarmonicsImpl = (
          FastSphericalHarmonics
      ),
      radius: float = 1.0,
      spmd_mesh: jax.sharding.Mesh | None = None,
  ) -> SphericalHarmonicGrid:
    """Constructs a `SphericalHarmonicGrid` by specifying only wavenumbers."""
    # The number of nodes is chosen for de-aliasing.
    order = {'linear': 2, 'quadratic': 3, 'cubic': 4}[dealiasing]
    longitude_nodes = order * longitude_wavenumbers + 1
    latitude_nodes = math.ceil(longitude_nodes / 2)
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=longitude_wavenumbers + 1,
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=spherical_harmonics_impl,
        radius=radius,
        spmd_mesh=spmd_mesh,
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  @classmethod
  def construct(
      cls,
      max_wavenumber: int,
      gaussian_nodes: int,
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      radius: float = 1.0,
      spherical_harmonics_impl: SphericalHarmonicsImpl = (
          FastSphericalHarmonics
      ),
      spmd_mesh: jax.sharding.Mesh | None = None,
  ) -> SphericalHarmonicGrid:
    """Constructs a `SphericalHarmonicGrid` with max_wavenumber.

    Args:
      max_wavenumber: maximum wavenumber to resolve.
      gaussian_nodes: number of nodes on the Gaussian grid between the equator
        and a pole.
      latitude_spacing: either 'gauss' or 'equiangular'. This determines the
        spacing of nodal grid points in the latitudinal (north-south) direction.
      longitude_offset: the value of the first longitude node, in radians.
      radius: radius of the sphere.
      spherical_harmonics_impl: class providing an implementation of spherical
        harmonics.
      spmd_mesh: optional mesh to use for SPMD sharding.

    Returns:
      Constructed SphericalHarmonicGrid object.
    """
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=max_wavenumber + 1,
        total_wavenumbers=max_wavenumber + 2,
        longitude_nodes=4 * gaussian_nodes,
        latitude_nodes=2 * gaussian_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=spherical_harmonics_impl,
        radius=radius,
        spmd_mesh=spmd_mesh,
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
    return cls.construct(max_wavenumber=21, gaussian_nodes=16, **kwargs)

  @classmethod
  def T31(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=24, **kwargs)

  @classmethod
  def T42(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=42, gaussian_nodes=32, **kwargs)

  @classmethod
  def T85(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=85, gaussian_nodes=64, **kwargs)

  @classmethod
  def T106(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=106, gaussian_nodes=80, **kwargs)

  @classmethod
  def T119(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=119, gaussian_nodes=90, **kwargs)

  @classmethod
  def T170(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=170, gaussian_nodes=128, **kwargs)

  @classmethod
  def T213(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=213, gaussian_nodes=160, **kwargs)

  @classmethod
  def T340(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=340, gaussian_nodes=256, **kwargs)

  @classmethod
  def T425(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=425, gaussian_nodes=320, **kwargs)

  # TL* grids do not truncate any frequencies, and hence can only model linear
  # terms exactly. ECMWF used "TL" (truncated linear) grids for semi-Lagrangian
  # advection (which eliminates quadratic terms) up to 2016, which it switched
  # to "cubic" grids for resolutions above TL1279:
  # https://www.ecmwf.int/sites/default/files/elibrary/2016/17262-new-grid-ifs.pdf

  @classmethod
  def TL31(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=31, gaussian_nodes=16, **kwargs)

  @classmethod
  def TL47(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=47, gaussian_nodes=24, **kwargs)

  @classmethod
  def TL63(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=63, gaussian_nodes=32, **kwargs)

  @classmethod
  def TL95(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=95, gaussian_nodes=48, **kwargs)

  @classmethod
  def TL127(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=127, gaussian_nodes=64, **kwargs)

  @classmethod
  def TL159(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=159, gaussian_nodes=80, **kwargs)

  @classmethod
  def TL179(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=179, gaussian_nodes=90, **kwargs)

  @classmethod
  def TL255(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=255, gaussian_nodes=128, **kwargs)

  @classmethod
  def TL639(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=639, gaussian_nodes=320, **kwargs)

  @classmethod
  def TL1279(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=1279, gaussian_nodes=640, **kwargs)

  def to_xarray(self) -> dict[str, xarray.Variable]:
    if self.spmd_mesh is not None:
      raise ValueError(
          'Conversion of SphericalHarmonicGrid with SPMD mesh to'
          ' xarray.DataArray is not supported.'
      )
    variables = super().to_xarray()
    metadata = dict(
        longitude_offset=self.longitude_offset,
        longitude_nodes=self.longitude_nodes,
        latitude_nodes=self.latitude_nodes,
        latitude_spacing=self.latitude_spacing,
        radius=self.radius,
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
    longitude_offset = float(
        coords['longitude_wavenumber'].attrs.get('longitude_offset', 0.0)
    )
    longitude_nodes = int(
        coords['longitude_wavenumber'].attrs.get('longitude_nodes', 0)
    )
    latitude_nodes = int(
        coords['longitude_wavenumber'].attrs.get('latitude_nodes', 0)
    )
    latitude_spacing = str(
        coords['longitude_wavenumber'].attrs.get('latitude_spacing', 'gauss')
    )

    candidate = cls(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=coords.sizes['total_wavenumber'],
        radius=radius,
        longitude_offset=longitude_offset,
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
    )

    expected = candidate._ylm_grid.modal_axes[0]
    got = coords['longitude_wavenumber'].data
    if not np.array_equal(expected, got):
      return cx.NoCoordinateMatch(
          'inferred longitude wavenumbers do not match coordinate data:'
          f' {expected} vs {got}. Perhaps you attempted to restore coordinate '
          ' data from FastSphericalHarmonics, which does not support '
          'restoration?'
      )

    expected = candidate._ylm_grid.modal_axes[1]
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
      return cx.NoCoordinateMatch(f"dimension {dim!r} != 'sigma'")

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
          f"dimension {dim!r} is not 'pressure' or 'level'"
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
      return cx.NoCoordinateMatch(f"dimension {dim!r} != 'layer_index'")

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

  @property
  def dinosaur_coords(self):
    """Returns the CoordinateSystem object from the Dinosaur package."""
    # TODO(dkochkov) Either make spmd_mesh an argument or ideally add
    # ShardingMesh object to the new API to hold sharding information.
    spmd_mesh = None  # make this an argument and change property to a method.
    horizontal, vertical = self.horizontal, self.vertical
    horizontal = horizontal.ylm_grid
    if isinstance(vertical, SigmaLevels):
      vertical = vertical.sigma_levels
    elif isinstance(vertical, PressureLevels):
      vertical = vertical.pressure_levels
    elif isinstance(vertical, LayerLevels):
      pass
    else:
      raise ValueError(f'Unsupported vertical {vertical=}')
    return dinosaur_coordinates.CoordinateSystem(
        horizontal=horizontal, vertical=vertical, spmd_mesh=spmd_mesh
    )

  @property
  def dinosaur_grid(self):
    return self.dinosaur_coords.horizontal

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


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class CoordinateShard(cx.Coordinate):
  """Coordinate that represents a shard of a full coordinate system.

  This class helps tag array shards that represent a subset of values of a
  `full_coordinare`. The most common use case is during data ingestion where
  shards are read from disk and are not yet converted to a sharded array.
  """

  coordinate: cx.Coordinate
  spmd_mesh_shape: collections.OrderedDict[str, int]
  dimension_partitions: dict[str, None | str | tuple[str, ...]]

  def __hash__(self):
    """Hash implementation that avoids un-hashable mesh and partitions attrs."""
    spmd_mesh_tuple = tuple(self.spmd_mesh_shape.items())
    partitions_tuple = tuple(self.dimension_partitions.items())
    return hash((self.coordinate, spmd_mesh_tuple, partitions_tuple))

  def __eq__(self, other):
    return (
        isinstance(other, CoordinateShard)
        and self.coordinate == other.coordinate
        and self.spmd_mesh_shape == other.spmd_mesh_shape
        and self.dimension_partitions == other.dimension_partitions
    )

  @property
  def dims(self) -> tuple[str, ...]:
    return self.coordinate.dims  # sharding only affects the shape.

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the coordinate."""
    dims, full_shape = self.dims, self.coordinate.shape
    shard_shape = []
    for dim, full_size in zip(dims, full_shape, strict=True):
      axes = self.dimension_partitions.get(dim, None)
      if not isinstance(axes, tuple):
        axes = (axes,)
      n_shards = math.prod(self.spmd_mesh_shape.get(axis, 1) for axis in axes)
      size, reminder = divmod(full_size, n_shards)
      if reminder:
        raise ValueError(
            f'Dimension {dim} has size {full_size} which is not divisible by'
            f' the number of shards {n_shards}.'
        )
      shard_shape.append(size)
    return tuple(shard_shape)

  @property
  def fields(self) -> dict[str, cx.Field]:
    """A maps from field names to their values."""
    return {}

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.NoCoordinateMatch:
    return cx.NoCoordinateMatch(
        'CoordinateShard cannot be deterministically inferred from xarray.'
    )


def get_unsharded(coordinate: cx.Coordinate) -> cx.Coordinate:
  """Returns the unsharded coordinate of a CoordinateShard."""
  unsharded = []
  for c in cx.canonicalize_coordinates(coordinate):
    unsharded.append(c.coordinate if isinstance(c, CoordinateShard) else c)
  return cx.compose_coordinates(*unsharded)


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
