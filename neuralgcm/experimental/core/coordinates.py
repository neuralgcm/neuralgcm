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
import functools
from typing import Any, Iterable, Literal, Self, Sequence, TYPE_CHECKING, cast

import coordax as cx
import coordax.experimental
from dinosaur import coordinate_systems as dinosaur_coordinates
from dinosaur import fourier
from dinosaur import hybrid_coordinates
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
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
# factor for globally constant conversion from real to YLM basis.
_YLM_CONSTANT_NORMALIZATION_FACTOR = 3.5449077


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
    return {'timedelta': cx.field(self.deltas, self)}

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str, Any], set[str]]:
    return coordax.experimental.map_indexers_using_ticks(
        self, indexers, ticks_are_sorted=True, method=method
    )

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    key = self.dims[0] if self.dims[0] in indexers else self
    indexer = indexers[key]
    if isinstance(indexer, int):
      return cx.Scalar()

    return TimeDelta(self.deltas[indexer])

  def to_xarray(self) -> dict[str, xarray.Variable]:
    variables = super().to_xarray()
    variables['timedelta'] = xarray.Variable(('timedelta',), self.deltas)
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'timedelta':
      return cx.coords.NoCoordinateMatch(f'dimension {dim!r} != "timedelta"')
    if 'timedelta' not in coords:
      return cx.coords.NoCoordinateMatch(
          'no associated coordinate for timedelta'
      )

    data = coords['timedelta'].data
    if data.ndim != 1:
      return cx.coords.NoCoordinateMatch(
          'timedelta coordinate is not a 1D array'
      )

    if not np.issubdtype(data.dtype, np.timedelta64):
      return cx.coords.NoCoordinateMatch(
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
    dts = [str(x) for x in self.deltas.tolist()]
    if len(dts) < 6:
      deltas = '[' + ', '.join(dts) + ']'
    else:
      deltas = '[' + ', '.join(dts[:2])
      deltas += ', ..., '
      deltas += ', '.join(dts[-2:]) + ']'
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


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LonLatGrid(cx.Coordinate):
  """Coordinates that discretize data as point values on lon-lat grid."""

  longitude_nodes: int
  latitude_nodes: int
  latitude_spacing: str = 'gauss'
  longitude_offset: float = 0.0
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
    lons = np.rad2deg(np.pad(self._ylm_grid.longitudes, (0, lon_pad)))
    lats = np.rad2deg(np.pad(self._ylm_grid.latitudes, (0, lat_pad)))
    return {
        'longitude': cx.field(lons, cx.SelectedAxis(self, axis=0)),
        'latitude': cx.field(lats, cx.SelectedAxis(self, axis=1)),
    }

  @functools.cached_property
  def cos_lat(self) -> cx.Field:
    padded_lats = np.pad(self._ylm_grid.latitudes, (0, self.lon_lat_padding[1]))
    return cx.field(np.cos(padded_lats), cx.SelectedAxis(self, axis=1))

  def _remap_indexers_to_labeled_axes(
      self, indexers: dict[str | cx.Coordinate, Any]
  ) -> tuple[
      dict[str | cx.Coordinate, Any], tuple[cx.LabeledAxis, cx.LabeledAxis]
  ]:
    """Remaps indexers to labeled axes."""
    longitudes_axis = cx.LabeledAxis('longitude', self.fields['longitude'].data)
    latitudes_axis = cx.LabeledAxis('latitude', self.fields['latitude'].data)
    idx_remap = {self.axes[0]: longitudes_axis, self.axes[1]: latitudes_axis}
    axes = (longitudes_axis, latitudes_axis)
    return {idx_remap.get(k, k): v for k, v in indexers.items()}, axes

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: coordax.experimental.SelMethod = None,
  ) -> tuple[dict[str, Any], set[str]]:
    # When slicing by longitude/latitude, we convert respective axes to
    # LabeledAxis, since sliced LonLatGrid is not necessarily a LonLatGrid.

    # We remap self.axes to LabeledAxes to reuse 1d indexing logic. To ensure
    # that downstream isel works correctly on LabledAxes and indexer validation
    # passes we use to axis name notation for mapped indexers.
    indexers, (lon_ax, lat_ax) = self._remap_indexers_to_labeled_axes(indexers)
    mapped, consumed = {}, set()
    if 'longitude' in indexers or lon_ax in indexers:
      key = 'longitude' if 'longitude' in indexers else lon_ax
      sub_mapped, sub_consumed = coordax.experimental.map_indexers_using_ticks(
          lon_ax,
          indexers,
          ticks_are_sorted=True,
          method=method,
      )
      mapped.update({'longitude': sub_mapped[key]})
      consumed.update(sub_consumed)

    if 'latitude' in indexers or lat_ax in indexers:
      key = 'latitude' if 'latitude' in indexers else lat_ax
      sub_mapped, sub_consumed = coordax.experimental.map_indexers_using_ticks(
          lat_ax,
          indexers,
          ticks_are_sorted=True,
          method=method,
      )
      mapped.update({'latitude': sub_mapped[key]})
      consumed.update(sub_consumed)
    # Remap LabeledAxes back to self.axes.
    consumed_remap = {l_ax: ax for l_ax, ax in zip((lon_ax, lat_ax), self.axes)}
    consumed = {consumed_remap.get(ax, ax) for ax in consumed}
    return mapped, consumed

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    indexers, axes = self._remap_indexers_to_labeled_axes(indexers)
    return cx.coords.compose(*axes).isel(indexers)

  def integrate(
      self,
      x: cx.Field,
      dims: str | Sequence[str] = ('longitude', 'latitude'),
      radius: float = 1.0,
  ) -> cx.Field:
    """Integrates `x` over dimensions in `dims` using quadrature.

    Args:
      x: Field to integrate.
      dims: Dimensions to integrate. Can be 'longitude', 'latitude', or a
        sequence of thereof.
      radius: Radius of sphere, defaults to 1.0.

    Returns:
      Field with all or a subset of dimensions of LonLatGrid integrated out.
    """
    _, lon_w = fourier.quadrature_nodes(self.longitude_nodes)
    _, lat_w = spherical_harmonic.get_latitude_nodes(
        self.latitude_nodes, self.latitude_spacing
    )

    def _integrate_lon(array):
      return radius**2 * jnp.sum(array[: self.longitude_nodes] * lon_w)

    def _integrate_lat(array):
      return radius**2 * jnp.sum(array[: self.latitude_nodes] * lat_w)

    def _integrate(array):
      unpadded = array[: self.longitude_nodes, : self.latitude_nodes]
      return radius**2 * jnp.sum(unpadded * lat_w * lon_w)

    dims_tuple = tuple(sorted({dims} if isinstance(dims, str) else set(dims)))

    match dims_tuple:
      case ('longitude',):
        return cx.cmap(_integrate_lon)(x.untag(cx.SelectedAxis(self, axis=0)))
      case ('latitude',):
        return cx.cmap(_integrate_lat)(x.untag(cx.SelectedAxis(self, axis=1)))
      case ('latitude', 'longitude'):
        return cx.cmap(_integrate)(x.untag(self))
      case _:
        raise ValueError(
            'dims must be "longitude", "latitude", or a sequence '
            f'thereof, got {dims}'
        )

  def mean(
      self,
      x: cx.Field,
      dims: str | Sequence[str] = ('longitude', 'latitude'),
  ) -> cx.Field:
    """Computes mean of `x` over dimensions in `dims`.

    Args:
      x: Field to average.
      dims: Dimensions to average over. Can be 'longitude', 'latitude', or a
        sequence of thereof.

    Returns:
      Field with all or a subset of dimensions of LonLatGrid averaged out.
    """
    integral = self.integrate(x, dims, radius=1.0)
    dims_tuple = tuple(sorted({dims} if isinstance(dims, str) else set(dims)))
    match dims_tuple:
      case ('longitude',):
        return integral / (2 * np.pi)
      case ('latitude',):
        return integral / 2.0
      case ('latitude', 'longitude'):
        return integral / (4 * np.pi)
      case _:
        raise ValueError(
            'dims must be "longitude", "latitude", or a sequence '
            f'thereof, got {dims}'
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
        lon_lat_padding=ylm_grid.nodal_padding,
    )

  @classmethod
  def construct(
      cls,
      gaussian_nodes: int,
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      mesh: parallelism.Mesh | None = None,
      partition_schema_key: str | None = None,
  ) -> LonLatGrid:
    """Constructs a `LonLatGrid` with specified number of latitude nodes.

    Args:
      gaussian_nodes: number of nodes between the equator and a pole.
      latitude_spacing: either 'gauss' or 'equiangular'. This determines the
        spacing of grid points in the latitudinal (north-south) direction.
      longitude_offset: the value of the first longitude node, in radians.
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
        lon_lat_padding=self.lon_lat_padding,
    )
    variables['longitude'].attrs = metadata
    variables['latitude'].attrs = metadata
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    if dims[:2] != ('longitude', 'latitude'):
      return cx.coords.NoCoordinateMatch(
          "leading dimensions are not ('longitude', 'latitude')"
      )

    if coords['longitude'].dims != ('longitude',):
      return cx.coords.NoCoordinateMatch('longitude is not a 1D coordinate')

    if coords['latitude'].dims != ('latitude',):
      return cx.coords.NoCoordinateMatch('latitude is not a 1D coordinate')

    lon = coords['longitude'].data
    lat = coords['latitude'].data

    if lon.max() < 2 * np.pi:
      return cx.coords.NoCoordinateMatch(
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

    lon_lat_padding = coords['longitude'].attrs.get('lon_lat_padding', (0, 0))
    result = cls(
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        lon_lat_padding=tuple(int(x) for x in lon_lat_padding),
    )
    result_lat = np.rad2deg(result._ylm_grid.latitudes)
    if not np.allclose(result_lat, lat, atol=1e-3):
      return cx.coords.NoCoordinateMatch(
          f'inferred latitudes with spacing={latitude_spacing!r} do not '
          f' match coordinate data: {result_lat} vs {lat}'
      )

    result_lon = np.rad2deg(result._ylm_grid.longitudes)
    if not np.allclose(result_lon, lon, atol=1e-3):
      return cx.coords.NoCoordinateMatch(
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
    ms = np.pad(unpadded_ms, (0, m_pad))
    ls = np.pad(unpadded_ls, (0, l_pad))
    axes_fields = {
        k: cx.field(v, cx.SelectedAxis(self, i))
        for i, (k, v) in enumerate(zip(self.dims, [ms, ls]))
    }
    unpadded_mask = self._ylm_grid.mask
    mask = np.pad(unpadded_mask, ((0, m_pad), (0, l_pad)))
    mask_field = cx.field(mask, self)
    return axes_fields | {'mask': mask_field}

  def _remap_indexers_to_labeled_axes(
      self, indexers: dict[str | cx.Coordinate, Any]
  ) -> tuple[
      dict[str | cx.Coordinate, Any], tuple[cx.LabeledAxis, cx.LabeledAxis]
  ]:
    """Remaps indexers to labeled axes."""
    ms_ax = cx.LabeledAxis(
        'longitude_wavenumber', self.fields['longitude_wavenumber'].data
    )
    ls_ax = cx.LabeledAxis(
        'total_wavenumber', self.fields['total_wavenumber'].data
    )
    idx_remap = {self.axes[0]: ms_ax, self.axes[1]: ls_ax}
    axes = (ms_ax, ls_ax)
    return {idx_remap.get(k, k): v for k, v in indexers.items()}, axes

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str, Any], set[str]]:
    mapped, consumed = {}, set()
    # We remap self.axes to LabeledAxes to reuse 1d indexing logic. To ensure
    # that downstream isel works correctly on LabledAxes and indexer validation
    # passes we use to axis name notation for mapped indexers.
    indexers, (ms_ax, ls_ax) = self._remap_indexers_to_labeled_axes(indexers)
    if 'longitude_wavenumber' in indexers or ms_ax in indexers:
      key = ms_ax if ms_ax in indexers else 'longitude_wavenumber'
      sub_mapped, sub_consumed = coordax.experimental.map_indexers_using_ticks(
          ms_ax,
          indexers,
          ticks_are_sorted=True,
          method=method,
      )
      mapped.update({'longitude_wavenumber': sub_mapped[key]})
      consumed.update(sub_consumed)

    if 'total_wavenumber' in indexers or ls_ax in indexers:
      key = ls_ax if ls_ax in indexers else 'total_wavenumber'
      sub_mapped, sub_consumed = coordax.experimental.map_indexers_using_ticks(
          ls_ax,
          indexers,
          ticks_are_sorted=True,
          method=method,
      )
      mapped.update({'total_wavenumber': sub_mapped[key]})
      consumed.update(sub_consumed)
    # Remap consumed ms, ls axes back to self to ensure keys are accounted for.
    consumed_remap = {l_ax: ax for l_ax, ax in zip((ms_ax, ls_ax), self.axes)}
    consumed = {consumed_remap.get(ax, ax) for ax in consumed}
    return mapped, consumed

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    indexers, (ms_ax, ls_ax) = self._remap_indexers_to_labeled_axes(indexers)
    return cx.coords.compose(ms_ax, ls_ax).isel(indexers)

  def add_constant(
      self,
      x: cx.Field,
      c: float | jax.Array | cx.Field,
  ) -> cx.Field:
    """Adds the constant `c` to the field `x` in the spectral basis."""
    if not cx.is_field(c):
      c = cx.field(jnp.squeeze(c))
    assert isinstance(c, cx.Field)  # make pytype happy.
    if c.positional_shape:
      raise ValueError(
          f'Adding non-scalar constants without axes is not supported. {c=}'
      )
    if any(d in self.dims for d in c.dims):
      raise ValueError(
          f'Coordinate {c} cannot have any of the dimensions {self.dims}.'
      )
    if not set(c.dims).issubset(x.dims):
      raise ValueError(
          'Introduction of new axes via add_constant is not supported. For'
          f' that coordinate {c} must have dimensions that are subset of the'
          f' dimensions {x.dims=} but it is not.'
      )

    def add_fn(x, y):
      x = jnp.asarray(x)
      return x.at[0, 0].add(_YLM_CONSTANT_NORMALIZATION_FACTOR * y)

    x = x.untag(self)
    return cx.cmap(add_fn, x.named_axes)(x, c).tag(self)

  def clip_wavenumbers(self, x: cx.Field, n: int) -> cx.Field:
    if n <= 0:
      raise ValueError(f'`n` must be >= 0; got {n}.')

    def clip(x):
      # Multiplication by the mask is significantly faster than directly using
      # `x.at[..., -n:].set(0)`
      num_zeros = n + self.wavenumber_padding[-1]
      mask = jnp.ones(self.shape[-1], x.dtype).at[-num_zeros:].set(0)
      return x * mask

    return cx.cpmap(clip)(x.untag(self)).tag(self)

  def __lt__(self, other: SphericalHarmonicGrid) -> bool:
    """Custom comparison operator for sorting SphericalHarmonicGrids."""

    # Implementing __lt__ enables SphericalHarmonicGrid to be keys in a dict
    # and be compatible with jax.tree. operations.
    def _to_tuple(x):
      xt = (x.total_wavenumbers, x.longitude_wavenumbers, x.wavenumber_padding)
      return xt

    if not isinstance(other, SphericalHarmonicGrid):
      return NotImplemented
    # Sort by total, then longitude, then padding.
    self_compare_values_in_order = _to_tuple(self)
    other_compare_values_in_order = _to_tuple(other)
    return self_compare_values_in_order < other_compare_values_in_order

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
        spherical_harmonics_method=method_name,
        wavenumber_padding=ylm_grid.modal_padding,
    )

  @classmethod
  def with_wavenumbers(
      cls,
      longitude_wavenumbers: int,
      spherical_harmonics_method: SphericalHarmonicsMethodNames = 'fast',
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
        spmd_mesh=_mesh_to_dinosaur_spmd_mesh(dims, mesh, partition_schema_key),
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  @classmethod
  def construct(
      cls,
      max_wavenumber: int,
      spherical_harmonics_method: SphericalHarmonicsMethodNames = 'fast',
      mesh: parallelism.Mesh | None = None,
      partition_schema_key: str | None = None,
  ) -> SphericalHarmonicGrid:
    """Constructs a `SphericalHarmonicGrid` with max_wavenumber.

    Args:
      max_wavenumber: maximum wavenumber to resolve.
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
        wavenumber_padding=self.wavenumber_padding,
        spherical_harmonics_method=self.spherical_harmonics_method,
    )
    variables['longitude_wavenumber'].attrs = metadata
    variables['total_wavenumber'].attrs = metadata
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:

    if dims[:2] != ('longitude_wavenumber', 'total_wavenumber'):
      return cx.coords.NoCoordinateMatch(
          "leading dimensions are not ('longitude_wavenumber',"
          " 'total_wavenumber')"
      )

    if coords['longitude_wavenumber'].dims != ('longitude_wavenumber',):
      return cx.coords.NoCoordinateMatch(
          'longitude_wavenumber is not a 1D coordinate'
      )

    if coords['total_wavenumber'].dims != ('total_wavenumber',):
      return cx.coords.NoCoordinateMatch(
          'total_wavenumber is not a 1D coordinate'
      )

    longitude_wavenumbers = (coords.sizes['longitude_wavenumber'] + 1) // 2
    wavenumber_padding = coords['longitude_wavenumber'].attrs.get(
        'wavenumber_padding', (0, 0)
    )
    spherical_harmonics_method = coords['longitude_wavenumber'].attrs.get(
        'spherical_harmonics_method', 'fast'
    )
    candidate = cls(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=coords.sizes['total_wavenumber'],
        spherical_harmonics_method=spherical_harmonics_method,
        wavenumber_padding=tuple(int(x) for x in wavenumber_padding),
    )

    expected = candidate.fields['longitude_wavenumber'].data
    got = coords['longitude_wavenumber'].data
    if not np.array_equal(expected, got):
      return cx.coords.NoCoordinateMatch(
          'inferred longitude wavenumbers do not match coordinate data:'
          f' {expected} vs {got}. Perhaps you attempted to restore coordinate '
          ' data from FastSphericalHarmonics, which does not support '
          'restoration?'
      )

    expected = candidate.fields['total_wavenumber'].data
    got = coords['total_wavenumber'].data
    if not np.array_equal(expected, got):
      return cx.coords.NoCoordinateMatch(
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
    boundaries, centers = self.boundaries, self.sigma_levels.centers
    return {
        'sigma': cx.field(centers, self),
        'sigma_boundaries': cx.field(boundaries, SigmaBoundaries(boundaries)),
    }

  @functools.cached_property
  def sigma_boundaries(self) -> SigmaBoundaries:
    return SigmaBoundaries(boundaries=self.boundaries)

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str, Any], set[str]]:
    sigma_labeled = cx.LabeledAxis(self.dims[0], self.sigma_levels.centers)
    if self in indexers:
      indexers[sigma_labeled] = indexers.pop(self)
    if self.dims[0] in indexers or sigma_labeled in indexers:
      key = sigma_labeled if sigma_labeled in indexers else self.dims[0]
      i_indexers, consumed = sigma_labeled.map_indexers(indexers, method)
      i_indexers = {self.dims[0]: i_indexers[key]}
    else:
      i_indexers = {}
      consumed = set()
    remap_consumed = {sigma_labeled: self}
    consumed = {remap_consumed.get(k, k) for k in consumed}
    return i_indexers, consumed

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    sigma_labeled = cx.LabeledAxis(self.dims[0], self.sigma_levels.centers)
    if self in indexers:
      indexers[sigma_labeled] = indexers.pop(self)
    return sigma_labeled.isel(indexers)

  def integrate(self, x: cx.Field) -> cx.Field:
    """Integrates `x` over the sigma levels."""
    sigma_integrate = functools.partial(
        sigma_coordinates.sigma_integral,
        coordinates=self.sigma_levels,
        axis=0,
        keepdims=False,
    )
    return cx.cmap(sigma_integrate)(x.untag(self))

  def integrate_over_pressure(
      self,
      x: cx.Field,
      surface_pressure: cx.Field,
      sim_units: units.SimUnits | None = None,
  ) -> cx.Field:
    """Integrates `x` over the sigma levels weighted by surface pressure."""
    del sim_units  # unused
    return surface_pressure * self.integrate(x)

  def integrate_cumulative(
      self,
      x: cx.Field,
      downward: bool = True,
      cumsum_method: str = 'dot',
      sharding: jax.sharding.NamedSharding | None = None,
  ) -> cx.Field:
    """Integrates `x` over the sigma levels."""
    sigma_cumulative_integrate = functools.partial(
        sigma_coordinates.cumulative_sigma_integral,
        coordinates=self.sigma_levels,
        axis=0,
        downward=downward,
        cumsum_method=cumsum_method,
        sharding=sharding,
    )
    return cx.cpmap(sigma_cumulative_integrate)(x.untag(self)).tag(self)

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

  def to_sigma_boundaries(self) -> SigmaBoundaries:
    return SigmaBoundaries(boundaries=self.boundaries)

  @classmethod
  def from_sigma_boundaries(cls, sigma_boundaries: SigmaBoundaries) -> Self:
    return cls(boundaries=sigma_boundaries.boundaries)

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
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'sigma':
      return cx.coords.NoCoordinateMatch(f'dimension {dim!r} != "sigma"')

    if coords['sigma'].ndim != 1:
      return cx.coords.NoCoordinateMatch('sigma coordinate is not a 1D array')

    centers = coords['sigma'].data
    candidate = cls.from_centers(centers)
    actual_centers = candidate.sigma_levels.centers
    if not np.array_equal(actual_centers, centers):
      return cx.coords.NoCoordinateMatch(
          'inferred sigma boundaries do not exactly match coordinate data:'
          f' {actual_centers} vs {centers}.'
      )
    return candidate


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SigmaBoundaries(SigmaLevels):
  """Coordinate that represents boundaries of SigmaLevels."""

  @property
  def dims(self):
    return ('sigma_boundaries',)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.boundaries.shape

  @property
  def fields(self):
    boundaries, centers = self.boundaries, self.sigma_levels.centers
    return {
        'sigma_boundaries': cx.field(boundaries, self),
        'sigma': cx.field(centers, SigmaLevels(boundaries)),
    }

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str, Any], set[str]]:
    sigma_labeled = cx.LabeledAxis(self.dims[0], self.boundaries)
    if self in indexers:
      indexers[sigma_labeled] = indexers.pop(self)
    i_indexers, consumed = sigma_labeled.map_indexers(indexers, method)
    remap_consumed = {sigma_labeled: self}
    consumed = {remap_consumed.get(k, k) for k in consumed}
    return i_indexers, consumed

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    sigma_labeled = cx.LabeledAxis(self.dims[0], self.boundaries)
    if self in indexers:
      indexers[sigma_labeled] = indexers.pop(self)
    return sigma_labeled.isel(indexers)

  def __eq__(self, other):
    return (
        isinstance(other, SigmaBoundaries)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash((self._components(), 'boundaries'))

  def to_sigma_levels(self) -> SigmaLevels:
    return SigmaLevels(boundaries=self.boundaries)

  @classmethod
  def from_sigma_levels(cls, sigma_levels: SigmaLevels) -> Self:
    return cls(boundaries=sigma_levels.boundaries)

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'sigma_boundaries':
      return cx.coords.NoCoordinateMatch(
          f'dimension {dim!r} != "sigma_boundaries"'
      )
    sigma_dim = dim.removesuffix('_boundaries')
    return cls.from_sigma_levels(SigmaLevels.from_xarray((sigma_dim,), coords))


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
    return {'pressure': cx.field(self.centers, self)}

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str, Any], set[str]]:
    return coordax.experimental.map_indexers_using_ticks(
        self,
        indexers,
        ticks_are_sorted=True,
        method=method,
    )

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    key = self.dims[0] if self.dims[0] in indexers else self
    indexer = indexers[key]
    if isinstance(indexer, int):
      return cx.Scalar()

    return PressureLevels(self.centers[indexer])

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
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim not in {'level', 'pressure'}:
      return cx.coords.NoCoordinateMatch(
          f'dimension {dim!r} is not "pressure" or "level"'
      )
    if coords[dim].ndim != 1:
      return cx.coords.NoCoordinateMatch(
          'pressure coordinate is not a 1D array'
      )
    centers = coords[dim].data
    if not 0 < centers[0] < 100:
      return cx.coords.NoCoordinateMatch(
          f'pressure levels must start between 0 and 100, got: {centers}'
      )
    if not 900 < centers[-1] < 1025:
      return cx.coords.NoCoordinateMatch(
          f'pressure levels must end between 900 and 1025, got: {centers}'
      )
    return cls(centers=centers)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class HybridLevels(cx.Coordinate):
  """Coordinates with hybrid sigma-pressure levels."""

  a_boundaries: np.ndarray
  b_boundaries: np.ndarray
  hybrid_levels: hybrid_coordinates.HybridCoordinates = dataclasses.field(
      init=False, repr=False, compare=False
  )

  def __init__(
      self,
      a_boundaries: Iterable[float] | np.ndarray,
      b_boundaries: Iterable[float] | np.ndarray,
  ):
    a_boundaries = np.asarray(a_boundaries, np.float32)
    b_boundaries = np.asarray(b_boundaries, np.float32)
    object.__setattr__(self, 'a_boundaries', a_boundaries)
    object.__setattr__(self, 'b_boundaries', b_boundaries)
    self.__post_init__()

  def __post_init__(self):
    hybrid_levels = hybrid_coordinates.HybridCoordinates(
        a_boundaries=self.a_boundaries, b_boundaries=self.b_boundaries
    )
    object.__setattr__(self, 'hybrid_levels', hybrid_levels)

  @property
  def dims(self):
    return ('hybrid',)

  @property
  def shape(self) -> tuple[int, ...]:
    return (self.hybrid_levels.layers,)

  @property
  def fields(self):
    # Use index starting at 1 to match ECMWF's IFS convention.
    a = (self.a_boundaries[:-1] + self.a_boundaries[1:]) / 2.0
    b = (self.b_boundaries[:-1] + self.b_boundaries[1:]) / 2.0
    return {
        'hybrid': cx.field(np.arange(1, self.shape[0] + 1), self),
        'a': cx.field(a, self),
        'b': cx.field(b, self),
    }

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    hybrid_labeled = cx.LabeledAxis(self.dims[0], self.fields['hybrid'].data)
    if self in indexers:
      indexers[hybrid_labeled] = indexers.pop(self)
    return hybrid_labeled.isel(indexers)

  def asdict(self) -> dict[str, Any]:
    return {
        'a_boundaries': self.a_boundaries.tolist(),
        'b_boundaries': self.b_boundaries.tolist(),
    }

  def _components(self):
    return (ArrayKey(self.a_boundaries), ArrayKey(self.b_boundaries))

  def __eq__(self, other):
    return (
        isinstance(other, HybridLevels)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  def pressure_boundaries(
      self,
      surface_pressure: cx.Field,
      sim_units: units.SimUnits | None = None,
  ) -> cx.Field:
    a_boundaries, b_boundaries = self.a_boundaries, self.b_boundaries
    a_boundaries = a_boundaries * 100  # Convert to Pascal.
    if sim_units is not None:
      a_boundaries = sim_units.nondimensionalize(a_boundaries * typing.units.Pa)

    def _boundaries(p_surface: jax.Array) -> jax.Array:
      return a_boundaries + b_boundaries * p_surface

    # TODO(dkochkov): Consider adding HybridLevelsBoundaries coordinate and
    # tagging this output with it.
    out_axes = {k: v + 1 for k, v in surface_pressure.named_axes}
    return cx.cmap(_boundaries(surface_pressure), out_axes)

  def pressure_centers(
      self,
      surface_pressure: cx.Field,
      sim_units: units.SimUnits | None = None,
  ) -> cx.Field:
    """Returns pressure at layer centers given `surface_pressure`."""
    a = self.fields['a'] * 100  # In Pascal.
    if sim_units is not None:
      a = cx.field(sim_units.nondimensionalize(a.data * typing.units.Pa), self)
    return a + self.fields['b'] * surface_pressure

  def integrate_over_pressure(
      self,
      x: cx.Field,
      surface_pressure: cx.Field,
      sim_units: units.SimUnits | None,
  ) -> cx.Field:
    a, b = self.hybrid_levels.a_boundaries, self.hybrid_levels.b_boundaries
    if sim_units is not None:
      a_nondim = sim_units.nondimensionalize(a * typing.units.hPa)
    else:
      a_nondim = a * 100  # default hPa -> Pa conversion.
    nondim_hybrid_levels = hybrid_coordinates.HybridCoordinates(a_nondim, b)
    integral_fn = functools.partial(
        hybrid_coordinates.integral_over_pressure,
        coordinates=nondim_hybrid_levels,
        axis=0,
        keepdims=False,
    )
    return cx.cmap(integral_fn)(x.untag(self), surface_pressure)

  def to_xarray(self) -> dict[str, xarray.Variable]:
    variables = super().to_xarray()
    metadata = dict(
        a_boundaries=self.a_boundaries.tolist(),
        b_boundaries=self.b_boundaries.tolist(),
    )
    variables['hybrid'].attrs = metadata
    return variables

  @classmethod
  def from_coefficients(
      cls,
      a_coefficients: list[float] | np.ndarray,
      b_coefficients: list[float] | np.ndarray,
      p0_in_hpa: float = 1000.0,
  ) -> Self:
    hybrid_levels = hybrid_coordinates.HybridCoordinates.from_coefficients(
        a_coeffs=a_coefficients, b_coeffs=b_coefficients, p0=p0_in_hpa
    )
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  @classmethod
  def from_dinosaur_hybrid_levels(
      cls,
      hybrid_levels: hybrid_coordinates.HybridCoordinates,
  ) -> Self:
    return cls(
        a_boundaries=hybrid_levels.a_boundaries,
        b_boundaries=hybrid_levels.b_boundaries,
    )

  @classmethod
  def with_n_levels(
      cls,
      n_levels,
      sigma_exponent: float = 1.5,
      stretch_exponent: float = 0.5,
  ) -> Self:
    hybrid_levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
        n_levels=n_levels,
        sigma_exponent=sigma_exponent,
        stretch_exponent=stretch_exponent,
    )
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  # TODO(janniyuval) To remove after submitting cl/844884846
  @classmethod
  def analytic_ecmwf_like(
      cls,
      n_levels: int = 137,
      p0: float = 1013.25,
      p_top: float = 0.0,
      p_transition: float = 80.0,
      alpha_strat: float = 4.2,
      alpha_surf: float = 0.25,
      b_exponent: float = 1.3,
  ) -> Self:
    """Returns HybridLevels constructed using analytic_ecmwf_like strategy."""
    hybrid_levels = hybrid_coordinates.HybridCoordinates.analytic_ecmwf_like(
        n_levels=n_levels,
        p0=p0,
        p_top=p_top,
        p_transition=p_transition,
        alpha_strat=alpha_strat,
        alpha_surf=alpha_surf,
        b_exponent=b_exponent,
    )
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  @classmethod
  def ecmwf137_interpolated(
      cls,
      n_levels: int,
  ) -> Self:
    """Returns HybridLevels interpolated from ECMWF 137 levels."""
    hybrid_levels = hybrid_coordinates.HybridCoordinates.ecmwf137_interpolated(
        n_levels=n_levels,
    )
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  @classmethod
  def ECMWF137(cls) -> Self:
    """Returns HybridLevels with 137 levels from ECMWF's IFS."""
    hybrid_levels = hybrid_coordinates.HybridCoordinates.ECMWF137()
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  @classmethod
  def UFS127(cls) -> Self:
    """Returns HybridLevels with 127 levels from NOAA's UFS."""
    hybrid_levels = hybrid_coordinates.HybridCoordinates.UFS127()
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'hybrid':
      return cx.coords.NoCoordinateMatch(f'dimension {dim!r} != "hybrid"')

    if 'hybrid' not in coords:
      return cx.coords.NoCoordinateMatch(
          'no associated coordinate for "hybrid"'
      )

    if coords['hybrid'].ndim != 1:
      return cx.coords.NoCoordinateMatch(
          '"hybrid" coordinate is not a 1D array'
      )

    attrs = coords['hybrid'].attrs
    if 'a_boundaries' not in attrs or 'b_boundaries' not in attrs:
      return cx.coords.NoCoordinateMatch(
          'a_boundaries or b_boundaries not in "hybrid" attributes'
      )

    a_boundaries = np.asarray(attrs['a_boundaries'])
    b_boundaries = np.asarray(attrs['b_boundaries'])

    # Check that the hybrid coordinate is just an index.
    n_layers = len(a_boundaries) - 1
    if coords.sizes[dim] != n_layers:
      return cx.coords.NoCoordinateMatch('level dimension size mismatch')
    if not np.array_equal(coords['hybrid'].data, np.arange(1, n_layers + 1)):
      return cx.coords.NoCoordinateMatch(
          'hybrid coordinate is not a simple index'
      )

    return cls(a_boundaries=a_boundaries, b_boundaries=b_boundaries)


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
    return {self.name: cx.field(np.arange(self.n_layers), self)}

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str, Any], set[str]]:
    labeled_layers = cx.LabeledAxis(self.name, np.arange(self.n_layers))
    if self in indexers:
      indexers[labeled_layers] = indexers.pop(self)
    i_indexers, consumed = labeled_layers.map_indexers(indexers, method)
    remap_consumed = {labeled_layers: self}
    consumed = {remap_consumed.get(k, k) for k in consumed}
    return i_indexers, consumed

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    labeled_layers = cx.LabeledAxis(self.name, np.arange(self.n_layers))
    if self in indexers:
      indexers[labeled_layers] = indexers.pop(self)
    return labeled_layers.isel(indexers)

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'layer_index':
      return cx.coords.NoCoordinateMatch(f'dimension {dim!r} != "layer_index"')

    if coords['layer_index'].ndim != 1:
      return cx.coords.NoCoordinateMatch(
          'layer_index coordinate is not a 1D array'
      )

    n_layers = coords.sizes['layer_index']
    got = coords['layer_index'].data
    if not np.array_equal(got, np.arange(n_layers)):
      return cx.coords.NoCoordinateMatch(
          f'unexpected layer_index coordinate is not sequential integers: {got}'
      )
    return cls(n_layers=n_layers)


def validate_nonnegative_depth(centers: Iterable[float] | np.ndarray):
  if not np.min(centers) >= 0:
    raise ValueError('SoilLevels coordinate should have non-negative depth.')


def validate_increasing_depth(centers: Iterable[float] | np.ndarray):
  if not np.min(np.diff(centers)) > 0:
    raise ValueError(
        'SoilLevels coordinate should have monotonically increasing depth,'
        ' starting at the soil surface.'
    )


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SoilLevels(cx.Coordinate):
  """Coordinates that discretize the vertical depth of the soil layer.

  Attributes:
    centers: center depth of each soil layer, starting at the level closest to
      the soil surface. Must be monotonically increasing.
  """

  centers: np.ndarray

  def __init__(self, centers: Iterable[float] | np.ndarray):
    validate_nonnegative_depth(centers)
    validate_increasing_depth(centers)
    centers = np.asarray(centers, dtype=np.float32)
    object.__setattr__(self, 'centers', centers)

  @property
  def dims(self):
    return ('soil_levels',)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.centers.shape

  @property
  def fields(self):
    return {'soil_levels': cx.field(self.centers, self)}

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str, Any], set[str]]:
    return coordax.experimental.map_indexers_using_ticks(
        self,
        indexers,
        ticks_are_sorted=True,
        method=method,
    )

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    key = self.dims[0] if self.dims[0] in indexers else self
    indexer = indexers[key]
    if isinstance(indexer, int):
      return cx.Scalar()

    return SoilLevels(self.centers[indexer])

  def asdict(self) -> dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def _components(self):
    return (ArrayKey(self.centers),)

  def __eq__(self, other):
    return (
        isinstance(other, SoilLevels)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  def __fiddle_serialize__(self):
    return {'centers': self.centers.tolist()}

  @classmethod
  def with_era5_levels(cls, layers=(0, 1, 2, 3)):
    era5_centers = [3.5, 17.5, 64, 194.5]
    centers = [era5_centers[i] for i in layers]
    return cls(centers=centers)

  @classmethod
  def from_xarray(
      cls,
      dims: tuple[str, ...],
      coords: xarray.Coordinates,
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'soil_levels':
      return cx.coords.NoCoordinateMatch(
          f'Leading dimension {dim!r} != "soil_levels"'
      )
    name = dim
    if coords[name].ndim != 1:
      return cx.coords.NoCoordinateMatch(
          'SoilLevels coordinate is not a 1D array'
      )
    got = coords[name].data
    return cls(centers=got)


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
  vertical: (
      SigmaLevels | PressureLevels | LayerLevels | HybridLevels | SoilLevels
  ) = dataclasses.field()
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
