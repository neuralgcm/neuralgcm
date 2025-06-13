# Copyright 2022 Google LLC
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

"""Defines boundary conditions for standard computational domains."""

import abc
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import typing


@nnx_compat.dataclass
class BoundaryCondition(nnx.Module, abc.ABC):
  """Base class for boundary conditions that use pad-trim approach.

  This base class expresses boundary conditions on `cx.Field` and
  `Array` data types by padding and trimming ghost cells. The `pad` method
  introduces ghost values to the inputs to ensure that the computation over the
  original domain accounts for the type of the boundary condition implemented by
  the subclass. The `trim` method is used to remove the invalid ghost values and
  return data aligned with the original domain prior to `pad`. At the moment
  we require subclasses to implement methods for Arrays and Fields, though
  implementations are often reused.
  """

  @property
  @abc.abstractmethod
  def ndim(self) -> int:
    ...

  @abc.abstractmethod
  def pad_array(
      self,
      x: typing.Array,
      pad_width: tuple[tuple[int, int], ...],
  ) -> typing.Array:
    ...

  @abc.abstractmethod
  def pad(self, x: cx.Field, pad_sizes: dict[str, tuple[int, int]]) -> cx.Field:
    ...

  @abc.abstractmethod
  def trim_array(
      self,
      x: typing.Array,
      pad_width: tuple[tuple[int, int], ...],
  ) -> typing.Array:
    ...

  @abc.abstractmethod
  def trim(self, x: cx.Field) -> cx.Field:
    ...


@nnx_compat.dataclass
class LonLatBoundary(BoundaryCondition):
  """Implements boundary condition for data on lon-lat grids.

  Longitude axis is padded with wrapped values using jnp.wrap.
  Latitude axis is padded with half-shifted and reflected values.
  """

  @property
  def ndim(self) -> int:
    return 2

  def pad_array(
      self,
      x: typing.Array,
      pad_width: tuple[tuple[int, int], ...],
  ) -> typing.Array:
    if x.ndim != 2 or len(pad_width) != 2:
      raise ValueError(
          f'pad_array was called with {x.ndim=} and {len(pad_width)=} which '
          'are expected to be equal to 2 representing lon,lat respectively.'
      )
    if all(pad == 0 for pad in jax.tree.leaves(pad_width)):
      return x
    lon_size, lat_size = x.shape
    lon_pad, lat_pad = pad_width
    shift = lon_size // 2  # 180 degree rotation.
    before = jnp.roll(x[:, : lat_pad[0]], shift, axis=0)
    after = jnp.roll(x[:, (lat_size - lat_pad[1]) :], shift, axis=0)
    # The latitude is reversed so that the ghost cells start with the nearest
    # neighbor to the original domain.
    x = jnp.concatenate([before[:, ::-1], x, after[:, ::-1]], axis=1)
    return jnp.pad(x, (lon_pad, (0, 0)), mode='wrap')  # lon padding.

  def trim_array(
      self,
      x: typing.Array,
      pad_width: tuple[tuple[int, int], ...],
  ) -> typing.Array:
    if x.ndim != 2 or len(pad_width) != 2:
      raise ValueError(
          f'trim_array was called with {x.ndim=} and {len(pad_width)=} which '
          'are expected to be equal to 2 representing lon,lat respectively.'
      )
    if all(pad == 0 for pad in jax.tree.leaves(pad_width)):
      return x
    lon_pad, lat_pad = pad_width
    lon_slice = slice(lon_pad[0], x.shape[0] - lon_pad[1])
    lat_slice = slice(lat_pad[0], x.shape[1] - lat_pad[1])
    return x[lon_slice, lat_slice]

  def pad(self, x: cx.Field, pad_sizes: dict[str, tuple[int, int]]) -> cx.Field:
    lon_lat = ('longitude', 'latitude')
    lon_lat_sizes = {d: x.named_shape[d] for d in lon_lat}
    pads = tuple(pad_sizes.get(d) for d in lon_lat)
    grid = cx.compose_coordinates(
        *[x.axes.get(d, cx.DummyAxis(d, lon_lat_sizes[d])) for d in lon_lat]
    )
    padded_grid = coordinates.CoordinateWithPadding(grid, pad_sizes)
    x = x.untag(*grid.dims)
    return cx.cmap(self.pad_array, x.named_axes)(x, pads).tag(padded_grid)

  def trim(self, x: cx.Field) -> cx.Field:
    padded_lon_lat = tuple('padded_' + d for d in ['longitude', 'latitude'])
    grid = cx.compose_coordinates(*[x.axes.get(d) for d in padded_lon_lat])
    if (
        not isinstance(grid, coordinates.CoordinateWithPadding)
        or grid.ndim != 2
    ):
      raise ValueError(
          f'LonLatBoundary.trim received {x} for which inferred {grid=} is not'
          ' a 2D coordinates.CoordinateWithPadding, suggesting that `x` was'
          ' not padded using LonLatBoundary.pad method.'
      )
    pad_sizes = grid.pad_sizes
    pads = tuple(pad_sizes[d.removeprefix('padded_')] for d in padded_lon_lat)
    trimmed_grid = grid.coordinate
    x = x.untag(grid)
    return cx.cmap(self.trim_array, x.named_axes)(x, pads).tag(trimmed_grid)
