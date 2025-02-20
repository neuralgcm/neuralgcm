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
"""Coordinate systems for use on coordax.Field objects.

``Coordinate`` objects define a discretization schema, dimension names and
provide methods & coordinate field values to facilitate computations.
"""
from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import operator
from typing import Any, Self, TypeAlias, TYPE_CHECKING

import jax
from neuralgcm.experimental.coordax import utils
import numpy as np
# TODO(shoyer): consider making Xarray an optional dependency of core Coordax
import xarray

if TYPE_CHECKING:
  # import only under TYPE_CHECKING to avoid circular dependency
  # pylint: disable=g-bad-import-order
  from neuralgcm.experimental.coordax import fields


Pytree: TypeAlias = Any
Sequence = collections.abc.Sequence


@utils.export
@dataclasses.dataclass(frozen=True)
class NoCoordinateMatch:
  """For use when a coordinate does not match an xarray.Coordinates object."""

  reason: str


@utils.export
class Coordinate(abc.ABC):
  """Abstract class for coordinate objects.

  Coordinate subclasses are expected to obey several invariants:
  1. Dimension names may not be repeated: `len(set(dims)) == len(dims)`
  2. All dimensions must be named: `len(shape) == len(dims)`

  Every non-abstract Coordinate subclass must be registered as a "static"
  pytree node, e.g., by decorating the class with
  `@jax.tree_util.register_static`. Static pytrees nodes must implement
  `__hash__` and `__eq__` according to the requirements of keys in Python
  dictionaries. This is easiest to acheive with frozen dataclasses, but care
  must be taken when working with np.ndarray attributes.

  TODO(shoyer): add documentation examples, including a version using ArrayKey
  to wrap np.ndarray attributions.
  """

  @property
  @abc.abstractmethod
  def dims(self) -> tuple[str, ...]:
    """Dimension names of the coordinate."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def shape(self) -> tuple[int, ...]:
    """Shape of the coordinate."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def fields(self) -> dict[str, fields.Field]:
    """A maps from field names to their values."""

  @property
  def sizes(self) -> dict[str, int]:
    """Sizes of all dimensions on this coordinate."""
    return dict(zip(self.dims, self.shape))

  @property
  def ndim(self) -> int:
    """Dimensionality of the coordinate."""
    return len(self.dims)

  @property
  def axes(self) -> tuple[Coordinate, ...]:
    """Tuple of one-dimensional Coordinate objects for each dimension."""
    if self.ndim == 1:
      return (self,)
    else:
      return tuple(SelectedAxis(self, i) for i in range(self.ndim))

  def to_xarray(self) -> dict[str, xarray.Variable]:
    """Convert this coordinate into xarray variables."""
    variables = {}
    dims_set = set(self.dims)
    for name, coord_field in self.fields.items():
      if set(coord_field.dims) <= dims_set:
        # xarray.DataArray coordinate dimensions must be a subset of the
        # dimensions of the associated DataArray, which is not necessarily a
        # constraint for coordax.Field.
        variables[name] = xarray.Variable(coord_field.dims, coord_field.data)
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | NoCoordinateMatch:
    """Construct a matching Coordax coordinate from xarray, if possible.

    Args:
      dims: tuple of dimension names. Only the leading dimensions should be
        checks for a match.
      coords: xarray.Coordinates object providing dimension sizes and coordinate
        values.

    Returns:
      A matching instance of this coordinate or `NoCoordinateMatch` if this
      coordinate does not match the xarray dimensions and coordinates.
    """
    raise NotImplementedError('from_xarray not implemented')


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


@utils.export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class Scalar(Coordinate):
  """Zero dimensional sentinel coordinate used to label stand alone scalars."""

  @property
  def dims(self) -> tuple[str, ...]:
    return ()

  @property
  def shape(self) -> tuple[int, ...]:
    return ()

  @property
  def fields(self) -> dict[str, fields.Field]:
    return {}


@utils.export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SelectedAxis(Coordinate):
  """Coordinate that exposes one dimension of a multidimensional coordinate."""

  coordinate: Coordinate
  axis: int

  def __post_init__(self):
    if self.axis >= self.coordinate.ndim:
      raise ValueError(
          f'Dimension {self.axis=} of {self.coordinate=} is out of bounds'
      )

  @property
  def dims(self) -> tuple[str, ...]:
    """Dimension names of the coordinate."""
    return (self.coordinate.dims[self.axis],)

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the coordinate."""
    return (self.coordinate.shape[self.axis],)

  @property
  def fields(self) -> dict[str, fields.Field]:
    """A maps from field names to their values."""
    return self.coordinate.fields

  def __repr__(self):
    return f'coordax.SelectedAxis({self.coordinate!r}, axis={self.axis})'

  def to_xarray(self) -> dict[str, xarray.Variable]:
    """Convert this coordinate into xarray variables."""
    # Override the default method to avoid restricting variables to only those
    # along the selected axis.
    return self.coordinate.to_xarray()


def consolidate_coordinates(*coordinates: Coordinate) -> tuple[Coordinate, ...]:
  """Consolidates coordinates without SelectedAxis objects, if possible."""
  axes = []
  result = []

  def reset_axes():
    result.extend(axes)
    axes[:] = []

  def append_axis(c):
    axes.append(c)
    if len(axes) == c.coordinate.ndim:
      # sucessful consolidation
      result.append(c.coordinate)
      axes[:] = []

  for c in coordinates:
    if isinstance(c, Scalar):
      continue
    if isinstance(c, SelectedAxis) and c.axis == 0:
      # new SelectedAxis to consider consolidating
      reset_axes()
      append_axis(c)
    elif (
        isinstance(c, SelectedAxis)
        and axes
        and c.axis == len(axes)
        and c.coordinate == axes[-1].coordinate
    ):
      # continued SelectedAxis to consolidate
      append_axis(c)
    else:
      # coordinate cannot be consolidated
      reset_axes()
      result.append(c)

  reset_axes()

  return tuple(result)


@utils.export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class CartesianProduct(Coordinate):
  """Coordinate defined as the outer product of independent coordinates."""

  coordinates: tuple[Coordinate, ...]

  def __post_init__(self):
    new_coordinates = []
    for c in self.coordinates:
      new_coordinates.extend(c.axes)
    combined_coordinates = consolidate_coordinates(*new_coordinates)
    if len(combined_coordinates) <= 1:
      raise ValueError('CartesianProduct must contain more than 1 component')
    existing_dims = collections.Counter()
    for c in new_coordinates:
      existing_dims.update(c.dims)
    repeated_dims = [dim for dim, count in existing_dims.items() if count > 1]
    if repeated_dims:
      raise ValueError(f'CartesianProduct components contain {repeated_dims=}')
    object.__setattr__(self, 'coordinates', tuple(new_coordinates))

  def __eq__(self, other):
    # TODO(shoyer): require exact equality of coordinate types?
    if not isinstance(other, CartesianProduct):
      return len(self.coordinates) == 1 and self.coordinates[0] == other
    return isinstance(other, CartesianProduct) and all(
        self.coordinates[i] == other.coordinates[i]
        for i in range(len(self.coordinates))
    )

  @property
  def dims(self):
    return sum([c.dims for c in self.coordinates], start=tuple())

  @property
  def shape(self) -> tuple[int, ...]:
    """Returns the shape of the coordinate axes."""
    return sum([c.shape for c in self.coordinates], start=tuple())

  @property
  def fields(self) -> dict[str, fields.Field]:
    """Returns a mapping from field names to their values."""
    return functools.reduce(
        operator.or_, [c.fields for c in self.coordinates], {}
    )

  @property
  def axes(self) -> tuple[Coordinate, ...]:
    """Returns a tuple of Axis objects for each dimension."""
    return self.coordinates


@utils.export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SizedAxis(Coordinate):
  """One dimensional coordinate with fixed size but no associated fields."""

  name: str
  size: int

  @property
  def dims(self) -> tuple[str, ...]:
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return (self.size,)

  @property
  def fields(self) -> dict[str, fields.Field]:
    return {}

  def __repr__(self):
    return f'coordax.SizedAxis({self.name!r}, size={self.size})'

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | NoCoordinateMatch:
    dim = dims[0]
    if dim in coords:
      return NoCoordinateMatch(
          'can only reconstruct SizedAxis objects from xarray dimensions'
          ' without associated coordinate variables, but found a coordinate'
          f' variable for dimension {dim!r}'
      )
    for name, coord in coords.variables.items():
      if dim in coord.dims:
        return NoCoordinateMatch(
            'can only reconstruct SizedAxis objects from xarray dimensions'
            ' if the dimensions is not found on any coordinate variables, but '
            f' found a coordinate variable for dimension {dim!r} on {name!r}'
        )
    return cls(dim, size=coords.sizes[dim])


@utils.export
@dataclasses.dataclass(frozen=True)
class DummyAxis(Coordinate):
  """Dummy coordinate for dimensions without associated coordinate values.

  DummyAxis objects cannot be instantiated, but this class can be used in the
  `coord_types` argument to `Field.from_xarray` to match dimensions that do not
  have associated coordinate values.
  """

  name: str

  def __post_init__(self):
    raise TypeError('DummyAxis cannot be instantiated')

  @property
  def dims(self) -> tuple[str, ...]:
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return (0,)

  @property
  def fields(self) -> dict[str, fields.Field]:
    return {}

  def __repr__(self):
    return f'coordax.DummyAxis({self.name!r})'

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | NoCoordinateMatch:
    dim = dims[0]
    for name, coord in coords.variables.items():
      if dim in coord.dims:
        return NoCoordinateMatch(
            f'cannot omit a Coordinate object for dimension {dim!r}'
            f' because it is used by at least one coordinate variable: {name!r}'
        )
    result = object.__new__(cls)
    object.__setattr__(result, 'name', dim)
    return result


# TODO(dkochkov): consider storing tuple values instead of np.ndarray (which
# could be exposed as a property).
@utils.export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LabeledAxis(Coordinate):
  """One dimensional coordinate with custom coordinate values."""

  name: str
  ticks: np.ndarray

  def __post_init__(self):
    object.__setattr__(self, 'ticks', np.asarray(self.ticks))
    if self.ticks.ndim != 1:
      raise ValueError(f'ticks must be a 1D array, got {self.ticks.shape=}')

  @property
  def dims(self) -> tuple[str, ...]:
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.ticks.shape

  @property
  def fields(self) -> dict[str, fields.Field]:
    # needs local import to avoid circular dependency
    from neuralgcm.experimental.coordax import fields  # pylint: disable=g-import-not-at-top
    return {self.name: fields.wrap(self.ticks, self)}

  def _components(self):
    return (self.name, ArrayKey(self.ticks))

  def __eq__(self, other):
    return (
        isinstance(other, LabeledAxis)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  def __repr__(self):
    return f'coordax.LabeledAxis({self.name!r}, ticks={self.ticks!r})'

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | NoCoordinateMatch:
    dim = dims[0]
    if dim not in coords:
      return NoCoordinateMatch(
          f'no associated coordinate for dimension {dim!r}'
      )
    if coords[dim].ndim != 1:
      return NoCoordinateMatch(f'coordinate for dimension {dim!r} is not 1D')
    return cls(dim, coords[dim].data)


@utils.export
def compose_coordinates(*coordinates: Coordinate) -> Coordinate:
  """Composes `coords` into a single coordinate system by cartesian product."""
  if not coordinates:
    raise ValueError('No coordinates provided.')
  coordinate_axes = []
  for c in coordinates:
    if isinstance(c, CartesianProduct):
      coordinate_axes.extend(c.coordinates)
    else:
      coordinate_axes.append(c)
  coordinates = consolidate_coordinates(*coordinate_axes)
  if len(coordinates) == 1:
    return coordinates[0]
  return CartesianProduct(coordinates)
