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
"""Defines the Field class, which is the main data structure for Coordax.

``Field`` objects keep track of positional and named dimensions of an array.
Named dimensions of a ``Field`` are associated with coordinates that describe
their discretization.
"""
from __future__ import annotations

import collections
import functools
import itertools
import operator
import textwrap
import typing
from typing import Any, Callable, Self, TypeAlias, TypeGuard, TypeVar

import jax
import jax.numpy as jnp
from neuralgcm.experimental.coordax import coordinate_systems
from neuralgcm.experimental.coordax import named_axes
from neuralgcm.experimental.coordax import utils
import numpy as np
# TODO(shoyer): consider making Xarray an optional dependency of core Coordax
import xarray


Pytree: TypeAlias = Any
Sequence = collections.abc.Sequence

T = TypeVar('T')
Coordinate = coordinate_systems.Coordinate


def _dimension_names(*names: str | Coordinate) -> tuple[str, ...]:
  """Returns a tuple of dimension names from a list of names or coordinates."""
  dims_or_name_tuple = lambda x: x.dims if isinstance(x, Coordinate) else (x,)
  return sum([dims_or_name_tuple(c) for c in names], start=tuple())


@utils.export
def cmap(
    fun: Callable[..., Any], out_axes: dict[str, int] | None = None
) -> Callable[..., Any]:
  """Vectorizes `fun` over coordinate dimensions of ``Field`` inputs.

  Args:
    fun: Function to vectorize over coordinate dimensions.
    out_axes: Optional dictionary from dimension names to axis positions in the
      output. By default, dimension names appear as the trailing dimensions of
      every output, in order of their appearance on the inputs.

  Returns:
    A vectorized version of `fun` that applies original `fun` to locally
    positional dimensions in inputs, while vectorizing over all coordinate
    dimensions. All dimensions over which `fun` is vectorized will be present in
    every output.
  """
  if hasattr(fun, '__name__'):
    fun_name = fun.__name__
  else:
    fun_name = repr(fun)
  if hasattr(fun, '__doc__'):
    fun_doc = fun.__doc__
  else:
    fun_doc = None
  return _cmap_with_doc(fun, fun_name, fun_doc, out_axes)


def _cmap_with_doc(
    fun: Callable[..., Any],
    fun_name: str,
    fun_doc: str | None = None,
    out_axes: dict[str, int] | None = None,
) -> Callable[..., Any]:
  """Builds a coordinate-vectorized wrapped function with a docstring."""

  @functools.wraps(fun)
  def wrapped_fun(*args, **kwargs):
    leaves, treedef = jax.tree.flatten((args, kwargs), is_leaf=is_field)
    field_leaves = [leaf for leaf in leaves if is_field(leaf)]
    all_coords = {}
    for field in field_leaves:
      for dim_name, c in field.coords.items():
        if dim_name in all_coords and all_coords[dim_name] != c:
          other = all_coords[dim_name]
          raise ValueError(f'Coordinates {c=} != {other=} use same {dim_name=}')
        else:
          all_coords[dim_name] = c
    named_array_leaves = [x.named_array if is_field(x) else x for x in leaves]
    fun_on_named_arrays = named_axes.nmap(fun, out_axes=out_axes)
    na_args, na_kwargs = jax.tree.unflatten(treedef, named_array_leaves)
    result = fun_on_named_arrays(*na_args, **na_kwargs)

    def _wrap_field(leaf):
      return Field.from_namedarray(
          named_array=leaf,
          coords={k: all_coords[k] for k in leaf.dims if k in all_coords},
      )

    return jax.tree.map(_wrap_field, result, is_leaf=named_axes.is_namedarray)

  docstr = (
      f'Dimension-vectorized version of `{fun_name}`. Takes similar arguments'
      f' as `{fun_name}` but accepts and returns Fields in place of arrays.'
  )
  if fun_doc:
    docstr += f'\n\nOriginal documentation:\n\n{fun_doc}'
  wrapped_fun.__doc__ = docstr
  return wrapped_fun


def _swapped_binop(binop):
  """Swaps the order of operations for a binary operation."""

  def swapped(x, y):
    return binop(y, x)

  return swapped


def _wrap_scalar_conversion(scalar_conversion):
  """Wraps a scalar conversion operator on a Field."""

  def wrapped_scalar_conversion(self: Field):
    if self.shape:
      raise ValueError(
          f'Cannot convert a non-scalar Field with {scalar_conversion}'
      )
    return scalar_conversion(self.data)

  return wrapped_scalar_conversion


def _wrap_array_method(name):
  """Wraps an array method on a Field."""

  def func(array, *args, **kwargs):
    return getattr(array, name)(*args, **kwargs)

  array_method = getattr(jax.Array, name)
  wrapped_func = cmap(func)
  functools.update_wrapper(
      wrapped_func,
      array_method,
      assigned=('__name__', '__qualname__', '__annotations__'),
      updated=(),
  )
  wrapped_func.__module__ = __name__
  wrapped_func.__doc__ = (
      'Name-vectorized version of array method'
      f' `{name} <numpy.ndarray.{name}>`. Takes similar arguments as'
      f' `{name} <numpy.ndarray.{name}>` but accepts and returns Fields'
      ' in place of regular arrays.'
  )
  return wrapped_func


@utils.export
@jax.tree_util.register_pytree_node_class
class Field:
  """An array with optional named dimensions and associated coordinates."""

  _named_array: named_axes.NamedArray
  _coords: dict[str, Coordinate]

  def __init__(
      self,
      data: jax.typing.ArrayLike,
      dims: tuple[str | None, ...] | None = None,
      coords: dict[str, Coordinate] | None = None,
  ):
    """Construct a Field.

    Args:
      data: the underlying data array.
      dims: optional tuple of dimension names, with the same length as
        `data.ndim`. Strings indicate named axes, and may not be repeated.
        `None` indicates positional axes. If `dims` is not provided, all axes
        are positional.
      coords: optional mapping from dimension names to associated
        `coordax.Coordinate` objects.
    """
    named_array = named_axes.NamedArray(data, dims)

    if coords is None:
      coords = {}

    self._named_array = named_array
    self._coords = coords
    self._check_valid()

  def _check_valid(self) -> None:
    """Checks that the field coordinates and dimension names are consistent."""

    # internal consistency of coordinates
    for dim, coord in self.coords.items():
      if coord.ndim > 1:
        raise ValueError(
            f'all coordinates in the coords dict must be 1D, got {coord} for '
            f'dimension {dim}. Consider using Field.tag() instead to associate '
            'multi-dimensional coordinates.'
        )
      if (dim,) != coord.dims:
        raise ValueError(
            f'coordinate under key {dim!r} in the coords dict must have '
            f'dims={(dim,)!r} but got {coord.dims=}'
        )

    data_dims = set(self.named_array.named_dims)
    keys_dims = set(self.coords.keys())
    if not keys_dims <= data_dims:
      raise ValueError(
          'coordinate keys must be a subset of the named dimensions of the '
          f'underlying named array, got coordinate keys {keys_dims} vs '
          f'data dimensions {data_dims}'
      )

    for dim, coord in self.coords.items():
      if self.named_shape[dim] != coord.sizes[dim]:
        raise ValueError(
            f'inconsistent size for dimension {dim!r} between data and '
            f'coordinates: {self.named_shape[dim]} vs {coord.sizes[dim]} on '
            f'named array vs coordinate:\n{self.named_array}\n{coord}'
        )

  @classmethod
  def from_namedarray(
      cls,
      named_array: named_axes.NamedArray,
      coords: dict[str, Coordinate] | None = None,
  ) -> Self:
    """Creates a Field from a named array."""
    return cls(named_array.data, named_array.dims, coords)

  @classmethod
  def from_xarray(
      cls,
      data_array: xarray.DataArray,
      coord_types: Sequence[type[Coordinate]] = (
          coordinate_systems.LabeledAxis,
          coordinate_systems.DummyAxis,
      ),
  ) -> Self:
    """Converts an xarray.DataArray into a coordax.Field.

    Args:
      data_array: xarray.DataArray to convert into a Field.
      coord_types: sequence of coordax.Coordinate subclasses with `from_xarray`
        methods defined. The first coordinate class that returns a coordinate
        object (indicating a match) will be used. By default, coordinates will
        use only generic coordax.LabeledAxis objects.

    Returns:
      A coordax.Field object with the same data as the input xarray.DataArray.
    """
    field = cls(data_array.data)
    dims = data_array.dims
    coords = []

    if not all(isinstance(dim, str) for dim in dims):
      raise TypeError(
          'can only convert DataArray objects with string dimensions to Field'
      )
    dims = typing.cast(tuple[str, ...], dims)

    if not coord_types:
      raise ValueError('coord_types must be non-empty')

    def get_next_match():
      reasons = []
      for coord_type in coord_types:
        result = coord_type.from_xarray(dims, data_array.coords)
        if isinstance(result, Coordinate):
          return result
        assert isinstance(result, coordinate_systems.NoCoordinateMatch)
        coord_name = coord_type.__module__ + '.' + coord_type.__name__
        reasons.append(f'{coord_name}: {result.reason}')

      reasons_str = '\n'.join(reasons)
      raise ValueError(
          'failed to convert xarray.DataArray to coordax.Field, because no '
          f'coordinate type matched the dimensions starting with {dims}:\n'
          f'{data_array}\n\n'
          f'Reasons why coordinate matching failed:\n{reasons_str}'
      )

    while dims:
      coord = get_next_match()
      coords.append(
          coord.name
          if isinstance(coord, coordinate_systems.DummyAxis)
          else coord
      )
      assert coord.ndim > 0  # dimensions will shrink by at least one
      dims = dims[coord.ndim :]

    return field.tag(*coords)

  def to_xarray(self) -> xarray.DataArray:
    """Convert this Field to an xarray.DataArray.

    Returns:
      An xarray.DataArray object with the same data as the input coordax.Field.
      This DataArray will still be wrapping a jax.Array, and have operations
      implemented on jax.Array objects using the Python Array API interface.
    """
    if not all(isinstance(dim, str) for dim in self.dims):
      raise ValueError(
          'can only convert Field objects with fully named dimensions to '
          f'xarray.DataArray objects, got dimensions {self.dims!r}'
      )

    coords = {}
    for coord in self.coords.values():
      for name, variable in coord.to_xarray().items():
        if name in coords and not variable.identical(coords[name]):
          raise ValueError(
              f'inconsistent coordinate fields for {name!r}:\n'
              f'{variable}\nvs\n{coords[name]}'
          )
        coords[name] = variable

    return xarray.DataArray(data=self.data, dims=self.dims, coords=coords)

  @property
  def named_array(self) -> named_axes.NamedArray:
    """The value of the underlying named array."""
    return self._named_array

  @property
  def coords(self) -> dict[str, Coordinate]:
    """The coordinate objects associated with this field."""
    return self._coords

  @property
  def data(self) -> jax.Array:
    """The value of the underlying data array."""
    return self.named_array.data

  @property
  def dtype(self) -> np.dtype:
    """The dtype of the field."""
    return self.named_array.dtype

  @property
  def named_shape(self) -> dict[str, int]:
    """A mapping of axis names to their sizes."""
    return self.named_array.named_shape

  @property
  def positional_shape(self) -> tuple[int, ...]:
    """A tuple of axis sizes for any anonymous axes."""
    return self.named_array.positional_shape

  @property
  def shape(self) -> tuple[int, ...]:
    """A tuple of axis sizes of the underlying data array."""
    return self.named_array.shape

  @property
  def ndim(self) -> int:
    return len(self.dims)

  @property
  def dims(self) -> tuple[str | None, ...]:
    """Named and unnamed dimensions of this array."""
    return self.named_array.dims

  @property
  def named_dims(self) -> tuple[str, ...]:
    """Namd dimensions of this array."""
    return self.named_array.named_dims

  @property
  def named_axes(self) -> dict[str, int]:
    """Mapping from dimension names to axis positions."""
    return self.named_array.named_axes

  @property
  def coord_fields(self) -> dict[str, Field]:
    """A mapping from coordinate field names to their values."""
    return functools.reduce(
        operator.or_, [c.fields for c in self.coords.values()], {}
    )

  def tree_flatten(self):
    """Flatten this object for JAX pytree operations."""
    return [self.named_array], tuple(self.coords.items())

  @classmethod
  def tree_unflatten(cls, coords, leaves) -> Self:
    """Unflatten this object for JAX pytree operations."""
    [named_array] = leaves
    result = object.__new__(cls)
    result._named_array = named_array
    result._coords = dict(coords)
    if isinstance(named_array.data, jnp.ndarray):
      result._check_valid()
    return result

  def unwrap(self, *names: str | Coordinate) -> jax.Array:
    """Extracts underlying data from a field without named dimensions."""
    names = _dimension_names(*names)
    if names != self.named_dims:
      raise ValueError(
          f'Field has {self.named_dims=} but {names=} were requested.'
      )
    return self.data

  def _validate_matching_coords(
      self, dims_or_coords: Sequence[str | Coordinate]
  ):
    """Validate that given coordinates are all found on this field."""
    axes = []
    for part in dims_or_coords:
      if isinstance(part, Coordinate):
        axes.extend(part.axes)

    for c in axes:
      [dim] = c.dims
      if dim not in self.coords:
        raise ValueError(
            f'coordinate not found on this field:\n{c}\n'
            f'not found in coordinates {list(self.coords)}'
        )
      if self.coords[dim] != c:
        raise ValueError(
            'coordinate not equal to the corresponding coordinate on this'
            f' field:\n{c}\nvs\n{self.coords[dim]}'
        )

  def untag(self, *axis_order: str | Coordinate) -> Field:
    """Returns a view of the field with the requested axes made positional."""
    self._validate_matching_coords(axis_order)
    untag_dims = _dimension_names(*axis_order)
    named_array = self.named_array.untag(*untag_dims)
    coords = {k: v for k, v in self.coords.items() if k not in untag_dims}
    result = Field.from_namedarray(named_array=named_array, coords=coords)
    return result

  def tag(self, *names: str | Coordinate | ellipsis | None) -> Field:
    """Returns a Field with attached coordinates to the positional axes."""
    tag_dims = _dimension_names(*names)
    tagged_array = self.named_array.tag(*tag_dims)
    coords = {}
    coords.update(self.coords)
    for c in names:
      if isinstance(c, Coordinate):
        for dim, axis in zip(c.dims, c.axes):
          coords[dim] = axis
    result = Field.from_namedarray(tagged_array, coords)
    return result

  # Note: Can't call this "transpose" like Xarray, to avoid conflicting with the
  # positional only ndarray method.
  def order_as(self, *axis_order: str | Coordinate) -> Field:
    """Returns a field with the axes in the given order."""
    self._validate_matching_coords(axis_order)
    ordered_dims = _dimension_names(*axis_order)
    ordered_array = self.named_array.order_as(*ordered_dims)
    result = Field.from_namedarray(ordered_array, self.coords)
    return result

  def __repr__(self):
    data_repr = textwrap.indent(repr(self.data), prefix=' ' * 13)[13:]
    if self.coords:
      coords_repr = (
          '{\n'
          + '\n'.join(
              textwrap.indent(f'{dim!r}: {self.coords[dim]},', prefix=' ' * 12)
              for dim in self.coords
          )
          + '\n        }'
      )
    else:
      coords_repr = '{}'
    return textwrap.dedent(f"""\
    {type(self).__name__}(
        data={data_repr},
        dims={self.dims},
        coords={coords_repr},
    )""")

  # Convenience wrappers: Elementwise infix operators.
  __lt__ = _cmap_with_doc(operator.lt, 'jax.Array.__lt__')
  __le__ = _cmap_with_doc(operator.le, 'jax.Array.__le__')
  __eq__ = _cmap_with_doc(operator.eq, 'jax.Array.__eq__')
  __ne__ = _cmap_with_doc(operator.ne, 'jax.Array.__ne__')
  __ge__ = _cmap_with_doc(operator.ge, 'jax.Array.__ge__')
  __gt__ = _cmap_with_doc(operator.gt, 'jax.Array.__gt__')

  __add__ = _cmap_with_doc(operator.add, 'jax.Array.__add__')
  __sub__ = _cmap_with_doc(operator.sub, 'jax.Array.__sub__')
  __mul__ = _cmap_with_doc(operator.mul, 'jax.Array.__mul__')
  __truediv__ = _cmap_with_doc(operator.truediv, 'jax.Array.__truediv__')
  __floordiv__ = _cmap_with_doc(operator.floordiv, 'jax.Array.__floordiv__')
  __mod__ = _cmap_with_doc(operator.mod, 'jax.Array.__mod__')
  __divmod__ = _cmap_with_doc(divmod, 'jax.Array.__divmod__')
  __pow__ = _cmap_with_doc(operator.pow, 'jax.Array.__pow__')
  __lshift__ = _cmap_with_doc(operator.lshift, 'jax.Array.__lshift__')
  __rshift__ = _cmap_with_doc(operator.rshift, 'jax.Array.__rshift__')
  __and__ = _cmap_with_doc(operator.and_, 'jax.Array.__and__')
  __or__ = _cmap_with_doc(operator.or_, 'jax.Array.__or__')
  __xor__ = _cmap_with_doc(operator.xor, 'jax.Array.__xor__')

  __radd__ = _cmap_with_doc(_swapped_binop(operator.add), 'jax.Array.__radd__')
  __rsub__ = _cmap_with_doc(_swapped_binop(operator.sub), 'jax.Array.__rsub__')
  __rmul__ = _cmap_with_doc(_swapped_binop(operator.mul), 'jax.Array.__rmul__')
  __rtruediv__ = _cmap_with_doc(
      _swapped_binop(operator.truediv), 'jax.Array.__rtruediv__'
  )
  __rfloordiv__ = _cmap_with_doc(
      _swapped_binop(operator.floordiv), 'jax.Array.__rfloordiv__'
  )
  __rmod__ = _cmap_with_doc(_swapped_binop(operator.mod), 'jax.Array.__rmod__')
  __rdivmod__ = _cmap_with_doc(_swapped_binop(divmod), 'jax.Array.__rdivmod__')
  __rpow__ = _cmap_with_doc(_swapped_binop(operator.pow), 'jax.Array.__rpow__')
  __rlshift__ = _cmap_with_doc(
      _swapped_binop(operator.lshift), 'jax.Array.__rlshift__'
  )
  __rrshift__ = _cmap_with_doc(
      _swapped_binop(operator.rshift), 'jax.Array.__rrshift__'
  )
  __rand__ = _cmap_with_doc(_swapped_binop(operator.and_), 'jax.Array.__rand__')
  __ror__ = _cmap_with_doc(_swapped_binop(operator.or_), 'jax.Array.__ror__')
  __rxor__ = _cmap_with_doc(_swapped_binop(operator.xor), 'jax.Array.__rxor__')

  __abs__ = _cmap_with_doc(operator.abs, 'jax.Array.__abs__')
  __neg__ = _cmap_with_doc(operator.neg, 'jax.Array.__neg__')
  __pos__ = _cmap_with_doc(operator.pos, 'jax.Array.__pos__')
  __invert__ = _cmap_with_doc(operator.inv, 'jax.Array.__invert__')

  # Convenience wrappers: Scalar conversions.
  __bool__ = _wrap_scalar_conversion(bool)
  __complex__ = _wrap_scalar_conversion(complex)
  __int__ = _wrap_scalar_conversion(int)
  __float__ = _wrap_scalar_conversion(float)
  __index__ = _wrap_scalar_conversion(operator.index)

  # elementwise operations
  astype = _wrap_array_method('astype')
  clip = _wrap_array_method('clip')
  conj = _wrap_array_method('conj')
  conjugate = _wrap_array_method('conjugate')
  imag = _wrap_array_method('imag')
  real = _wrap_array_method('real')
  round = _wrap_array_method('round')
  view = _wrap_array_method('view')

  # Intentionally not included: anything that acts on a subset of axes or takes
  # an axis as an argument (e.g., mean). It is ambiguous whether these should
  # act over positional or named axes.
  # TODO(shoyer): re-write some of these with explicit APIs similar to xarray.

  # maybe include some of below with names that signify positional nature?
  # reshape = _wrap_array_method('reshape')
  # squeeze = _wrap_array_method('squeeze')
  # transpose = _wrap_array_method('transpose')
  # T = _wrap_array_method('T')
  # mT = _wrap_array_method('mT')  # pylint: disable=invalid-name


@utils.export
def wrap(array: jax.typing.ArrayLike, *names: str | Coordinate | None) -> Field:
  """Wraps a positional array as a ``Field``."""
  field = Field(array)
  if names:
    field = field.tag(*names)
  return field


@utils.export
def wrap_like(array: jax.typing.ArrayLike, other: Field) -> Field:
  """Wraps `array` with the same coordinates as `other`."""
  array = jnp.asarray(array)
  if array.shape != other.shape:
    raise ValueError(f'{array.shape=} and {other.shape=} must be equal')
  return Field(array, other.dims, other.coords)


def zeros(
    *coords: Coordinate, dtype: jax.typing.DTypeLike | None = None
) -> Field:
  """Returns a zero-filled Field with the given coordinates."""
  coord = coordinate_systems.compose_coordinates(*coords)
  return Field(jnp.zeros(coord.shape, dtype)).tag(coord)


@utils.export
def is_field(value) -> TypeGuard[Field]:
  """Returns True if `value` is of type `Field`."""
  return isinstance(value, Field)


PyTree = Any


@utils.export
def tag(tree: PyTree, *dims: str | Coordinate | ellipsis | None) -> PyTree:
  """Tag dimensions on all NamedArrays in a PyTree."""
  tag_arrays = lambda x: x.tag(*dims) if is_field(x) else x
  return jax.tree.map(tag_arrays, tree, is_leaf=is_field)


@utils.export
def untag(tree: PyTree, *dims: str | Coordinate) -> PyTree:
  """Untag dimensions from all NamedArrays in a PyTree."""
  untag_arrays = lambda x: x.untag(*dims) if is_field(x) else x
  return jax.tree.map(untag_arrays, tree, is_leaf=is_field)
