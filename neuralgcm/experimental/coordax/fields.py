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
import operator
import textwrap
from typing import Any, Callable, Literal, Self, TypeAlias, TypeGuard, TypeVar

import jax
import jax.numpy as jnp
from neuralgcm.experimental.coordax import coordinate_systems
from neuralgcm.experimental.coordax import named_axes
from neuralgcm.experimental.coordax import ndarrays
from neuralgcm.experimental.coordax import utils
import numpy as np
import treescope
from treescope import lowering
from treescope import rendering_parts
# TODO(shoyer): consider making Xarray an optional dependency of core Coordax
import xarray


Pytree: TypeAlias = Any
Sequence = collections.abc.Sequence

T = TypeVar('T')

Coordinate = coordinate_systems.Coordinate
LabeledAxis = coordinate_systems.LabeledAxis
DummyAxis = coordinate_systems.DummyAxis

Array = ndarrays.Array
ArrayLike = ndarrays.ArrayLike


def _dimension_names(*names: str | Coordinate) -> tuple[str, ...]:
  """Returns a tuple of dimension names from a list of names or coordinates."""
  dims_or_name_tuple = lambda x: x.dims if isinstance(x, Coordinate) else (x,)
  return sum([dims_or_name_tuple(c) for c in names], start=tuple())


def _axes_attrs(field: Field) -> str:
  """Returns a string representation of the coordinate attributes."""

  def _coord_name(c: coordinate_systems.Coordinate):
    if isinstance(c, coordinate_systems.SelectedAxis):
      return f'SelectedAxis({c.coordinate.__class__.__name__}, axis={c.axis})'
    return c.__class__.__name__

  coord_names = {k: _coord_name(c) for k, c in field.axes.items()}
  return '{' + ', '.join(f'{k!r}: {v}' for k, v in coord_names.items()) + '}'


@utils.export
def cmap(
    fun: Callable[..., Any],
    out_axes: dict[str, int] | None = None,
    *,
    vmap: Callable = jax.vmap,  # pylint: disable=g-bare-generic
) -> Callable[..., Any]:
  """Vectorizes `fun` over coordinate dimensions of ``Field`` inputs.

  Args:
    fun: Function to vectorize over coordinate dimensions.
    out_axes: Optional dictionary from dimension names to axis positions in the
      output. By default, dimension names appear as the trailing dimensions of
      every output, in order of their appearance on the inputs.
    vmap: Vectorizing transformation to use when mapping over named axes.
      Defaults to jax.vmap. A different implementation can be used to make
      coordax compatible with custom objects (e.g. neural net modules).

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
  return _cmap_with_doc(fun, fun_name, fun_doc, out_axes, vmap=vmap)


def _cmap_with_doc(
    fun: Callable[..., Any],
    fun_name: str,
    fun_doc: str | None = None,
    out_axes: dict[str, int] | None = None,
    *,
    vmap: Callable = jax.vmap,  # pylint: disable=g-bare-generic
) -> Callable[..., Any]:
  """Builds a coordinate-vectorized wrapped function with a docstring."""

  @functools.wraps(fun)
  def wrapped_fun(*args, **kwargs):
    leaves, treedef = jax.tree.flatten((args, kwargs), is_leaf=is_field)
    field_leaves = [leaf for leaf in leaves if is_field(leaf)]
    all_axes = {}
    for field in field_leaves:
      for dim_name, c in field.axes.items():
        if dim_name in all_axes and all_axes[dim_name] != c:
          other = all_axes[dim_name]
          raise ValueError(f'Coordinates {c=} != {other=} use same {dim_name=}')
        else:
          all_axes[dim_name] = c
    named_array_leaves = [x.named_array if is_field(x) else x for x in leaves]
    fun_on_named_arrays = named_axes.nmap(fun, out_axes=out_axes, vmap=vmap)
    na_args, na_kwargs = jax.tree.unflatten(treedef, named_array_leaves)
    result = fun_on_named_arrays(*na_args, **na_kwargs)

    def _wrap_field(leaf):
      return Field.from_namedarray(
          named_array=leaf,
          axes={k: all_axes[k] for k in leaf.dims if k in all_axes},
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


def _check_valid(
    named_array: named_axes.NamedArray, axes: dict[str, Coordinate]
) -> None:
  """Checks that the field coordinates and dimension names are consistent."""

  # internal consistency of coordinates
  for dim, coord in axes.items():
    if coord.ndim > 1:
      raise ValueError(
          f'all coordinates in the axes dict must be 1D, got {coord} for '
          f'dimension {dim}. Consider using Field.tag() instead to associate '
          'multi-dimensional coordinates.'
      )
    if (dim,) != coord.dims:
      raise ValueError(
          f'coordinate under key {dim!r} in the axes dict must have '
          f'dims={(dim,)!r} but got {coord.dims=}'
      )

  data_dims = set(named_array.named_dims)
  keys_dims = set(_remove_dummy_axes(axes).keys())
  if not keys_dims <= data_dims:
    raise ValueError(
        'axis keys must be a subset of the named dimensions of the '
        f'underlying named array, got axis keys {keys_dims} vs '
        f'data dimensions {data_dims}'
    )

  for dim, coord in axes.items():
    if named_array.named_shape[dim] != coord.sizes[dim]:
      raise ValueError(
          f'inconsistent size for dimension {dim!r} between data and'
          f' coordinates: {named_array.named_shape[dim]} vs'
          f' {coord.sizes[dim]} on named array vs'
          f' coordinate:\n{named_array}\n{coord}'
      )


def _remove_dummy_axes(axes: dict[str, Coordinate]) -> dict[str, Coordinate]:
  """Removes dummy axes from a dict of coordinates."""
  return {k: v for k, v in axes.items() if not isinstance(v, DummyAxis)}


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


def _in_treescope_abbreviation_mode() -> bool:
  """Returns True if treescope.abbreviation is set by context or globally."""
  return treescope.abbreviation_threshold.get() is not None


@utils.export
@jax.tree_util.register_pytree_node_class
class Field:
  """An array with optional named dimensions and associated coordinates."""

  _named_array: named_axes.NamedArray
  _axes: dict[str, Coordinate]

  def __init__(
      self,
      data: ArrayLike,
      dims: tuple[str | None, ...] | None = None,
      axes: dict[str, Coordinate] | None = None,
  ):
    """Construct a Field.

    Args:
      data: the underlying data array.
      dims: optional tuple of dimension names, with the same length as
        `data.ndim`. Strings indicate named axes, and may not be repeated.
        `None` indicates positional axes. If `dims` is not provided, all axes
        are positional.
      axes: optional mapping from dimension names to associated
        `coordax.Coordinate` objects.
    """
    self._named_array = named_axes.NamedArray(data, dims)
    if axes is None:
      self._axes = {}
    else:
      _check_valid(self._named_array, axes)
      self._axes = _remove_dummy_axes(axes)

  @classmethod
  def from_namedarray(
      cls,
      named_array: named_axes.NamedArray,
      axes: dict[str, Coordinate] | None = None,
  ) -> Self:
    """Creates a Field from a named array."""
    return cls(named_array.data, named_array.dims, axes)

  @classmethod
  def from_xarray(
      cls,
      data_array: xarray.DataArray,
      coord_types: Sequence[type[Coordinate]] = (LabeledAxis, DummyAxis),
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
    coord = coordinate_systems.from_xarray(data_array, coord_types)
    return field.tag(coord)

  def to_xarray(self) -> xarray.DataArray:
    """Convert this Field to an xarray.DataArray with NumPy array data.

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

    # TODO(shoyer): Consider making this conversion optional, for use cases
    # where it is desirable to wrap jax.Array objects inside Xarray.
    data = ndarrays.to_numpy_array(self.data)

    coords = {}
    for coord in self.axes.values():
      for name, variable in coord.to_xarray().items():
        if name in coords and not variable.identical(coords[name]):
          raise ValueError(
              f'inconsistent coordinate fields for {name!r}:\n'
              f'{variable}\nvs\n{coords[name]}'
          )
        coords[name] = variable

    return xarray.DataArray(data=data, dims=self.dims, coords=coords)

  @property
  def named_array(self) -> named_axes.NamedArray:
    """The value of the underlying named array."""
    return self._named_array

  @property
  def axes(self) -> dict[str, Coordinate]:
    """The coordinate axes associated with this field."""
    return self._axes

  @property
  def data(self) -> Array:
    """The value of the underlying data array."""
    return self.named_array.data

  @property
  def dtype(self) -> np.dtype | None:
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
        operator.or_, [c.fields for c in self.axes.values()], {}
    )

  def tree_flatten(self):
    """Flatten this object for JAX pytree operations."""
    return [self.named_array], tuple(self.axes.items())

  @classmethod
  def tree_unflatten(cls, axes, leaves) -> Self:
    """Unflatten this object for JAX pytree operations."""
    [named_array] = leaves
    result = object.__new__(cls)
    result._named_array = named_array
    result._axes = dict(axes)
    if isinstance(named_array.data, Array):
      _check_valid(result.named_array, result.axes)
    return result

  def unwrap(self, *names: str | Coordinate) -> Array:
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
      if dim not in self.axes:
        raise ValueError(
            f'coordinate not found on this field:\n{c}\n'
            f'not found in coordinates {list(self.axes)}'
        )
      if self.axes[dim] != c:
        raise ValueError(
            'coordinate not equal to the corresponding coordinate on this'
            f' field:\n{c}\nvs\n{self.axes[dim]}'
        )

  def untag(self, *axis_order: str | Coordinate) -> Field:
    """Returns a view of the field with the requested axes made positional."""
    self._validate_matching_coords(axis_order)
    untag_dims = _dimension_names(*axis_order)
    named_array = self.named_array.untag(*untag_dims)
    axes = {k: v for k, v in self.axes.items() if k not in untag_dims}
    result = Field.from_namedarray(named_array=named_array, axes=axes)
    return result

  def tag(self, *names: str | Coordinate | ellipsis | None) -> Field:
    """Returns a Field with attached coordinates to the positional axes."""
    tag_dims = _dimension_names(*names)
    tagged_array = self.named_array.tag(*tag_dims)
    axes = {}
    axes.update(self.axes)
    for c in names:
      if isinstance(c, Coordinate):
        for dim, axis in zip(c.dims, c.axes):
          # TODO(shoyer): consider raising an error if an unnamed axis has the
          # wrong size.
          if dim is not None:
            axes[dim] = axis
    result = Field.from_namedarray(tagged_array, axes)
    return result

  # Note: Can't call this "transpose" like Xarray, to avoid conflicting with the
  # positional only ndarray method.
  def order_as(self, *axis_order: str | Coordinate) -> Field:
    """Returns a field with the axes in the given order."""
    self._validate_matching_coords(axis_order)
    ordered_dims = _dimension_names(*axis_order)
    ordered_array = self.named_array.order_as(*ordered_dims)
    result = Field.from_namedarray(ordered_array, self.axes)
    return result

  def broadcast_like(self, other: Self) -> Self:
    """Returns a field broadcasted like `other`."""
    for k, v in self.axes.items():
      if other.axes.get(k) != v:
        raise ValueError(
            'cannot broadcast field because axes corresponding to dimension '
            f'{k!r} do not match: {v} vs {other.axes.get(k)}'
        )
    return Field.from_namedarray(
        self.named_array.broadcast_like(other.named_array), other.axes
    )

  def __repr__(self):
    if _in_treescope_abbreviation_mode():
      return treescope.render_to_text(self)
    else:
      with treescope.abbreviation_threshold.set_scoped(1):
        with treescope.using_expansion_strategy(9, 80):
          return treescope.render_to_text(self)

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    """Treescope handler for Field."""

    def _make_label():
      # reuse dim/shape summary from the underlying NamedArray.
      attrs, summary, _ = named_axes.attrs_summary_type(
          self.named_array, False
      )
      axes_attrs = _axes_attrs(self)
      attrs = ' '.join([attrs, f'axes={axes_attrs}'])

      return rendering_parts.summarizable_condition(
          summary=rendering_parts.abbreviation_color(  # non-expanded repr.
              rendering_parts.text(
                  f'<{type(self).__name__} {attrs} {summary}>'
              )
          ),
          detail=rendering_parts.siblings(
              rendering_parts.text(f'<{type(self).__name__} ('),
          ),
      )

    children = rendering_parts.build_field_children(
        self,
        path,
        subtree_renderer,
        fields_or_attribute_names=('dims', 'shape', 'axes'),
    )
    indented_children = rendering_parts.indented_children(children)
    return rendering_parts.build_custom_foldable_tree_node(
        contents=rendering_parts.summarizable_condition(
            detail=rendering_parts.siblings(indented_children, ')')
        ),
        label=lowering.maybe_defer_rendering(
            main_thunk=lambda _: _make_label(),
            placeholder_thunk=_make_label,
        ),
        path=path,
        expand_state=rendering_parts.ExpandState.WEAKLY_COLLAPSED,
    )

  def __treescope_ndarray_adapter__(self):
    """Treescope handler for named arrays."""

    def _summary_fn(field, inspect_data):
      attrs, array_summary, data_type = named_axes.attrs_summary_type(
          field.named_array, inspect_data
      )
      axes_attrs = _axes_attrs(field)
      attrs = ', '.join([attrs, f'axes={axes_attrs}'])
      return attrs, array_summary, data_type

    return named_axes.NamedArrayAdapter(_summary_fn)

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
def wrap(array: ArrayLike, *names: str | Coordinate | None) -> Field:
  """Wraps a positional array as a ``Field``."""
  field = Field(array)
  if names:
    field = field.tag(*names)
  return field


@utils.export
def wrap_like(array: ArrayLike, other: Field) -> Field:
  """Wraps `array` with the same coordinates as `other`."""
  if isinstance(array, jax.typing.ArrayLike):
    array = jnp.asarray(array)
  if array.shape != other.shape:
    raise ValueError(f'{array.shape=} and {other.shape=} must be equal')
  return Field(array, other.dims, other.axes)


@utils.export
def is_field(value) -> TypeGuard[Field]:
  """Returns True if `value` is of type `Field`."""
  return isinstance(value, Field)


MissingAxes = Literal['error', 'dummy', 'skip']


def get_coordinate(
    field: Field, *, missing_axes: MissingAxes = 'dummy'
) -> Coordinate:
  """Returns a single coordinate for a field.

  Args:
    field: coordax.Field from which the coordinate will be extracted.
    missing_axes: controls how axes without coorinates are handled. Options are:

      * ``'dummy'``: uses DummyAxis for dimensions without a coordinate.
      * ``'skip'``: ignores dimensions without a coordinate.
      * ``'error'``: raises if dimensions without a coordinate are present.

  Returns:
    Coordinate associated with the `field`.
  """
  if missing_axes not in ('dummy', 'skip', 'error'):
    raise ValueError(
        'missing axes must be one of "dummy", "skip", or "error", got'
        f' {missing_axes!r}'
    )
  axes = []
  for d, s in zip(field.dims, field.shape, strict=True):
    if d in field.axes:
      axes.append(field.axes[d])
    elif missing_axes == 'dummy':
      axes.append(coordinate_systems.DummyAxis(d, s))
    elif missing_axes == 'error':
      raise ValueError(f'{field.dims=} has unnamed dims and {missing_axes=}')
  return coordinate_systems.compose(*axes)


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
