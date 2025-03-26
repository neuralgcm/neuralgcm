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
"""Array with optional named axes, inspired by Penzai's NamedArray.

This module is intended to be nearly a drop-in replacement for Penzai's
NamedArray, but with an alternative, simpler implementation. Dimensions are
specified by a tuple, where each element is either a string or `None`. `None` is
used to indicate strictly positional dimensions.

Some code and documentation is adapted from penzai.core.named_axes.
"""
from __future__ import annotations

import dataclasses
import functools
import operator
import textwrap
import types
from typing import Any, Callable, Self, TypeGuard, TypeVar

import jax
import jax.numpy as jnp
from neuralgcm.experimental.coordax import ndarrays
import numpy as np
from treescope import dtype_util
from treescope import lowering
from treescope import ndarray_adapters
from treescope import rendering_parts
from treescope.external import jax_support
from treescope.external import numpy_support


Array = ndarrays.Array
ArrayLike = ndarrays.ArrayLike


def attrs_summary_type(
    named_array: NamedArray,
    inspect_data: bool,
) -> tuple[str, str, str]:
  """Returns a summary of a `named_array` and its data type."""
  if isinstance(named_array.data, jax.Array) and not isinstance(
      named_array.data, jax.core.Tracer
  ):
    contained_type = 'jax.Array'
  else:
    contained_type = type(named_array.data).__name__

  type_info = dtype_util.get_dtype_name(named_array.dtype)
  attrs_parts = []
  attrs_parts.append(f'dims={named_array.dims}')
  attrs_parts.append(f'shape={named_array.shape}')
  summary_parts = []
  summary_parts.append(f'dtype={type_info}')
  if (
      inspect_data
      and isinstance(named_array.data, jax.Array)
      and jax_support.safe_to_summarize(named_array.data)
  ):
    summary_parts.append(jax_support.summarize_array_data(named_array.data))
  attrs = ' '.join(attrs_parts)
  summary = ','.join(summary_parts)
  return attrs, summary, contained_type


class NamedArrayAdapter(ndarray_adapters.NDArrayAdapter['NamedArray']):
  """Array adapter for named arrays."""

  def __init__(self, summary_fn: Callable[..., tuple[str, str, str]]):
    self.summary_fn = summary_fn

  def get_axis_info_for_array_data(
      self, array: NamedArray
  ) -> tuple[ndarray_adapters.AxisInfo, ...]:
    infos = {}
    for axis, dim in enumerate(array.dims):
      if dim is None:
        infos[axis] = ndarray_adapters.PositionalAxisInfo(
            axis, array.shape[axis]
        )
      else:
        infos[axis] = ndarray_adapters.NamedPositionalAxisInfo(
            axis_logical_index=axis,
            axis_name=dim,  # pytype: disable=wrong-arg-types
            size=array.shape[axis],
        )
    return tuple(infos[i] for i in range(len(infos)))

  def get_array_data_with_truncation(
      self,
      array: NamedArray,
      mask: NamedArray | jax.Array | np.ndarray | None,
      edge_items_per_axis: tuple[int | None, ...],
  ) -> tuple[np.ndarray, np.ndarray]:
    if mask is None:
      mask_data = None  # sets no mask in the JAXArrayAdapter.
    else:
      # Make sure mask is compatible.
      if isinstance(mask, NamedArray):
        if mask.dims.count(None):
          raise ValueError(f'Mask must be fully named, got {mask.dims=}')
        bad_names = set(mask.dims) - set(array.dims)
        if bad_names:
          raise ValueError(
              'Valid mask must be broadcastable to the shape of `array`, but it'
              f' had extra axis names {bad_names}'
          )
        mask = mask.order_as(*(d for d in array.dims if d in mask.dims))
        mask_data = mask.data
      else:
        if np.broadcast_shapes(mask.shape, array.shape) != array.shape:
          raise ValueError(
              f'{mask.shape=} is not broadcastable to {array.shape=}'
          )
        mask_data = jnp.broadcast_to(mask, array.shape)

    if isinstance(array.data, jax.Array):
      return jax_support.JAXArrayAdapter().get_array_data_with_truncation(
          array=array.data,
          mask=mask_data,
          edge_items_per_axis=edge_items_per_axis,
      )
    else:
      assert isinstance(array.data, np.ndarray | ndarrays.NDArray)
      return numpy_support.NumpyArrayAdapter().get_array_data_with_truncation(
          array=ndarrays.to_numpy_array(array.data),
          mask=mask_data,
          edge_items_per_axis=edge_items_per_axis,
      )

  def get_array_summary(self, array: NamedArray, fast: bool) -> str:
    attrs, summary, contained_type = self.summary_fn(
        array, inspect_data=(not fast)
    )
    full_summary = (
        f'{type(array).__name__}({attrs}, {summary}) (wrapping'
        f' {contained_type})'
    )
    return full_summary

  def get_numpy_dtype(self, array: NamedArray) -> np.dtype | None:
    if isinstance(array.dtype, np.dtype):
      return array.dtype
    else:
      return None

  def get_sharding_info_for_array_data(
      self, array: NamedArray
  ) -> ndarray_adapters.ShardingInfo | None:
    if not isinstance(array.data, jax.Array):
      return None
    return jax_support.JAXArrayAdapter().get_sharding_info_for_array_data(
        array.data
    )

  def should_autovisualize(self, array: NamedArray) -> bool:
    # only visualize jax.Arrays that are not Tracers and not deleted to avoid
    # raising an errors during visualization.
    return isinstance(array.data, np.ndarray) or (
        isinstance(array.data, jax.Array)
        and not isinstance(array.data, jax.core.Tracer)
        and not array.data.is_deleted()
    )


def _collect_named_shape(
    leaves_and_paths: list[tuple[jax.tree_util.KeyPath, Any]],
    source_description: str,
) -> dict[str, int]:
  """Collect shared named_shape, or raise an informative error."""
  known_sizes = {}
  bad_dims = []
  for _, leaf in leaves_and_paths:
    if isinstance(leaf, NamedArray):
      for name, size in leaf.named_shape.items():
        if name in known_sizes:
          if known_sizes[name] != size and name not in bad_dims:
            bad_dims.append(name)
        else:
          known_sizes[name] = size

  if bad_dims:
    shapes_str = []
    for keypath, leaf in leaves_and_paths:
      if isinstance(leaf, NamedArray):
        if keypath[0] == jax.tree_util.SequenceKey(0):
          prefix = 'args'
        else:
          assert keypath[0] == jax.tree_util.SequenceKey(1)
          prefix = 'kwargs'
        path = jax.tree_util.keystr(keypath[1:])
        shapes_str.append(f'  {prefix}{path}.named_shape == {leaf.named_shape}')
    shapes_message = '\n'.join(shapes_str)

    raise ValueError(
        f'Inconsistent sizes in a call to {source_description} for dimensions '
        f'{bad_dims}:\n{shapes_message}'
    )

  return known_sizes


def _normalize_out_axes(
    out_axes: dict[str, int] | None, named_shape: dict[str, int]
) -> dict[str, int]:
  """Normalize the out_axes argument to nmap."""
  if out_axes is None:
    return {dim: -(i + 1) for i, dim in enumerate(reversed(named_shape.keys()))}

  if out_axes.keys() != named_shape.keys():
    raise ValueError(
        f'out_axes keys {list(out_axes)} must must match the named '
        f'dimensions {list(named_shape)}'
    )
  any_negative = any(axis < 0 for axis in out_axes.values())
  any_non_negative = any(axis >= 0 for axis in out_axes.values())
  if any_negative and any_non_negative:
    # TODO(shoyer) consider supporting mixed positive and negative out_axes.
    # This would require using jax.eval_shape() to determine the
    # dimensionality of all output arrays.
    raise ValueError(
        'out_axes must be either all positive or all negative, but got '
        f'{out_axes}'
    )
  if len(set(out_axes.values())) != len(out_axes):
    raise ValueError(
        f'out_axes must all have unique values, but got {out_axes}'
    )
  return out_axes


def _nest_vmap_axis(inner_axis: int, outer_axis: int) -> int:
  """Update a vmap in/out axis to account for wrapping in an outer vmap."""
  if outer_axis >= 0 and inner_axis >= 0:
    if inner_axis > outer_axis:
      return inner_axis - 1
    else:
      assert inner_axis < outer_axis
      return inner_axis
  else:
    assert outer_axis < 0 and inner_axis < 0
    if inner_axis > outer_axis:
      return inner_axis
    else:
      assert inner_axis < outer_axis
      return inner_axis + 1


def nmap(
    fun: Callable,  # pylint: disable=g-bare-generic
    out_axes: dict[str, int] | None = None,
    *,
    vmap: Callable = jax.vmap,  # pylint: disable=g-bare-generic
) -> Callable:  # pylint: disable=g-bare-generic
  """Automatically vectorizes ``fun`` over named dimensions.

  ``nmap`` is a "named dimension vectorizing map". It wraps an ordinary
  positional-axis-based function so that it accepts NamedArrays as input and
  produces NamedArrays as output, and vectorizes over all of named dimensions,
  calling the original function with positionally-indexed slices corresponding
  to each argument's `positional_shape`.

  Unlike `jax.vmap`, the axes to vectorize over are inferred
  automatically from the named dimensions in the NamedArray inputs, rather
  than being specified as part of the mapping transformation. Specifically, each
  dimension name that appears in any of the arguments is vectorized over jointly
  across all arguments that include that dimension, and is then included as a
  named dimension in the output. To make an axis visible to ``fun``, you can
  call `untag` on the argument and pass the axis name(s) of interest; ``fun``
  will then see those axes as positional axes instead of mapping over them.

  `untag` and ``nmap`` are together the primary ways to apply individual
  operations to axes of a NamedArray. `tag` can then be used on the result to
  re-bind names to positional axes.

  Within ``fun``, any mapped-over axes will be accessible using standard JAX
  collective operations like ``psum``, although doing this is usually
  unnecessary.

  Args:
    fun: Function to vectorize by name. This can take arbitrary arguments (even
      non-JAX-arraylike arguments or "static" axis sizes), but must produce a
      PyTree of JAX ArrayLike outputs.
    out_axes: Optional dictionary from dimension names to axis positions in the
      output. By default, dimension names appear as the trailing dimensions of
      every output, in order of their appearance on the inputs.
    vmap: Vectorizing transformation to use when mapping over named axes.
      Defaults to jax.vmap. A different implementation can be used to make
      coordax compatible with custom objects (e.g. neural net modules).

  Returns:
    An automatically-vectorized version of ``fun``, which can optionally be
    called with NamedArrays instead of ordinary arrays, and which will always
    return NamedArrays for each of its output leaves. Any argument (or PyTree
    leaf of an argument) that is a NamedArray will have its named dimensions
    vectorized over; ``fun`` will then be called with batch tracers
    corresponding to slices of the input array that are shaped like
    ``named_array_arg.positional_shape``.
  """
  if hasattr(fun, '__module__') and hasattr(fun, '__name__'):
    fun_name = f'{fun.__module__}.{fun.__name__}'
  else:
    fun_name = repr(fun)
  if hasattr(fun, '__doc__'):
    fun_doc = fun.__doc__
  else:
    fun_doc = None
  return _nmap_with_doc(fun, fun_name, fun_doc, out_axes, vmap=vmap)


def _nmap_with_doc(
    fun: Callable,  # pylint: disable=g-bare-generic
    fun_name: str,
    fun_doc: str | None = None,
    out_axes: dict[str, int] | None = None,
    *,
    vmap: Callable = jax.vmap,  # pylint: disable=g-bare-generic
) -> Callable:  # pylint: disable=g-bare-generic
  """Implementation of nmap."""

  @functools.wraps(fun)
  def wrapped_fun(*args, **kwargs):
    leaves_and_paths, treedef = jax.tree_util.tree_flatten_with_path(
        (args, kwargs),
        is_leaf=lambda node: isinstance(node, NamedArray),
    )
    leaves = [leaf for _, leaf in leaves_and_paths]

    named_shape = _collect_named_shape(
        leaves_and_paths, source_description=f'nmap({fun})'
    )
    all_dims = tuple(named_shape.keys())
    out_axes_dict = _normalize_out_axes(out_axes, named_shape)

    nested_in_axes = {}
    nested_out_axes = {}

    # working_input_dims and working_out_axes will be iteratively updated to
    # account for nesting in outer vmap calls
    working_input_dims: list[list[str | None]] = [
        list(leaf.dims) if isinstance(leaf, NamedArray) else []
        for leaf in leaves
    ]
    working_out_axes = out_axes_dict.copy()

    # Calculate in_axes and out_axes for all calls to vmap.
    for vmap_dim in all_dims:
      in_axes = []
      for dims in working_input_dims:
        if vmap_dim in dims:
          axis = dims.index(vmap_dim)
          del dims[axis]
        else:
          axis = None
        in_axes.append(axis)
      nested_in_axes[vmap_dim] = in_axes

      out_axis = working_out_axes.pop(vmap_dim)
      for dim in working_out_axes:
        working_out_axes[dim] = _nest_vmap_axis(working_out_axes[dim], out_axis)
      nested_out_axes[vmap_dim] = out_axis

    assert not working_out_axes  # all dimensions processed

    def vectorized_fun(leaf_data):
      args, kwargs = jax.tree.unflatten(treedef, leaf_data)
      return fun(*args, **kwargs)

    # Recursively apply vmap, in the reverse of the order in which we calculated
    # nested_in_axes and nested_out_axes.
    for vmap_dim in reversed(all_dims):
      vectorized_fun = vmap(
          vectorized_fun,
          in_axes=(nested_in_axes[vmap_dim],),
          out_axes=nested_out_axes[vmap_dim],
          axis_name=vmap_dim,
      )

    leaf_data = [
        leaf.data if isinstance(leaf, NamedArray) else leaf for leaf in leaves
    ]
    result = vectorized_fun(leaf_data)

    def wrap_output(data: Array) -> NamedArray:
      dims = [None] * data.ndim
      for dim, axis in out_axes_dict.items():
        dims[axis] = dim
      return NamedArray(data, tuple(dims))

    is_array = lambda x: isinstance(x, Array)
    return jax.tree.map(wrap_output, result, is_leaf=is_array)

  docstr = (
      f'Dimension-vectorized version of `{fun_name}`. Takes similar arguments'
      f' as `{fun_name}` but accepts and returns NamedArray objects in place '
      'of arrays.'
  )
  if fun_doc:
    docstr += f'\n\nOriginal documentation:\n\n{fun_doc}'
  wrapped_fun.__doc__ = docstr
  return wrapped_fun


def _nmap_unary_op(
    fun: Callable[..., Any], fun_name: str
) -> Callable[[NamedArray], NamedArray]:
  return _nmap_with_doc(fun, fun_name)


def _nmap_binary_op(
    fun: Callable[..., Any], fun_name: str
) -> Callable[[NamedArray, NamedArray | jax.typing.ArrayLike], NamedArray]:
  return _nmap_with_doc(fun, fun_name)


def _nmap_binary_op_two_outputs(
    fun: Callable[..., Any], fun_name: str
) -> Callable[
    [NamedArray, NamedArray | jax.typing.ArrayLike],
    tuple[NamedArray, NamedArray],
]:
  return _nmap_with_doc(fun, fun_name)


def _swapped_binop(binop):
  """Swaps the order of operations for a binary operation."""

  def swapped(x, y):
    return binop(y, x)

  return swapped


T = TypeVar('T')


def _wrap_scalar_conversion(
    scalar_conversion: Callable[..., T],
) -> Callable[[NamedArray], T]:
  """Wraps a scalar conversion operator on a Field."""

  def wrapped_scalar_conversion(self: NamedArray):
    if self.shape:
      raise ValueError(
          f'Cannot convert a non-scalar NamedArray with {scalar_conversion}'
      )
    return scalar_conversion(self.data)

  return wrapped_scalar_conversion


def _wrap_array_method(
    name: str, from_property: bool = False
) -> Callable[..., NamedArray]:
  """Wraps an array method on a Field."""

  if from_property:

    def func(array):
      return getattr(array, name)

  else:

    def func(array, /, *args, **kwargs):
      return getattr(array, name)(*args, **kwargs)

  array_method = getattr(jax.Array, name)
  wrapped_func = nmap(func)
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
      f' `{name} <numpy.ndarray.{name}>` but accepts and returns NamedArray'
      ' objects in place of regular arrays.'
  )
  return wrapped_func


_VALID_PYTREE_OPS = (
    'JAX pytree operations on NamedArray objects are only valid when they'
    ' insert new leading dimensions, or trim unnamed leading dimensions. The'
    ' sizes and positions (from the end) of all named dimensions must be'
    ' preserved.'
)


@dataclasses.dataclass
class _ShapedLeaf:
  """Helper for NamedArray tree_unflatten/tree_flatten."""

  value: Any
  shape: tuple[int, ...]


def _named_shape(
    dims: tuple[str | None, ...], shape: tuple[int, ...]
) -> dict[str, int]:
  return {dim: size for dim, size in zip(dims, shape) if dim is not None}


@jax.tree_util.register_pytree_node_class
class NamedArray:
  """Array with optionally named axes.

  Axis names are either a string or None, indicating an unnamed axis.

  Attributes:
    data: the underlying data array.
    dims: tuple of dimension names, with the same length as `data.ndim`. Strings
      indicate named axes, and may not be repeated. `None` indicates positional
      axes.
  """

  _data: Array
  _dims: tuple[str | None, ...]

  def __init__(
      self,
      data: ArrayLike,
      dims: tuple[str | None, ...] | None = None,
  ):
    """Construct a NamedArray.

    Arguments:
      data: the underlying data array.
      dims: optional tuple of dimension names, with the same length as
        `data.ndim`. Strings indicate named axes, and may not be repeated.
        `None` indicates positional axes. If `dims` is not provided, all axes
        are positional.
    """
    data = ndarrays.to_array(data)
    if dims is None:
      dims = (None,) * data.ndim
    else:
      if data.ndim != len(dims):
        raise ValueError(f'{data.ndim=} != {len(dims)=}')
      named_dims = [dim for dim in dims if dim is not None]
      if len(set(named_dims)) < len(named_dims):
        raise ValueError('dimension names may not be repeated: {dims}')
    self._data = data
    self._dims = dims

  @property
  def data(self) -> Array:
    """Data associated with this array."""
    return self._data

  @property
  def dims(self) -> tuple[str | None, ...]:
    """Named and unnamed dimensions of this array."""
    return self._dims

  @property
  def named_dims(self) -> tuple[str, ...]:
    """Named dimensions of this array."""
    return tuple(dim for dim in self._dims if dim is not None)

  @property
  def named_axes(self) -> dict[str, int]:
    """Mapping from dimension names to axis positions."""
    return {dim: axis for axis, dim in enumerate(self._dims) if dim is not None}

  @property
  def ndim(self) -> int:
    """Number of dimensions in the array, including postional and named axes."""
    return self.data.ndim

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the array, including positional and named axes."""
    return self.data.shape

  @property
  def positional_shape(self) -> tuple[int, ...]:
    """Shape of the array with all named axes removed."""
    return tuple(
        size for dim, size in zip(self.dims, self.data.shape) if dim is None
    )

  @property
  def named_shape(self) -> dict[str, int]:
    """Mapping from dimension names to sizes."""
    return _named_shape(self.dims, self.data.shape)

  @property
  def dtype(self) -> jnp.dtype | None:
    """The dtype of the underlying data array."""
    data = self.data
    return data.dtype if isinstance(data, np.ndarray | jax.Array) else None

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    """Treescope handler."""

    def _make_label():
      attrs, summary, contained_type = attrs_summary_type(self, False)
      return rendering_parts.text(
          f'<{type(self).__name__}({attrs} {summary}) (wrapping'
          f' {contained_type})>'
      )

    return rendering_parts.abbreviation_color(
        lowering.maybe_defer_rendering(
            main_thunk=lambda _: _make_label(),
            placeholder_thunk=_make_label,
        )
    )

  def __treescope_ndarray_adapter__(self):
    """Treescope handler for named arrays."""
    return NamedArrayAdapter(attrs_summary_type)

  def __repr__(self) -> str:
    indent = lambda x: textwrap.indent(x, prefix=' ' * 13)[13:]
    return textwrap.dedent(f"""\
    {type(self).__name__}(
        data={indent(repr(self.data))},
        dims={self.dims},
    )""")

  def tree_flatten(self):
    """Flatten this object for JAX pytree operations."""
    # Arrays unflattened from non-ndarray leaves are wrapped with _ShapedLeaf,
    # which gives them a shape.
    data = self.data.value if isinstance(self.data, _ShapedLeaf) else self.data
    return [data], (self.dims, self.shape)

  @classmethod
  def tree_unflatten(cls, treedef, leaves: list[Array | object]) -> Self:
    """Unflatten this object for JAX pytree operations."""
    dims, shape = treedef
    [data] = leaves

    if not all(
        # isinstance(x, jax.typing.ArrayLike)
        isinstance(x, Array)
        for x in jax.tree.leaves(data, is_leaf=lambda y: y is None)
    ):
      # JAX builds pytrees with non-ndarray leaves inside some transformations,
      # such as vmap, for handling the in_axes argument. We wrap these leaves
      # with _ShapedLeaf to ensure that they produce the same treedef when
      # unflattened.
      # pylint: disable=protected-access
      obj = super().__new__(cls)
      obj._data = _ShapedLeaf(data, shape)  # type: ignore
      obj._dims = dims
      return obj

    # Restored NamedArray objects may have additional or removed leading
    # dimensions, if produced with scan or vmap.
    result = cls._new_with_padded_or_trimmed_dims(data, dims)
    expected_named_shape = _named_shape(dims, shape)
    if result.named_shape != expected_named_shape:
      raise ValueError(
          'named shape mismatch when unflattening to a NamedArray: '
          f'{result.named_shape} != {expected_named_shape}. {_VALID_PYTREE_OPS}'
      )
    return result

  @classmethod
  def _new_with_padded_or_trimmed_dims(
      cls, data: Array, dims: tuple[str | None, ...]
  ) -> Self:
    """Create a new NamedArray, padding or trimming dims to match data.ndim."""
    assert isinstance(data, Array)
    if len(dims) <= data.ndim:
      dims = (None,) * (data.ndim - len(dims)) + dims
    else:
      trimmed_dims = dims[: -data.ndim]
      if any(dim is not None for dim in trimmed_dims):
        raise ValueError(
            'cannot trim named dimensions when unflattening to a NamedArray:'
            f' {trimmed_dims}. {_VALID_PYTREE_OPS} If you are using vmap or'
            ' scan, the first dimension must be unnamed.'
        )
      dims = dims[-data.ndim :]
    return cls(data, dims)

  def tag(self, *dims: str | ellipsis | None) -> Self:
    """Attaches dimension names to the positional axes of an array.

    Args:
      *dims: axis names to assign to each positional axis in the array. Must
        have exactly the same length as the number of unnamed axes in the array,
        unless ellipsis is used (at most once), which indicates all remaining
        unnamed axes at that position.

    Raises:
      ValueError: If the wrong number of dimensions are provided.

    Returns:
      A NamedArray with the given names assigned to the positional axes, and no
      remaining positional axes.
    """
    if not all(
        isinstance(name, (str, types.EllipsisType, types.NoneType))
        for name in dims
    ):
      raise TypeError(f'dimension names must be strings, ... or None: {dims}')

    pos_ndim = len(self.positional_shape)

    ellipsis_count = sum(dim is ... for dim in dims)
    if ellipsis_count > 1:
      raise ValueError(
          f'dimension names contain multiple ellipses (...): {dims}'
      )
    elif ellipsis_count == 1:
      if len(dims) - 1 > pos_ndim:
        raise ValueError(
            'too many dimensions supplied to `tag` for the '
            f'{pos_ndim} positional {"axis" if pos_ndim == 1 else "axes"}: '
            f'{dims}'
        )
      inserted_dims = (None,) * (pos_ndim - len(dims) + 1)
      i = dims.index(...)
      dims = dims[:i] + inserted_dims + dims[i + 1 :]
    else:
      if len(dims) != pos_ndim:
        raise ValueError(
            'there must be exactly as many dimensions given to `tag` as there'
            f' are positional axes in the array, but got {dims} for '
            f'{pos_ndim} positional {"axis" if pos_ndim == 1 else "axes"}.'
        )

    dim_queue = list(reversed(dims))
    new_dims = tuple(
        dim_queue.pop() if dim is None else dim for dim in self.dims
    )
    assert not dim_queue
    return type(self)(self.data, new_dims)

  def untag(self, *dims: str) -> Self:
    """Removes the requested dimension names.

    `untag` can only be called on a `NamedArray` that does not have any
    positional axes. It produces a new `NamedArray` where the axes with the
    requested dimension names are now treated as positional instead.

    Args:
      *dims: axis names to make positional, in the order they should appear in
        the positional array.

    Raises:
      ValueError: if the provided axis ordering is not valid.

    Returns:
      A named array with the given dimensions converted to positional axes.
    """
    if self.positional_shape:
      raise ValueError(
          '`untag` cannot be used to introduce positional axes for a NamedArray'
          ' that already has positional axes. Please assign names to the'
          ' existing positional axes first using `tag`.'
      )

    named_shape = self.named_shape
    if any(dim not in named_shape for dim in dims):
      raise ValueError(
          f'cannot untag {dims} because they are not a subset of the current '
          f'named dimensions {tuple(self.dims)}'
      )

    ordered = tuple(sorted(dims, key=self.dims.index))
    if ordered != dims:
      raise ValueError(
          f'cannot untag {dims} because they do not appear in the order of '
          f'the current named dimensions {ordered}'
      )

    untagged = set(dims)
    new_dims = tuple(None if dim in untagged else dim for dim in self.dims)
    return type(self)(self.data, new_dims)

  def order_as(self, *dims: str | types.EllipsisType) -> Self:
    """Reorder the dimensions of an array.

    All dimensions must be named. Use `tag` first to name any positional axes.

    Args:
      *dims: dimension names that appear on this array, in the desired order on
        the result. `...` may be used once, to indicate all other dimensions in
        order of appearance on this array.

    Returns:
      Array with transposed data and reordered dimensions, as indicated.
    """
    if any(dim is None for dim in self.dims):
      raise ValueError(
          'cannot reorder the dimensions of an array with unnamed '
          f'dimensions: {self.dims}'
      )

    ellipsis_count = sum(dim is ... for dim in dims)
    if ellipsis_count > 1:
      raise ValueError(
          f'dimension names contain multiple ellipses (...): {dims}'
      )
    elif ellipsis_count == 1:
      explicit_dims = {dim for dim in dims if dim is not ...}
      implicit_dims = tuple(
          dim for dim in self.dims if dim not in explicit_dims
      )
      i = dims.index(...)
      dims = dims[:i] + implicit_dims + dims[i + 1 :]

    order = tuple(self.dims.index(dim) for dim in dims)
    return type(self)(self.data.transpose(order), dims)

  # Convenience wrappers: Elementwise infix operators.
  __lt__ = _nmap_binary_op(operator.lt, 'jax.Array.__lt__')
  __le__ = _nmap_binary_op(operator.le, 'jax.Array.__le__')
  __eq__ = _nmap_binary_op(operator.eq, 'jax.Array.__eq__')
  __ne__ = _nmap_binary_op(operator.ne, 'jax.Array.__ne__')
  __ge__ = _nmap_binary_op(operator.ge, 'jax.Array.__ge__')
  __gt__ = _nmap_binary_op(operator.gt, 'jax.Array.__gt__')

  __add__ = _nmap_binary_op(operator.add, 'jax.Array.__add__')
  __sub__ = _nmap_binary_op(operator.sub, 'jax.Array.__sub__')
  __mul__ = _nmap_binary_op(operator.mul, 'jax.Array.__mul__')
  __truediv__ = _nmap_binary_op(operator.truediv, 'jax.Array.__truediv__')
  __floordiv__ = _nmap_binary_op(operator.floordiv, 'jax.Array.__floordiv__')
  __mod__ = _nmap_binary_op(operator.mod, 'jax.Array.__mod__')
  __divmod__ = _nmap_binary_op_two_outputs(divmod, 'jax.Array.__divmod__')
  __pow__ = _nmap_binary_op(operator.pow, 'jax.Array.__pow__')
  __lshift__ = _nmap_binary_op(operator.lshift, 'jax.Array.__lshift__')
  __rshift__ = _nmap_binary_op(operator.rshift, 'jax.Array.__rshift__')
  __and__ = _nmap_binary_op(operator.and_, 'jax.Array.__and__')
  __or__ = _nmap_binary_op(operator.or_, 'jax.Array.__or__')
  __xor__ = _nmap_binary_op(operator.xor, 'jax.Array.__xor__')

  __radd__ = _nmap_binary_op(_swapped_binop(operator.add), 'jax.Array.__radd__')
  __rsub__ = _nmap_binary_op(_swapped_binop(operator.sub), 'jax.Array.__rsub__')
  __rmul__ = _nmap_binary_op(_swapped_binop(operator.mul), 'jax.Array.__rmul__')
  __rtruediv__ = _nmap_binary_op(
      _swapped_binop(operator.truediv), 'jax.Array.__rtruediv__'
  )
  __rfloordiv__ = _nmap_binary_op(
      _swapped_binop(operator.floordiv), 'jax.Array.__rfloordiv__'
  )
  __rmod__ = _nmap_binary_op(_swapped_binop(operator.mod), 'jax.Array.__rmod__')
  __rdivmod__ = _nmap_binary_op_two_outputs(
      _swapped_binop(divmod), 'jax.Array.__rdivmod__'
  )
  __rpow__ = _nmap_binary_op(_swapped_binop(operator.pow), 'jax.Array.__rpow__')
  __rlshift__ = _nmap_binary_op(
      _swapped_binop(operator.lshift), 'jax.Array.__rlshift__'
  )
  __rrshift__ = _nmap_binary_op(
      _swapped_binop(operator.rshift), 'jax.Array.__rrshift__'
  )
  __rand__ = _nmap_binary_op(
      _swapped_binop(operator.and_), 'jax.Array.__rand__'
  )
  __ror__ = _nmap_binary_op(_swapped_binop(operator.or_), 'jax.Array.__ror__')
  __rxor__ = _nmap_binary_op(_swapped_binop(operator.xor), 'jax.Array.__rxor__')

  __abs__ = _nmap_unary_op(operator.abs, 'jax.Array.__abs__')
  __neg__ = _nmap_unary_op(operator.neg, 'jax.Array.__neg__')
  __pos__ = _nmap_unary_op(operator.pos, 'jax.Array.__pos__')
  __invert__ = _nmap_unary_op(operator.inv, 'jax.Array.__invert__')

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
  imag = property(_wrap_array_method('imag', from_property=True))
  real = property(_wrap_array_method('real', from_property=True))
  round = _wrap_array_method('round')
  view = _wrap_array_method('view')


PyTree = Any


def is_namedarray(array: Any) -> TypeGuard[NamedArray]:
  return isinstance(array, NamedArray)


def tag(tree: PyTree, *dims: str | ellipsis | None) -> PyTree:
  """Tag dimensions on all NamedArrays in a PyTree."""
  tag_arrays = lambda x: x.tag(*dims) if is_namedarray(x) else x
  return jax.tree.map(tag_arrays, tree, is_leaf=is_namedarray)


def untag(tree: PyTree, *dims: str) -> PyTree:
  """Untag dimensions from all NamedArrays in a PyTree."""
  untag_arrays = lambda x: x.untag(*dims) if is_namedarray(x) else x
  return jax.tree.map(untag_arrays, tree, is_leaf=is_namedarray)
