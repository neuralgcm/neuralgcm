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

"""Utility functions that operate on pytrees."""

from collections import abc
import dataclasses
import functools
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import typing
import numpy as np


def pack_pytree(pytree: typing.Pytree, axis: int = -3) -> typing.Array:
  """Packs `pytree` by concatenating leaves along `axis`."""
  flat, _ = jax.tree.flatten(pytree)
  if not flat:
    return None  # pytype: disable=bad-return-type  # jax-ndarray
  packed = jnp.concatenate(flat, axis)
  return packed


def unpack_to_pytree(
    array: typing.Array, pytree_of_shapes: typing.Pytree, axis: int = -3
) -> typing.Pytree:
  """Unpacks an `array` into a pytree with shapes `pytree_of_shapes`."""
  shapes, tree_def = jax.tree.flatten(pytree_of_shapes)
  shapes = [x.shape for x in shapes]
  splits = np.cumsum(np.array([x[axis] for x in shapes]))[:-1]
  split = jnp.split(array, splits, axis)
  return jax.tree.unflatten(tree_def, split)


def expand_to_ndim(
    pytree: typing.Pytree, ndim: int, axis: int
) -> typing.Pytree:
  """Expands leaves of `pytree` of rank `ndim-1` to `ndim` along `axis`."""
  axis = _normalize_axis(axis, ndim)
  flat, treedef = jax.tree.flatten(pytree)
  ndims = [x.ndim for x in flat]
  if not set(ndims).issubset(set([ndim, ndim - 1])):
    raise ValueError(
        f'only arrays with ndim={ndim}/{ndim - 1} are supported, got {ndims=}'
    )

  def _maybe_expand(x):
    if x.ndim == ndim:
      return x
    else:
      if isinstance(x, jax.ShapeDtypeStruct):
        new_shape = x.shape[:axis] + (1,) + x.shape[axis:]
        return jax.ShapeDtypeStruct(new_shape, x.dtype)
      else:
        return jnp.expand_dims(jnp.asarray(x), axis)

  return jax.tree.unflatten(treedef, [_maybe_expand(x) for x in flat])


def squeeze_to_shapes(
    pytree: typing.Pytree, shapes: typing.Pytree, axis: int
) -> typing.Pytree:
  """Squeezes leaves of `pytree` to `shapes` along `axis`."""

  def _maybe_squeeze(
      array: typing.Array, array_struct: typing.ShapeDtypeStruct
  ) -> typing.Array:
    if array.shape == array_struct.shape:
      return array
    elif array.shape[:axis] + array.shape[(axis + 1) :] == array_struct.shape:
      return jnp.squeeze(array, axis)
    raise ValueError(
        f'{array.shape=} cannot be transformed into'
        f' {array_struct.shape=} by squeezing along {axis=}'
    )

  return jax.tree.map(_maybe_squeeze, pytree, shapes)


def stack_pytree(pytree: typing.Pytree, axis: int = 0) -> typing.Array:
  """Stacks `pytree` by stacking leaves along a new axis."""
  flat, _ = jax.tree.flatten(pytree)
  if not flat:
    return None  # pytype: disable=bad-return-type  # jax-ndarray
  stacked = jnp.stack(flat, axis)
  return stacked


def unstack_to_pytree(
    array: typing.Array, pytree_of_shapes: typing.Pytree, axis: int = 0
) -> typing.Pytree:
  """Unstacks an `array` into a pytree with shapes `pytree_of_shapes`."""
  _, tree_def = jax.tree.flatten(pytree_of_shapes)
  # `array` is split along `axis` and resulting singleton dimension is removed
  split = jnp.split(array, array.shape[axis], axis)
  split = jax.tree.map(lambda x: jnp.squeeze(x, axis=axis), split)
  return jax.tree.unflatten(tree_def, split)


def tree_map_where(
    condition_fn: Callable[[typing.Array], typing.Array],
    f: Callable[[typing.Array], typing.Array],
    g: Callable[[typing.Array], typing.Array],
    x: typing.Pytree,
) -> typing.Pytree:
  """Map `f` over pytree leaves if condition_fn(leaf), else use `g`."""

  def tm_where(x: typing.Array) -> typing.Array:
    return f(x) if condition_fn(x) else g(x)

  return jax.tree.map(tm_where, x)


def tree_map_over_nonscalars(
    f: Callable[[typing.Array], typing.Array],
    x: typing.Pytree,
    *,
    scalar_fn: Callable[[typing.Array], typing.Array] = lambda x: x,
    backend: str = 'jax',
) -> typing.Pytree:
  """Map `f` over nonscalar pytree leaves, but use `scalar_fn` on scalars."""
  as_array_fn = {'jax': jnp.asarray, 'numpy': np.asarray}[backend]

  def g(x: typing.Array) -> typing.Array:
    x = as_array_fn(x)
    return f(x) if x.ndim else scalar_fn(x)

  return jax.tree.map(g, x)


def shape_structure(inputs):
  """Returns `inputs` with leaves replaced by arrays of corresponding shapes."""
  return jax.eval_shape(lambda x: x, inputs)


def _normalize_axis(axis: int, ndim: int) -> int:
  """Validates and returns positive `axis` value."""
  if not -ndim <= axis < ndim:
    raise ValueError(f'invalid axis {axis} for ndim {ndim}')
  if axis < 0:
    axis += ndim
  return axis


def slice_along_axis(
    inputs: typing.Pytree,
    axis: int,
    idx: int | slice,
    expect_same_dims: bool = False,
) -> typing.Pytree:
  """Returns slice of `inputs` defined by `idx` along axis `axis`.

  Args:
    inputs: pytree to slice.
    axis: axis along which to slice `inputs`.
    idx: index or slice along axis `axis` that is returned.
    expect_same_dims: whether all arrays should have same number of dimensions.

  Returns:
    Slice of `inputs` defined by `idx` along axis `axis`. If idx is an int,
    the axis is dropped. If idx is a slice, the axis is handled accordingly.
  """
  arrays, tree_def = jax.tree.flatten(inputs)
  ndims = set(a.ndim for a in arrays)
  if expect_same_dims and len(ndims) != 1:
    raise ValueError(
        'arrays in `inputs` expected to have same ndims, but have '
        f'{ndims}. To allow this, pass expect_same_dims=False'
    )
  elif axis < 0:
    raise ValueError(f'Using {axis=} is error-prone if expect_same_dims=False')
  sliced = []
  for array in arrays:
    ndim = array.ndim
    slc = tuple(
        idx if j == _normalize_axis(axis, ndim) else slice(None)
        for j in range(ndim)
    )
    sliced.append(array[slc])
  return jax.tree.unflatten(tree_def, sliced)


def split_along_axis(
    inputs: typing.Pytree,
    split_idx: int,
    axis: int,
    expect_same_dims: bool = False,
) -> tuple[typing.Pytree, typing.Pytree]:
  """Returns 2 pytrees formed by splitting `inputs` along `axis` at `split_idx`.

  Args:
    inputs: pytree to split.
    split_idx: index along `axis` where the second split starts.
    axis: axis along which to split the `inputs`.
    expect_same_dims: whether all arrays should have same number of dimensions.

  Returns:
    Tuple of slices of `inputs` split along `axis` at `split_idx`.
  """

  first_slice = slice_along_axis(
      inputs, axis, slice(0, split_idx), expect_same_dims
  )
  second_slice = slice_along_axis(
      inputs, axis, slice(split_idx, None), expect_same_dims
  )
  return first_slice, second_slice


def split_axis(
    inputs: typing.Pytree,
    axis: int,
    keep_dims: bool = False,
) -> tuple[typing.Pytree, ...]:
  """Splits `inputs` along `axis`.

  Args:
    inputs: pytree to be split.
    axis: axis along which to split the `inputs`.
    keep_dims: whether to keep `axis` dimension.

  Returns:
    Tuple of pytrees that correspond to slices of `inputs` along `axis`. The
    `axis` dimension is removed if `squeeze is set to True.

  Raises:
    ValueError: if arrays in `inputs` don't have unique size along `axis`.
  """
  arrays, tree_def = jax.tree.flatten(inputs)
  axis_shapes = set(a.shape[axis] for a in arrays)
  if len(axis_shapes) != 1:
    raise ValueError(f'Arrays must have equal sized axis but got {axis_shapes}')
  (axis_shape,) = axis_shapes
  splits = [jnp.split(a, axis_shape, axis=axis) for a in arrays]
  if not keep_dims:
    splits = jax.tree.map(lambda a: jnp.squeeze(a, axis), splits)
  splits = zip(*splits)
  return tuple(jax.tree.unflatten(tree_def, leaves) for leaves in splits)


def concat_along_axis(
    pytrees: Sequence[typing.Pytree], axis: int
) -> typing.Pytree:
  """Concatenates `pytrees` along `axis`."""
  concat_leaves_fn = lambda *args: jnp.concatenate(args, axis)
  return jax.tree.map(concat_leaves_fn, *pytrees)


def as_dict(inputs: typing.Pytree) -> typing.Pytree:
  """Returns a dict representation of `inputs` and a from_dict_fn."""
  return_type = type(inputs)
  if dataclasses.is_dataclass(inputs):
    inputs = inputs.asdict()
  else:
    if return_type != dict:
      raise ValueError(f'Inputs of type {return_type} are not supported.')

  from_dict_fn = lambda dict_inputs: return_type(**dict_inputs)
  return inputs, from_dict_fn


def none_to_zeros(
    tree: typing.Pytree,
    reference_tree: typing.Pytree,
) -> typing.Pytree:
  """Returns tree where `None` is replaced with zeros_like of reference_tree."""
  return jax.tree.map(
      lambda x, y: jnp.zeros_like(y) if x is None else x, tree, reference_tree
  )


@dataclasses.dataclass(frozen=True)
class _HashableNDArrayWrapper:
  shape: tuple[int, ...]
  dtype: np.dtype
  data: bytes


def _hash_leaf(x: typing.Pytree) -> abc.Hashable:
  if isinstance(x, (jax.Array, np.ndarray)):
    return _HashableNDArrayWrapper(x.shape, x.dtype, x.tobytes())
  else:
    return x


def tree_hashable(x: typing.Pytree) -> abc.Hashable:
  """Convert a pytree into something hashable."""
  values, treedef = jax.tree.flatten(x)
  values = tuple(map(_hash_leaf, values))
  return values, treedef


def tree_cache(func):
  """Like functools.cache, but hashes with tree_hashable.

  Example usage::

    import numpy as np

    @tree_cache
    def f(x):
      print('caching')
      return x

    >>> f({'a': 1, 'b': np.arange(3)})
    caching
    {'a': 1, 'b': array([0, 1, 2])}
    >>> f({'a': 1, 'b': np.arange(3)})
    {'a': 1, 'b': array([0, 1, 2])}

  Args:
    func: function to cache.

  Returns:
    Function where cached results are reused.
  """
  results = {}

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    key = tree_hashable((args, kwargs))
    if key in results:
      return results[key]
    result = results[key] = func(*args, **kwargs)
    return result

  return wrapper


def flatten_dict(
    input_dict: dict[str, Any],
    prefix: str = '',
    sep: str = '&',
) -> tuple[dict[str, Any], tuple[str, ...]]:
  """Flattens potentially nested `input_dict`."""
  items = []
  empty_keys = []
  for k, v in input_dict.items():
    if sep in k:
      raise ValueError(f'Key {k} contains {sep=}. Use different name or sep.')
    new_key = prefix + sep + k if prefix else k
    if isinstance(v, dict) and v:
      sub_dict, sub_empty_keys = flatten_dict(v, new_key, sep=sep)
      items.extend(sub_dict.items())
      empty_keys.extend(sub_empty_keys)
    elif isinstance(v, dict) and not v:
      empty_keys.append(new_key)
    else:
      items.append((new_key, v))
  unique_keys, counts = np.unique(
      np.array([x[0] for x in items]), return_counts=True
  )
  if (counts > 1).any():
    raise ValueError(f'got duplicate keys {unique_keys[counts > 1]}')
  unique_empty_keys, counts = np.unique(
      np.array([x[0] for x in empty_keys]), return_counts=True
  )
  if (counts > 1).any():
    raise ValueError(f'got duplicate keys {unique_empty_keys[counts > 1]}')
  return dict(items), tuple(empty_keys)


def unflatten_dict(
    flat_dict: dict[str, Any],
    empty_keys: tuple[str, ...] = tuple(),
    sep: str = '&',
) -> dict[str, Any]:
  """Unflattens `flat_dict` with structure specified with separataion `sep`."""
  result = dict()
  empty_key_dict = {k: {} for k in empty_keys}
  for key, value in (flat_dict | empty_key_dict).items():
    sub_keys = key.split(sep)
    sub_dict = result
    for sub_key in sub_keys[:-1]:
      if sub_key in sub_dict:
        sub_dict = sub_dict[sub_key]
      else:
        sub_dict[sub_key] = dict()
        sub_dict = sub_dict[sub_key]
    sub_dict[sub_keys[-1]] = value
  return result


def replace_with_matching_or_default(
    x: dict[str, Any],
    replace: dict[str, Any],
    default: Any = None,
    check_used_all_replace_keys: bool = True,
) -> dict[str, Any]:
  """Returns `x` structure with leaves from `replace` or `default`."""
  flat_x, empty_keys = flatten_dict(x)
  flat_replace, _ = flatten_dict(replace)
  if check_used_all_replace_keys:
    unused_replace_keys = set(flat_replace.keys()) - set(flat_x.keys())
    if unused_replace_keys:
      raise ValueError(f'Keys {unused_replace_keys} not present in {x.keys()=}')
  flat_result = {k: flat_replace.get(k, default) for k in flat_x.keys()}
  return unflatten_dict(flat_result, empty_keys)


def map_over_matching_keys(
    inputs: dict[str, Any],
    fn: Callable[[typing.Array], typing.Array],
    keys_to_map_over: Sequence[str],
) -> dict[str, Any]:
  """Applies `fn` to values in `inputs` or sub-dictionaries for matching keys.

  Args:
    inputs: potentially nested dictionary to map over.
    fn: function to apply to elements in the `inputs` that have matching keys.
    keys_to_map_over: keys to which `fn` should be applied to.

  Returns:
    `inputs` with all elements that have keys matching an entry in
    `keys_to_map_over` transformed by function `fn`.
  """
  outputs = {}
  for k, v in inputs.items():
    if not isinstance(v, dict):
      outputs[k] = fn(v) if k in keys_to_map_over else v
    else:
      outputs[k] = map_over_matching_keys(v, fn, keys_to_map_over)
  return outputs
