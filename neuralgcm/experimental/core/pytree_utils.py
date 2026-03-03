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
from typing import Any, Callable

import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import typing
import numpy as np


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
