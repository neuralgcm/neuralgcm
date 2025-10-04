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

"""Utilities for manipulating and transforming modules."""

from __future__ import annotations
import dataclasses
import functools
from typing import Iterable, NamedTuple, Sequence
import chex
import coordax as cx
from flax import nnx
import jax
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import typing


def ensure_unchanged_state_structure(
    method=None, *, excluded_dims: Sequence[str] | None = None
):
  """Wraps `method` of a nnx.Module checking that pytree struct is unchanged.

  The check is performed by comparing the coordinate structure of the module
  state before and after calling the `method`. Coordinates with dimension names
  in excluded_dims are only checked for existence by squeezing coordinates to
  size 1. This enables checks on methods where a subset of dimensions may
  change shape, e.g. updating dynamic state of the model.

  Args:
    method: The method to wrap. If None, works as a decorator.
    excluded_dims: Dimensions to exclude from the coordinate structure check.

  Returns:
    The wrapped method or a decorator.
  """
  excluded_dims = excluded_dims or []

  if method is None:
    return functools.partial(
        ensure_unchanged_state_structure, excluded_dims=excluded_dims
    )

  def _get_coord_struct(pytree: typing.Pytree) -> typing.Pytree:
    is_coord = lambda x: isinstance(x, cx.Coordinate)
    to_coord = lambda c: c.coordinate if cx.is_field(c) else c

    def squeeze_excluded(c: cx.Coordinate) -> cx.Coordinate:
      if not is_coord(c):
        return c
      axes = [
          cx.DummyAxis(ax.dims[0], 1) if ax.dims[0] in excluded_dims else ax
          for ax in c.axes
      ]
      return cx.compose_coordinates(*axes)

    field_struct = pytree_utils.shape_structure(pytree)
    coord_struct = jax.tree.map(to_coord, field_struct, is_leaf=cx.is_field)
    coord_struct = jax.tree.map(
        squeeze_excluded, coord_struct, is_leaf=is_coord
    )
    return coord_struct

  @functools.wraps(method)
  def wrapper(module: nnx.Module, *args, **kwargs):
    if not isinstance(module, nnx.Module):
      raise TypeError(
          '`ensure_unchanged_state_structure` must wrap an nnx.Module method'
      )
    graph_def_before, state_before = nnx.split(module)
    state_before = _get_coord_struct(state_before)
    result = method(module, *args, **kwargs)  # runs the method.
    graph_def_after, state_after = nnx.split(module)
    state_after = _get_coord_struct(state_after)
    if graph_def_after != graph_def_before:
      raise ValueError(
          f'GraphDef changed: {graph_def_before=} {graph_def_after=}'
      )
    try:
      chex.assert_trees_all_equal_shapes_and_dtypes(state_before, state_after)
    except (AssertionError, ValueError) as e:
      raise ValueError(
          'change in the pytree structure detected while running'
          f' "{method.__name__}":\n{e}'
      ) from e
    return result

  return wrapper


def vectorize_module(
    module: nnx.Module,
    vectorization_specs: dict[nnx.filterlib.Filter, cx.Coordinate],
) -> None:
  """Vectorizes the state of a `module` in place using `vectorization_specs`."""

  def broadcast(x: cx.Field, coord: cx.Coordinate) -> cx.Field:
    if not cx.is_field(x):
      raise ValueError(
          'module state vectorization requires Field variables, but'
          f' encountered {type(x)=}'
      )
    return x.broadcast_like(cx.compose_coordinates(coord, x.coordinate))

  for k, coord in vectorization_specs.items():
    k_state = jax.tree.map(
        functools.partial(broadcast, coord=coord),
        nnx.state(module, k),
        is_leaf=cx.is_field,
    )
    nnx.update(module, k_state)


class ModuleAndMethod(NamedTuple):
  module: nnx.Module
  method_name: str


def format_callbacks(callback_specs):
  """Formats callback_specs to standardized format."""
  if isinstance(callback_specs, ModuleAndMethod):
    return callback_specs
  if isinstance(callback_specs, nnx.Module):  # single callback.
    return ModuleAndMethod(callback_specs, '__call__')  # call default method.
  if isinstance(callback_specs, Iterable) and (len(callback_specs) == 2):
    return ModuleAndMethod(*callback_specs)
  raise TypeError(f'Unexpected {type(callback_specs)=}')


def with_callback(
    module,
    callback_specs: ModuleAndMethod,
    method_name: str = '__call__',
):
  """Returns module with `callback_specs.module` attached to `method_name`."""
  base_class = type(module)

  def __init__(self, wrapped_instance, callback_specs):  # pylint: disable=invalid-name
    self.wrapped_instance = wrapped_instance
    self.callback_specs = format_callbacks(callback_specs)

  def __getattr__(self, attr_name):  # pylint: disable=invalid-name
    """Delegate attribute access to the wrapped instance."""
    return getattr(self.wrapped_instance, attr_name)

  @functools.wraps(getattr(base_class, method_name))
  def wrapped_fn(self, *args, **kwargs):
    result = getattr(self.wrapped_instance, method_name)(*args, **kwargs)
    # The reason we use getattr here is because we need to access of method of
    # the callback module that is an attribute of this module. Otherwise nnx
    # would raise an error informing that we are trying to mutate an object that
    # is out of current scope. (This is exactly what would happen if we added
    # a reference to a callback_module.method as attribute of this class.)
    callback_fn = getattr(
        self.callback_specs.module, self.callback_specs.method_name
    )
    callback_fn(result, *args, **kwargs)
    return result

  attrs = {
      '__init__': __init__,
      '__getattr__': __getattr__,
      method_name: wrapped_fn,
  }
  if dataclasses.is_dataclass(base_class):
    for field in dataclasses.fields(base_class):
      attrs[field.name] = property(
          lambda self, field=field: getattr(self.wrapped_instance, field.name)
      )
  cls = type(base_class.__name__ + 'WithCallbacks', (base_class,), attrs)
  return cls(module, callback_specs)


def retrieve_subclass_modules(module, subclass):
  """Returns list of all unique `subclass` instances on `module`."""
  subclass_modules = []
  for _, x in module.iter_modules():
    if isinstance(x, subclass):
      subclass_modules.append(x)
  return subclass_modules
