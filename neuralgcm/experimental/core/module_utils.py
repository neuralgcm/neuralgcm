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
import itertools
import operator
from typing import Any, Callable, Iterable, NamedTuple, Sequence, overload

import chex
import coordax as cx
from flax import nnx
import jax
from neuralgcm.experimental.core import field_utils
from neuralgcm.experimental.core import parallelism
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


def untag_module_state(
    module: nnx.Module,
    coordinate: cx.Coordinate,
    vectorized_axes: dict[nnx.filterlib.Filter, cx.Coordinate],
    skip_missing: bool = False,
) -> None:
  """Untags axes of `coordinate` from the vectorized state of the `module`.

  Performs in-place untagging of vectorization axes from the `module` state. For
  each vectorization slice specified by `vectorized_axes`, untags axes from
  `coordinate.axes` that are present in the associated state. If `coordinate`
  has axes that are not present in anywhere in `vectorized_axes`, those axes
  are ignored if `skip_missing` is True, otherwise an error is raised.

  Args:
    module: The module whose state will be untagged.
    coordinate: The coordinate axes to be untagged from the vectorized state.
    vectorized_axes: A dictionary mapping state filters to the coordinates
      representing vectorization axes for the associated state slice.
    skip_missing: If True, axes in `coordinate` that are not present in any
      of the coordinates in `vectorized_axes` will be ignored. Otherwise, a
      ValueError will be raised.
  """
  vectorized_axes_set = functools.reduce(
      operator.or_, (set(v.axes) for v in vectorized_axes.values()), set()
  )
  axes_not_in_vectorized_axes = set(coordinate.axes) - vectorized_axes_set
  if axes_not_in_vectorized_axes and not skip_missing:
    raise ValueError(
        f'untag_module_state got {coordinate=} with axes'
        f' ["{axes_not_in_vectorized_axes}"] that are not present in'
        f' {vectorized_axes=}'
    )
  for state_filter, coord in vectorized_axes.items():
    untag_components = [ax for ax in coordinate.axes if ax in coord.axes]
    if untag_components:
      untag_axis = cx.compose_coordinates(*untag_components)
      state_to_untag = nnx.state(module, state_filter)
      nnx.update(module, cx.untag(state_to_untag, untag_axis))


def tag_module_state(
    module: nnx.Module,
    coordinate: cx.Coordinate,
    vectorized_axes: dict[nnx.filterlib.Filter, cx.Coordinate],
    skip_missing: bool = False,
) -> None:
  """Tags axes of `coordinate` to the state of the `module`.

  Performs in-place tagging of vectorization axes to the `module` state. For
  each vectorization slice specified by `vectorized_axes`, tags axes from
  `coordinate.axes` that are present in the associated state. If `coordinate`
  has axes that are not present in anywhere in `vectorized_axes`, those axes
  are ignored if `skip_missing` is True, otherwise an error is raised.

  Args:
    module: The module whose state will be tagged.
    coordinate: The coordinate axes to be tagged to the vectorized state.
    vectorized_axes: A dictionary mapping state filters to the coordinates
      representing vectorization axes for the associated state slice.
    skip_missing: If True, axes in `coordinate` that are not present in any
      of the coordinates in `vectorized_axes` will be ignored. Otherwise, a
      ValueError will be raised.
  """
  vectorized_axes_set = functools.reduce(
      operator.or_, (set(v.axes) for v in vectorized_axes.values()), set()
  )
  axes_not_in_vectorized_axes = set(coordinate.axes) - vectorized_axes_set
  if axes_not_in_vectorized_axes and not skip_missing:
    raise ValueError(
        f'tag_module_state got {coordinate=} with axes'
        f' ["{axes_not_in_vectorized_axes}"] that are not present in'
        f' {vectorized_axes=}'
    )
  for state_filter, coord in vectorized_axes.items():
    tag_components = [ax for ax in coordinate.axes if ax in coord.axes]
    if tag_components:
      tag_axis = cx.compose_coordinates(*tag_components)
      state_to_untag = nnx.state(module, state_filter)
      nnx.update(module, cx.tag(state_to_untag, tag_axis))


def _are_certainly_disjoint_predicates(
    p1: nnx.Predicate, p2: nnx.Predicate
) -> bool:
  """Returns True if we can guarantee that two predicates are disjoint."""
  # Note this implementation assumes deconstruction of p1, some cases can be
  # handled by considering disjointness of (p2, p1).
  if isinstance(p1, nnx.filterlib.Nothing):  # Handle Nothing.
    return True

  if isinstance(p1, nnx.filterlib.Everything):  # Handle Everything.
    return isinstance(p2, nnx.filterlib.Nothing)

  if isinstance(p1, nnx.filterlib.Not):  # Handle Not.
    if p1.predicate == p2:
      return True

  if isinstance(p1, nnx.filterlib.Any):  # Handle Any.
    return all(
        _are_certainly_disjoint_filters(sub_p, p2) for sub_p in p1.predicates
    )

  if isinstance(p1, nnx.filterlib.All):  # Handle All
    return any(
        _are_certainly_disjoint_filters(sub_p, p2) for sub_p in p1.predicates
    )

  if isinstance(p1, nnx.filterlib.OfType):
    if isinstance(p2, nnx.filterlib.OfType):
      t1, t2 = p1.type, p2.type  # check if filters are in subclass relation.
      if not issubclass(t1, t2) and not issubclass(t2, t1):
        return True

  if isinstance(p1, nnx.filterlib.WithTag):  # Handle WithTag.
    if isinstance(p2, nnx.filterlib.WithTag):
      if p1.tag != p2.tag:
        return True
  # Other cases are hard to check, so we conservatively return False.
  return False


def _is_certainly_subset_predicate(
    p1: nnx.Predicate, p2: nnx.Predicate
) -> bool:
  """Returns True if we can guarantee that p1 is a subset of p2."""
  if isinstance(p2, nnx.filterlib.Everything):
    return True
  if isinstance(p1, nnx.filterlib.Nothing):
    return True
  if p1 == p2:
    return True

  if isinstance(p1, nnx.filterlib.OfType) and isinstance(
      p2, nnx.filterlib.OfType
  ):
    if issubclass(p1.type, p2.type):
      return True

  if isinstance(p1, nnx.filterlib.Any):
    return all(
        _is_certainly_subset_predicate(sub_p, p2) for sub_p in p1.predicates
    )

  if isinstance(p2, nnx.filterlib.Any):
    return any(
        _is_certainly_subset_predicate(p1, sub_p) for sub_p in p2.predicates
    )

  if isinstance(p1, nnx.filterlib.All):
    return any(
        _is_certainly_subset_predicate(sub_p, p2) for sub_p in p1.predicates
    )

  if isinstance(p2, nnx.filterlib.All):
    return all(
        _is_certainly_subset_predicate(p1, sub_p) for sub_p in p2.predicates
    )

  if isinstance(p1, nnx.filterlib.Not) and isinstance(p2, nnx.filterlib.Not):
    return _is_certainly_subset_predicate(p2.predicate, p1.predicate)

  return False


def _are_certainly_disjoint_filters(
    filter_a: nnx.filterlib.Filter, filter_b: nnx.filterlib.Filter
) -> bool:
  """Returns True if two filters can be guaranteed to be disjoint.

  Two filters are disjoint if there is no variable that can be matched by both.
  This function provides a best-effort check based on the filter types.
  It cannot prove disjointness for arbitrary callable filters.

  Args:
    filter_a: The first filter.
    filter_b: The second filter.

  Returns:
    True if the check determines that the filters are disjoint, False otherwise.
  """
  p1 = nnx.filterlib.to_predicate(filter_a)
  p2 = nnx.filterlib.to_predicate(filter_b)
  ab_direction_disjoint = _are_certainly_disjoint_predicates(p1, p2)
  ba_direction_disjoint = _are_certainly_disjoint_predicates(p2, p1)
  return ab_direction_disjoint or ba_direction_disjoint


def is_filter_subset(
    f: nnx.filterlib.Filter, filter_group: nnx.filterlib.Filter
) -> bool:
  """Returns True if `f` can be guaranteed to be a subset of `filter_group`."""
  p1 = nnx.filterlib.to_predicate(f)
  p2 = nnx.filterlib.to_predicate(filter_group)
  return _is_certainly_subset_predicate(p1, p2)


def merge_vectorized_axes(
    vectorized_axes_head: dict[nnx.filterlib.Filter, cx.Coordinate],
    vectorized_axes_tail: dict[nnx.filterlib.Filter, cx.Coordinate],
) -> dict[nnx.filterlib.Filter, cx.Coordinate]:
  """Returns merged vectorized axes with head specifying leading dimensions."""
  head = vectorized_axes_head.copy()
  tail = vectorized_axes_tail.copy()
  head_ellipsis_axes = head.pop(..., cx.Scalar())
  tail_ellipsis_axes = tail.pop(..., cx.Scalar())
  # Split keys into common and differences
  head_keys = set(head.keys())
  tail_keys = set(tail.keys())
  common_keys = head_keys.intersection(tail_keys)
  diff_head_keys = head_keys.difference(tail_keys)
  diff_tail_keys = tail_keys.difference(head_keys)
  merged = {k: cx.compose_coordinates(head[k], tail[k]) for k in common_keys}
  diff_head = {
      k: cx.compose_coordinates(head[k], tail_ellipsis_axes)
      for k in diff_head_keys
  }
  diff_tail = {
      k: cx.compose_coordinates(head_ellipsis_axes, tail[k])
      for k in diff_tail_keys
  }
  if not all(
      _are_certainly_disjoint_filters(k1, k2)
      for k1, k2 in itertools.product(diff_head_keys, diff_tail_keys)
  ):
    potentially_overlapping = next(
        (k1, k2)
        for k1, k2 in itertools.product(diff_head_keys, diff_tail_keys)
        if not _are_certainly_disjoint_filters(k1, k2)
    )
    raise ValueError(
        'Cannot merge vectorized axes with potentially overlapping filters: '
        f'{potentially_overlapping[0]!r} and {potentially_overlapping[1]!r}.'
    )
  merged.update(diff_head)
  merged.update(diff_tail)
  combined_ellipsis = cx.compose_coordinates(
      head_ellipsis_axes, tail_ellipsis_axes
  )
  # Add ellipsis back if we had it in either set of filters.
  if ... in vectorized_axes_head or ... in vectorized_axes_tail:
    merged[...] = combined_ellipsis

  return merged


@overload
def state_in_axes_for_coord(
    vectorized_axes: dict[nnx.filterlib.Filter, cx.Coordinate],
    coord: cx.Coordinate,
) -> nnx.StateAxes:
  ...


@overload
def state_in_axes_for_coord(
    vectorized_axes: dict[nnx.filterlib.Filter, cx.Coordinate],
    coord: Sequence[cx.Coordinate],
) -> Sequence[nnx.StateAxes]:
  ...


def state_in_axes_for_coord(
    vectorized_axes: dict[nnx.filterlib.Filter, cx.Coordinate],
    coord: cx.Coordinate | Sequence[cx.Coordinate],
) -> nnx.StateAxes | Sequence[nnx.StateAxes]:
  if isinstance(coord, Sequence):
    return nest_state_in_axes(
        *(state_in_axes_for_coord(vectorized_axes, c) for c in coord)
    )
  dummy = {k: cx.shape_struct_field(v) for k, v in vectorized_axes.items()}
  axes = {k: field_utils.in_axes_for_coord(v, coord) for k, v in dummy.items()}
  return nnx.StateAxes(axes)


def nest_state_in_axes(
    *state_axes_to_nest: nnx.StateAxes,
) -> tuple[nnx.StateAxes, ...]:
  """Returns `state_axes_to_nest` adjusted for vmap nesting from outer to inner.

  Args:
    *state_axes_to_nest: A sequence of `nnx.StateAxes` with equal keys
      representing `vmap` indices from outermost to innermost.

  Returns:
    A tuple of adjusted `nnx.StateAxes` for each level of `vmap`.
  """
  if not state_axes_to_nest:
    return ()

  state_filters = {tuple(x.keys()) for x in state_axes_to_nest}
  if len(state_filters) != 1:
    raise ValueError(
        f'nesting state_in_axes requires same keys, got {state_filters}'
    )
  [state_filters] = list(state_filters)
  axes_by_filter = {
      f: tuple(s[f] for s in state_axes_to_nest) for f in state_filters
  }
  nested_axes_by_filter = {
      f: field_utils.nest_in_axes(*trees) for f, trees in axes_by_filter.items()
  }
  return tuple(
      nnx.StateAxes({f: nested_axes_by_filter[f][i] for f in state_filters})
      for i in range(len(state_axes_to_nest))
  )


def vectorize_module_fn(
    fn: Callable[..., Any],
    vector_axes: dict[nnx.filterlib.Filter, cx.Coordinate],
    axes_to_vectorize: cx.Coordinate | Sequence[cx.Coordinate],
    custom_spmd_axis_names: None | str | tuple[str, ...] = None,
    custom_axis_names: None | str | tuple[str, ...] = None,
    mesh: parallelism.Mesh | None = None,
    allow_non_vector_axes: bool = False,
) -> Callable[..., Any]:
  """Returns a wrapped `fn` that vectorizes `fn` over `axes_to_vectorize`.

  This helper generalizes the process of applying `nnx.vmap` over a function
  that accepts a module and a variable number of arguments. It correctly
  constructs the `in_axes` for the module and each argument, nests the `vmap`
  transformations, and handles the necessary tagging and untagging of inputs
  and outputs.

  Args:
    fn: The function to vectorize, must have signature `fn(module, *args)`.
    vector_axes: A dictionary describing how module state is vectorized.
    axes_to_vectorize: A single coordinate or a sequence of coordinates
      representing the axes to vectorize `fn` over.
    custom_spmd_axis_names: Custom spmd_axis_name(s) to use in vmaps. If
      specified, must have length equal to the rank of axes_to_vectorize.
    custom_axis_names: Custom axis_name(s) to use in vmaps. If specified, should
      have length equal to the rank of axes_to_vectorize.
    mesh: Parallelism mesh to infer spmd_axis_name and axis_name vmap arguments
      from axes in axes_to_vectorize. If None, spmd_axis_name and axis_name are
      not inferred. Cannot be specified with custom_spmd_axis_names or
      custom_axis_names.
    allow_non_vector_axes: Whether to allow `axes_to_vectorize` to contain axes
      that do not appear in `vector_axes`. This can be helpful for mapping over
      axes that only appear in args. Defaults to False.

  Returns:
    A wrapped function that applies vectorization.
  """

  @functools.wraps(fn)
  def wrapped_fn(module: nnx.Module, *args: Any) -> Any:
    """Wrapped function that applies vectorization."""
    vmap_axes = axes_to_vectorize
    if isinstance(vmap_axes, Sequence):
      vmap_axes = cx.compose_coordinates(*axes_to_vectorize)
    if isinstance(vmap_axes, cx.Coordinate):
      axes_seq = vmap_axes.axes  # ensures that axes are 1d.
    else:
      raise ValueError(
          f'Unsupported type for `axes_to_vectorize`: {type(vmap_axes)}'
      )

    if not axes_seq:
      return fn(module, *args)

    if mesh is not None and mesh.axis_names:
      if custom_spmd_axis_names is not None or custom_axis_names is not None:
        raise ValueError(
            'Cannot specify both mesh and custom spmd_axis_names or '
            'custom_axis_names.'
        )
      dims_seq = sum([c.dims for c in axes_seq], start=())
      names_in_mesh = [d if d in mesh.axis_names else None for d in dims_seq]
      spmd_axis_names = names_in_mesh
      axis_names = names_in_mesh
    else:
      spmd_axis_names = custom_spmd_axis_names
      axis_names = custom_axis_names

    is_str = lambda x: isinstance(x, str)
    is_sequence = lambda x: isinstance(x, Sequence)
    if spmd_axis_names is None or is_str(spmd_axis_names):
      spmd_axis_names = [spmd_axis_names] * len(axes_seq)
    elif is_sequence(spmd_axis_names):
      if len(spmd_axis_names) != len(axes_seq):
        raise ValueError(
            'custom_spmd_axis_names must have length equal to the rank of '
            f'axes_to_vectorize, got {len(custom_spmd_axis_names)=} and '
            f'{len(axes_seq)=}'
        )
    else:
      raise ValueError(f'Unsupported {type(custom_spmd_axis_names)=}')

    if axis_names is None or is_str(axis_names):
      axis_names = [axis_names] * len(axes_seq)
    elif is_sequence(axis_names):
      if len(axis_names) != len(axes_seq):
        raise ValueError(
            'custom_axis_names must have length equal to the rank of '
            f'axes_to_vectorize, got {len(axis_names)=} and '
            f'{len(axes_seq)=}'
        )
    else:
      raise ValueError(f'Unsupported {type(custom_axis_names)=}')

    # Calculate nested axes for the module state and args.
    map_vector_axes = vector_axes.copy()
    if ... not in map_vector_axes:
      map_vector_axes[...] = cx.Scalar()  # Replicate non-vectorized state.
    nested_model_axes = state_in_axes_for_coord(map_vector_axes, axes_seq)
    nested_args_axes_list = [
        field_utils.in_axes_for_coord(arg, axes_seq) for arg in args
    ]
    vmap_levels_axes = zip(
        reversed(nested_model_axes),
        *(reversed(arg_axes) for arg_axes in nested_args_axes_list),
    )
    reversed_axis_names = tuple(reversed(axis_names))
    reversed_spmd_axis_names = tuple(reversed(spmd_axis_names))

    vmapped_fn = fn
    for i, (state_axes, *current_args_axes) in enumerate(vmap_levels_axes):
      # `in_axes` is a tuple where the first element corresponds to the module's
      # state and the rest correspond to the `*args`.
      in_axes = (state_axes, *current_args_axes)
      axis_name = reversed_axis_names[i]  # mapping happens in reverse order.
      spmd_axis_name = reversed_spmd_axis_names[i]
      vmapped_fn = nnx.vmap(
          vmapped_fn,
          in_axes=in_axes,
          axis_name=axis_name,
          spmd_axis_name=spmd_axis_name,
      )

    def _untag_coord_in_any_order(tree, c):
      untag_f = lambda x: x.untag(*sorted(c.axes, key=x.coordinate.axes.index))
      untag_arrays = lambda x: untag_f(x) if cx.is_field(x) else x
      return jax.tree.map(untag_arrays, tree, is_leaf=cx.is_field)

    coord = cx.compose_coordinates(*axes_seq)
    untagged_args = [_untag_coord_in_any_order(arg, coord) for arg in args]
    untag_module_state(module, coord, vector_axes, allow_non_vector_axes)
    result = vmapped_fn(module, *untagged_args)
    tag_module_state(module, coord, vector_axes, allow_non_vector_axes)
    if result is not None:
      result = cx.tag(result, coord)

    return result

  return wrapped_fn


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
  method_to_wrap = getattr(base_class, method_name)

  depth = getattr(module, '_wrapper_depth', 0)
  new_depth = depth + 1
  callback_module_attr = f'_callback_module_{new_depth}'
  callback_method_name_attr = f'_callback_method_name_{new_depth}'

  def __init__(self, wrapped_instance, callback_specs):  # pylint: disable=invalid-name
    self.wrapped_instance = wrapped_instance
    self._wrapper_depth = new_depth  # pylint: disable=protected-access
    formatted_specs = format_callbacks(callback_specs)
    setattr(self, callback_module_attr, formatted_specs.module)
    setattr(self, callback_method_name_attr, formatted_specs.method_name)

  def __getattr__(self, attr_name):  # pylint: disable=invalid-name
    """Delegate attribute access to the wrapped instance."""
    return getattr(self.wrapped_instance, attr_name)

  @functools.wraps(method_to_wrap)
  def wrapped_fn(self, *args, **kwargs):
    result = method_to_wrap(self.wrapped_instance, *args, **kwargs)
    # The reason this function closes over method and module attrs and uses
    # getattr to access those is to ensure that the method acts on modules that
    # are in a valid context. Closing over nnx modules and calling them is not
    # allowed in nnx (an error is raised under nnx.jit). This effectively
    # mimicks an explicit injection of a callback module that would be accessed
    # via a qualifying class attribute.
    callback_module = getattr(self, callback_module_attr)
    callback_method_name = getattr(self, callback_method_name_attr)
    callback_fn = getattr(callback_module, callback_method_name)
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
  cls = type(
      base_class.__name__ + 'WithCallbacks', (base_class,), attrs, pytree=False
  )
  return cls(module, callback_specs)


def retrieve_subclass_modules(module, subclass):
  """Returns list of all unique `subclass` instances on `module`."""
  subclass_modules = []
  for _, x in module.iter_modules():
    if isinstance(x, subclass):
      subclass_modules.append(x)
  return subclass_modules
