# Copyright 2025 Google LLC
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

"""Utility functions that operate on coordax fields and dicts of fields."""

import functools
import itertools
from typing import overload, Sequence

import coordax as cx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import typing
import numpy as np


def _validate_aligned_dims(
    aligned_dims: tuple[str, ...],
) -> None:
  """Validates that all fields in `fields` are aligned with `dims_to_align`."""
  uniques, counts = np.unique(aligned_dims, return_counts=True)
  if counts.size > 0 and counts.max() > 1:
    repeated_dims = [str(x) for x in uniques[counts > 1]]
    raise ValueError(
        f'`dims_to_align` must be unique, but got {repeated_dims=}.'
    )


def combine_fields(
    fields: dict[str, cx.Field],
    dims_to_align: Sequence[str | cx.Coordinate],
    out_axis_tag: str | cx.Coordinate | None = None,
) -> cx.Field:
  """Aligns and combines fields from dict values in inputs into a single field.

  All input fields should include all of the dimensions in `dims_to_align`,
  with at most one additional axis that will be used for concatenation.
  Dimensions to align on can be specified either via names or coordinates.

  Args:
    fields: A dictionary of fields to combine.
    dims_to_align: Dimension names or coordinates to remain aligned.
    out_axis_tag: Tag for the resulting combined axis. Default is `None`.

  Returns:
    A single field formed by combining the input fields.
  """
  aligned_dims_and_axes_tuples = [
      (c,) if isinstance(c, str) else c.axes for c in dims_to_align
  ]
  aligned_dims_and_axes = tuple(itertools.chain(*aligned_dims_and_axes_tuples))
  aligned_dims_tuples = [
      (c,) if isinstance(c, str) else c.dims for c in dims_to_align
  ]
  aligned_dims = tuple(itertools.chain(*aligned_dims_tuples))
  _validate_aligned_dims(aligned_dims)

  def _get_concat_axis(f: cx.Field) -> cx.Coordinate | None:
    """Returns Coordinate | None representing the axis to concatenate over."""
    coord = cx.get_coordinate(f)
    axes = []
    for ax in coord.axes:
      if ax in aligned_dims_and_axes or ax.dims[0] in dims_to_align:
        axes.append(ax)

    if set(aligned_dims) - set(cx.compose_coordinates(*axes).dims):
      raise ValueError(
          f'Cannot combine {f} because it does not align with {aligned_dims}'
      )
    other_axes = list(set(coord.axes) - set(axes))
    if not other_axes:
      return None  # no coordinate to untag, dummy axis will be inserted.
    elif len(other_axes) == 1:
      return other_axes[0]  # will be untagged prior to concatenation.
    raise ValueError(
        f'Field {f} has more than 1 axis other than {aligned_dims_and_axes=}.'
    )

  concat_axes = {k: _get_concat_axis(v) for k, v in fields.items()}
  expand = cx.cmap(lambda x: x[np.newaxis])
  add_positional = lambda f, c: expand(f) if c is None else f.untag(*c.dims)
  fields = [
      add_positional(f, concat_axes[k]) for k, f in sorted(fields.items())
  ]
  named_axes_set = set(tuple(sorted(f.named_axes.items())) for f in fields)
  if len(named_axes_set) == 1:
    out_axes = dict(named_axes_set.pop())
  else:
    raise ValueError(f'No unique out_axes found in inputs: {named_axes_set=}')
  return cx.cmap(jnp.concatenate, out_axes=out_axes)(fields).tag(out_axis_tag)


def split_to_fields(
    field: cx.Field,
    targets: dict[str, cx.Coordinate],
) -> dict[str, cx.Field]:
  """Splits a field to a dict of fields.

  Args:
    field: The field to split.
    targets: A dictionary of coordinates defining the split structure.

  Returns:
    A dictionary of fields formed by splitting `field` along a single axis not
    present in `targets`.

  Raises:
    ValueError: If `targets` do not defined a valid split. This can happen if:
        - `field` has more or less than one dimension not present in `targets`.
        - The size of the split-dimension on `field` does not match the combined
          size of the new axes in `targets`.
        - An entry in `targets` differs from `field` in more than two axes.
  """
  dims_in_targets = itertools.chain(*[c.dims for c in targets.values()])
  dim_to_split = set(field.dims) - set(dims_in_targets)
  if len(dim_to_split) != 1:
    raise ValueError(
        'Field must have exactly one dimension not present in coords for '
        f'splitting, but found {len(dim_to_split)}: {dim_to_split}.'
    )
  [dim_to_split] = list(dim_to_split)
  if dim_to_split:  # skip untag if dim_to_split is already positional.
    field = field.untag(dim_to_split)
  aligned_coords = cx.get_coordinate(field, missing_axes='skip')

  def _get_new_axis(c: cx.Coordinate) -> cx.Coordinate | None:
    if set(aligned_coords.axes) - set(c.axes):
      raise ValueError(
          f'Coordinate {c} does not specify a valid split element because it'
          f' is not aligned with the non-split part of input field {field=}'
      )
    new_axes = list(set(c.axes) - set(aligned_coords.axes))
    if not new_axes:
      return None  # no coordinate to tag, dummy axis will be squeezed.
    elif len(new_axes) == 1:
      return new_axes[0]  # will be tagged to the corresponding split.
    raise ValueError(
        f'Coordinate {c} has more than 1 new axis compared to input {field=}.'
    )

  new_axes = [_get_new_axis(c) for c in targets.values()]
  splits = np.cumsum([1 if c is None else c.shape[0] for c in new_axes])
  if splits[-1] != field.positional_shape[0]:
    raise ValueError(
        f'The total size of the dimensions defined in `targets` ({splits[-1]})'
        ' does not match the size of the dimension being split in the input'
        f' field ({field.positional_shape[0]}).'
    )
  split_fn = functools.partial(jnp.split, indices_or_sections=splits[:-1])
  splits = cx.cmap(split_fn, out_axes=field.named_axes)(field)
  maybe_tag = lambda x, c: cx.cmap(jnp.squeeze)(x) if c is None else x.tag(c)
  return {
      k: maybe_tag(splits[i], new_axes[i]).order_as(*c.dims)
      for i, (k, c) in enumerate(targets.items())
  }


@overload
def in_axes_for_coord(
    inputs: typing.Pytree,
    coord: cx.Coordinate,
) -> typing.Pytree:
  ...


@overload
def in_axes_for_coord(
    inputs: typing.Pytree,
    coord: Sequence[cx.Coordinate],
) -> Sequence[typing.Pytree]:
  ...


def in_axes_for_coord(
    inputs: typing.Pytree,
    coord: cx.Coordinate | Sequence[cx.Coordinate],
) -> typing.Pytree | Sequence[typing.Pytree]:
  """Returns vmap `in_axes` specs for mapping over `coord` in `inputs`.

  If multiple coordinates are provided, then in_axes are computed for a nested
  vmap calls, accounting for axes consumed by vmap on the outer coordinates.
  Any leaves in inputs that are not of Field type will be set for replication.

  Args:
    inputs: A pytree of fields.
    coord: A single coordinate or a sequence of coordinates.

  Returns:
    Per element `in_axes` specs for mapping over `coord` in `inputs`.

  Examples:
    x, y = cx.SizedAxis('x', 3), cx.SizedAxis('y', 4)
    f1 = cx.wrap(np.zeros((3, 4)), x, y)
    f2 = cx.wrap(np.zeros((4, 3)), y, x)
    f3 = cx.wrap(np.zeros((4,)), y)
    inputs = {'a': f1, 'b': (f2, 123, f3)}
    # in_axes for mapping over x:
    # in_axes_for_coord(inputs, x) returns {'a': 0, 'b': (1, None, None)}
    # in_axes for mapping over y:
    # in_axes_for_coord(inputs, y) returns {'a': 1, 'b': (0, None, 0)}
  """
  if isinstance(coord, Sequence):
    return nest_in_axes(*(in_axes_for_coord(inputs, c) for c in coord))
  if coord.ndim != 1:
    raise ValueError(f'idx can be computed only for 1d coord, got {coord}')
  dim = coord.dims[0]
  leaves, treedef = jax.tree.flatten(inputs, is_leaf=cx.is_field)
  indices = [x.named_axes.get(dim) if cx.is_field(x) else None for x in leaves]
  return jax.tree.unflatten(treedef, indices)


def nest_in_axes(*in_axes_to_nest: typing.Pytree) -> tuple[typing.Pytree, ...]:
  """Computes adjusted `in_axes_to_nest` for a series of nested `vmap` calls."""
  if not in_axes_to_nest:
    return ()

  def _validate_in_axes_for_leaf(*original_axes: Sequence[int | None]):
    non_none_axes = [ax for ax in original_axes if ax is not None]
    if any(ax < 0 for ax in non_none_axes):
      raise ValueError(f'Negative axes are not allowed. Got: {original_axes}')
    if len(set(non_none_axes)) != len(non_none_axes):
      raise ValueError(
          'leaf in *in_axes is mapped over the same axis multiple times. '
          f'Got: {original_axes} '
      )
  is_none = lambda x: x is None
  jax.tree.map(_validate_in_axes_for_leaf, *in_axes_to_nest, is_leaf=is_none)

  def _nest_once(current: typing.Pytree, outer: typing.Pytree) -> typing.Pytree:
    # shifts if outer index is smaller than current, resulting in a shift.
    shift_idx = lambda i, o: i if i is None or o is None or i < o else i - 1
    return jax.tree.map(shift_idx, current, outer, is_leaf=is_none)

  nested_in_axes = [in_axes_to_nest[0]]  # no shifts in outermost vmap.
  for i in range(1, len(in_axes_to_nest)):
    adjusted = functools.reduce(  # shift by outer in_axes in reverse order.
        _nest_once, reversed(in_axes_to_nest[:i]), in_axes_to_nest[i]
    )
    nested_in_axes.append(adjusted)
  return tuple(nested_in_axes)


def shape_struct_fields_from_coords(
    coords: dict[str, cx.Coordinate],
) -> dict[str, cx.Field]:
  """Returns Fields constructed from `coords` with ShapeDtypeStruct data."""
  make_fn = lambda d: {k: cx.wrap(jnp.zeros(c.shape), c) for k, c in d.items()}
  return jax.eval_shape(make_fn, coords)
