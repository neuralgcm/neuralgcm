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
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import nnx_compat
import numpy as np


@nnx_compat.dataclass
class FieldCombiner:
  """Aligns and combines fields from dict values in inputs into a single field.

  All input fields should include all of the dimensions in `dims_to_align`,
  with at most one additional axis that will be used for concatenation.
  Dimensions to align on can be specified either via names or coordinates.

  Attributes:
    dims_to_align: Dimension names or coordinates to remain aligned.
    order_method: Defines how the order of output dimensions is determined.
      The following options are allowed:
        - 'as_inputs': uses axis order from inputs if unique, otherwise raises.
        - 'as_dims': uses `dims_to_align` with a leading positional axis.
        - 'try_as_inputs': same as 'as_inputs', but falls back to `as_dims`.
  """

  dims_to_align: Sequence[str | cx.Coordinate]
  order_method: Literal['as_dims', 'as_inputs', 'try_as_inputs'] = 'as_inputs'

  def __post_init__(self):
    uniques, counts = np.unique(self.aligned_dims, return_counts=True)
    if counts.max() > 1:
      repeated_dims = [str(x) for x in uniques[counts > 1]]
      raise ValueError(
          f'`dims_to_align` must be unique, but got {repeated_dims=}.'
      )

  @property
  def aligned_dims(self) -> tuple[str, ...]:
    """Aligned dimension names."""
    dims = []
    for x in self.dims_to_align:
      if isinstance(x, str):
        dims.append(x)
      else:
        dims.extend(x.dims)
    return tuple(dims)

  def _infer_out_axes(self, xs: Sequence[cx.Field]) -> dict[str, int]:
    """Returns indices for named dimensions based on xs and order_method."""
    out_axes_as_dims = {d: i + 1 for i, d in enumerate(self.aligned_dims)}
    named_axes_in_xs = set([tuple(sorted(f.named_axes.items())) for f in xs])
    if len(named_axes_in_xs) == 1:
      out_axes_as_inputs = dict(list(named_axes_in_xs)[0])
    else:
      out_axes_as_inputs = None  # no unique named_axes found in inputs.
    if self.order_method == 'as_inputs' and out_axes_as_inputs is None:
      raise ValueError(
          f'No unique out_axes found in inputs: {named_axes_in_xs=}'
      )
    if self.order_method == 'as_dims':
      return out_axes_as_dims
    elif self.order_method in ['try_as_inputs', 'as_inputs']:
      return out_axes_as_inputs or out_axes_as_dims
    else:
      raise ValueError(f'Unknown order_method: {self.order_method}')

  def __call__(self, fields: dict[str, cx.Field]) -> cx.Field:
    """Combines fields from dictionary values into a single field."""
    dims_and_axes = tuple(
        itertools.chain.from_iterable(
            [c] if isinstance(c, str) else c.axes for c in self.dims_to_align
        )
    )

    def _get_concat_axis(f: cx.Field) -> cx.Coordinate | None:
      """Returns Coordinate | None representing the axis to concatenate over."""
      coord = cx.get_coordinate(f)
      axes = []
      for ax in coord.axes:
        if ax in dims_and_axes or ax.dims[0] in self.dims_to_align:
          axes.append(ax)

      if set(self.aligned_dims) - set(cx.compose_coordinates(*axes).dims):
        raise ValueError(
            f'Cannot combine {f} because it does not align with {dims_and_axes}'
        )
      other_axes = list(set(coord.axes) - set(axes))
      if not other_axes:
        return None  # no coordinate to untag, dummy axis will be inserted.
      elif len(other_axes) == 1:
        return other_axes[0]  # will be untagged prior to concatenation.
      raise ValueError(
          f'Field {f} has more than 1 axis other than {dims_and_axes=}.'
      )

    concat_axes = {k: _get_concat_axis(v) for k, v in fields.items()}
    expand = cx.cmap(lambda x: x[np.newaxis])
    add_positional = lambda f, c: expand(f) if c is None else f.untag(*c.dims)
    fields = [add_positional(f, concat_axes[k]) for k, f in fields.items()]
    out_axes = self._infer_out_axes(fields)
    return cx.cmap(jnp.concatenate, out_axes=out_axes)(fields)


def split_to_fields(
    f: cx.Field,
    coords: dict[str, cx.Coordinate],
) -> dict[str, cx.Field]:
  """Splits a field to a dict of fields.

  Args:
    f: The field to split.
    coords: A dictionary of coordinates defining the split structure.

  Returns:
    A dictionary of fields formed by splitting `f` along a single axis not
    present in `coords`.

  Raises:
    ValueError: If `coords` do not defined a valid split. This can happen if:
        - `f` has more or less than one dimension not present in `coords`.
        - The size of the split-dimension on `f` does not match the combined
          size of the new axes in `coords`.
        - An entry in `coords` differs from `f` in more than two axes.
  """
  dims_in_coords = itertools.chain(*[c.dims for c in coords.values()])
  dim_to_split = set(f.dims) - set(dims_in_coords)
  if len(dim_to_split) != 1:
    raise ValueError(
        'Field must have exactly one dimension not present in coords for '
        f'splitting, but found {len(dim_to_split)}: {dim_to_split}.'
    )
  [dim_to_split] = list(dim_to_split)
  if dim_to_split:  # skip untag if dim_to_split is already positional.
    f = f.untag(dim_to_split)
  aligned_coords = cx.get_coordinate(f, missing_axes='skip')

  def _get_new_axis(c: cx.Coordinate) -> cx.Coordinate | None:
    if set(aligned_coords.axes) - set(c.axes):
      raise ValueError(
          f'Coordinate {c} does not specify a valid split element because it'
          f' is not aligned with the non-split part of input field {f=}'
      )
    new_axes = list(set(c.axes) - set(aligned_coords.axes))
    if not new_axes:
      return None  # no coordinate to tag, dummy axis will be squeezed.
    elif len(new_axes) == 1:
      return new_axes[0]  # will be tagged to the corresponding split.
    raise ValueError(
        f'Coordinate {c} has more than 1 new axis compared to input field {f}.'
    )

  new_axes = [_get_new_axis(c) for c in coords.values()]
  splits = np.cumsum([1 if c is None else c.shape[0] for c in new_axes])
  if splits[-1] != f.positional_shape[0]:
    raise ValueError(
        f'The total size of the dimensions defined in `coords` ({splits[-1]})'
        ' does not match the size of the dimension being split in the input'
        f' field ({f.positional_shape[0]}).'
    )
  split_fn = functools.partial(jnp.split, indices_or_sections=splits[:-1])
  splits = cx.cmap(split_fn, out_axes=f.named_axes)(f)
  maybe_tag = lambda x, c: cx.cmap(jnp.squeeze)(x) if c is None else x.tag(c)
  return {
      k: maybe_tag(splits[i], new_axes[i]).order_as(*c.dims)
      for i, (k, c) in enumerate(coords.items())
  }


def shape_struct_fields_from_coords(
    coords: dict[str, cx.Coordinate],
) -> dict[str, cx.Field]:
  """Returns Fields constructed from `coords` with ShapeDtypeStruct data."""
  make_fn = lambda d: {k: cx.wrap(jnp.zeros(c.shape), c) for k, c in d.items()}
  return jax.eval_shape(make_fn, coords)
