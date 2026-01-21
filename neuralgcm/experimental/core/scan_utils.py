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

"""Utilities for nested scan transformations used to rollout models."""

import functools
from typing import Sequence, TypeAlias

import coordax as cx
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import pytree_utils
import numpy as np


def _extract_timedelta(
    coordinate: cx.Coordinate,
) -> coordinates.TimeDelta:
  """Extracts TimeDelta axis from a coordinate or raises if none found."""
  return cx.coords.extract(coordinate, coordinates.TimeDelta)


# A pytree of `cx.Coordinate` that defines the inputs to a nested scan.
InputsSpecLike: TypeAlias = (
    dict[str, dict[str, cx.Coordinate]] | dict[str, cx.Coordinate]
)
# A pytree of `cx.Field` that provides data for a nested scan.
InputsLike: TypeAlias = dict[str, dict[str, cx.Field]] | dict[str, cx.Field]


def _drop_none_from_nested_dict(nested: InputsSpecLike) -> InputsSpecLike:
  """Filters out None values from a pytree."""
  flat, _ = pytree_utils.flatten_dict(nested)
  flat = {k: v for k, v in flat.items() if v is not None}
  return pytree_utils.unflatten_dict(flat)


def _group_by_timedeltas(
    inputs_spec: InputsSpecLike,
    dt: np.timedelta64 | None = None,
    ref_t0: np.timedelta64 | None = None,
) -> Sequence[tuple[np.timedelta64, InputsSpecLike]]:
  """Returns input specs grouped by their dt steps in the increasing order."""
  is_coord = lambda c: isinstance(c, cx.Coordinate)
  map_coords = functools.partial(jax.tree.map, is_leaf=is_coord)
  td_axes = map_coords(_extract_timedelta, inputs_spec)
  td_axes_flat = jax.tree.leaves(td_axes, is_leaf=is_coord)
  unique_timedelta_axes = set(td_axes_flat)

  def _get_step(timedelta_axis: coordinates.TimeDelta):
    steps = np.unique(np.diff(timedelta_axis.deltas))
    if steps.size == 0:
      if ref_t0 is None:
        raise ValueError(
            'Cannot infer step from TimeDelta of size 1 without reference t0.'
        )
      return timedelta_axis.deltas[0] - ref_t0
    elif steps.size == 1:
      return steps[0]
    else:
      raise ValueError(f'Non-uniform TimeDelta found: {timedelta_axis}')

  step_to_axis = {_get_step(td): td for td in unique_timedelta_axes}
  if dt is not None and dt not in step_to_axis:
    step_to_axis[dt] = None
  groups = []
  for step, axis in step_to_axis.items():
    group = map_coords(lambda x: x if axis in x.axes else None, inputs_spec)  # pylint: disable=cell-var-from-loop
    group = _drop_none_from_nested_dict(group)
    groups.append((step, group))
  return sorted(tuple(groups), key=lambda x: x[0])


def _shared_final_leadtime(inputs_spec: InputsSpecLike) -> np.timedelta64:
  """Returns the shared end time for all timedeltas in `inputs_spec`."""
  is_coord = lambda c: isinstance(c, cx.Coordinate)
  coords = jax.tree.leaves(inputs_spec, is_leaf=is_coord)
  final_deltas = [_extract_timedelta(c).deltas[-1] for c in coords]
  final_deltas = set(final_deltas)
  if len(final_deltas) == 1:
    [final_delta] = list(final_deltas)
  else:
    raise ValueError(
        f'Expected exactly one shared final delta, found:: {final_deltas}'
    )
  return final_delta


def _compute_steps_and_validate(
    by_td: Sequence[tuple[np.timedelta64, InputsSpecLike]],
    outer_delta: np.timedelta64,
    ref_t0: np.timedelta64 | None = None,
) -> tuple[int, ...]:
  """Computes scan steps and validates that timedelta axes are congruent."""
  if ref_t0 is not None:
    outer_delta = outer_delta - ref_t0
  steps = []
  for delta, _ in reversed(by_td):
    n_steps, reminder = divmod(outer_delta, delta)
    if reminder:
      raise ValueError(
          f'deltas are not congruent with: {outer_delta=} and {delta=}.'
      )
    steps.append(int(n_steps))
    outer_delta = delta
  return tuple(reversed(steps))


def nested_scan_specs(
    inputs_spec: InputsSpecLike,
    dt: np.timedelta64 | None = None,
    ref_t0: np.timedelta64 | None = None,
) -> tuple[InputsSpecLike, ...]:
  """Returns sequence of input spec for `inputs_spec` with nestable timedeltas.

  Partitions single inputs_spec with potentially varying TimeDelta axes into a
  sequence of nested spec objects ordered from most frequently appearing to
  least frequently appearing entries. This can be used to set up a nested scan
  computation with returned specs ordered from the most inner to the most outer
  scan loops. For such partition to work, all timedelta axes must be congruent,
  meaning that all timedelta axes must be uniformly spaced and have steps
  that nest inside one another and share a common final timedelta value. If `dt`
  is provided, it is treated as a smallest step in addition to steps present in
  timedelta axes.

  Args:
    inputs_spec: Specification of data with timedelta axes to process.
    dt: Optional numpy timedelta defining the inner-most scan step.
    ref_t0: Optional timedelta to use for step inference of TimeDelta of size 1.

  Returns:
    A tuple of input specs, one for each level of the nested scan, ordered from
    innermost to outermost.
  """
  dt_and_specs = _group_by_timedeltas(inputs_spec, dt, ref_t0)
  _compute_steps_and_validate(
      dt_and_specs, _shared_final_leadtime(inputs_spec), ref_t0
  )
  return tuple(spec for _, spec in dt_and_specs)


def nested_scan_steps(
    inputs_spec: InputsSpecLike,
    dt: np.timedelta64 | None = None,
    ref_t0: np.timedelta64 | None = None,
) -> tuple[int, ...]:
  """Returns nested scan lengths from innermost to outermost.

  Computes the number of steps for each level of a nested scan based on the
  `inputs_spec` and an optional finest-grained timestep `dt`. The time steps in
  `inputs_spec` must be congruent.

  Args:
    inputs_spec: Specification of data with timedelta axes to process.
    dt: Optional numpy timedelta defining the inner-most scan step.
    ref_t0: Optional timedelta to use for step inference of TimeDelta of size 1.

  Returns:
    A tuple of integers representing the number of scan steps for each level,
    from the innermost to the outermost scan.
  """
  dts_and_specs = _group_by_timedeltas(inputs_spec, dt, ref_t0)
  outer_delta = _shared_final_leadtime(inputs_spec)
  return _compute_steps_and_validate(dts_and_specs, outer_delta, ref_t0)


def nest_data_for_scans(
    inputs: InputsLike,
    dt: np.timedelta64 | None = None,
    ref_t0: np.timedelta64 | None = None,
) -> tuple[InputsLike, ...]:
  """Returns `inputs` partitioned into subsets corresponding to scan nesting."""
  in_spec = jax.tree.map(lambda x: x.coordinate, inputs, is_leaf=cx.is_field)
  scan_steps = nested_scan_steps(in_spec, dt, ref_t0)
  scan_specs = nested_scan_specs(in_spec, dt, ref_t0)
  nested_data = []
  for i, spec in enumerate(scan_specs):
    shape = scan_steps[i:][::-1]
    dummy_td = cx.coords.compose(*[cx.DummyAxis(None, s) for s in shape])

    # pylint: disable=cell-var-from-loop
    def _reshape(field: cx.Field):
      coord = field.coordinate
      td = _extract_timedelta(coord)
      out_coord = cx.coords.replace_axes(coord, td, dummy_td)
      out_axes = {d: i for i, d in enumerate(out_coord.dims) if d}
      reshape = cx.cmap(lambda x: x.reshape(shape), out_axes=out_axes)
      return reshape(field.untag(td))

    fs_for_spec = pytree_utils.replace_with_matching_or_default(
        spec, inputs, None, check_used_all_replace_keys=False
    )
    fs_for_spec = jax.tree.map(_reshape, fs_for_spec, is_leaf=cx.is_field)
    nested_data.append(fs_for_spec)
  return tuple(nested_data)


def ravel_data_from_nested_scans(
    outputs: InputsLike,
    outputs_spec: InputsSpecLike,
) -> InputsLike:
  """Returns `inputs` raveled and labeled with timedeltas in `outputs_spec`."""

  def _retag(field: cx.Field, coord):
    timedelta = cx.coords.extract(coord, coordinates.TimeDelta)
    result = cx.cmap(lambda x: x.ravel())(field).tag(timedelta)
    if result.coordinate != coord:
      raise ValueError(f'Coordinate mismatch: {result.coordinate} vs {coord}')
    return result

  return jax.tree.map(_retag, outputs, outputs_spec, is_leaf=cx.is_field)
