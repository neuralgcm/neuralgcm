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
# pylint: disable=line-too-long
r"""Defines data loading for rollout experiments."""

import collections
from collections.abc import Callable, Iterator, Sequence
import dataclasses
import functools
import logging
import math
from typing import Any, cast

import coordax as cx
import grain.python as grain
import jax
from jax.experimental import colocated_python
import jax.numpy as jnp
import jax.sharding
from neuralgcm.experimental import xreader
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import data_specs
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import xarray_utils
from neuralgcm.experimental.training import train_utils
import numpy as np
import pandas as pd
import xarray


PyTree = typing.Pytree


class DeviceKeyedDict(dict):
  """A dictionary with jax.Device keys that supports jax.tree operations."""


def _device_keyed_mapping_flatten(d: DeviceKeyedDict):
  if not d:
    return tuple(), []
  sorted_keys = sorted(d.keys(), key=lambda device: device.id)
  children = tuple(d[k] for k in sorted_keys)
  aux_data = sorted_keys
  return children, aux_data


def _device_keyed_mapping_unflatten(aux_data, children) -> DeviceKeyedDict:
  return DeviceKeyedDict(zip(aux_data, children))


jax.tree_util.register_pytree_node(
    DeviceKeyedDict,
    _device_keyed_mapping_flatten,
    _device_keyed_mapping_unflatten,
)


def _extract_timedelta(coord: cx.Coordinate) -> coordinates.TimeDelta:
  """Extracts a TimeDelta coordinate from a coordinate."""
  timedelta = cx.coords.extract(coord, coordinates.TimeDelta)
  return cast(coordinates.TimeDelta, timedelta)


def _transpose_for_to_global_array(
    data: dict[Any, PyTree], is_leaf: Callable[[Any], bool]
) -> PyTree:
  """Transpose a dict of pytrees into a pytree of dicts."""
  treedefs = {k: jax.tree.structure(v, is_leaf) for k, v in data.items()}
  treedefs_set = set(treedefs.values())
  if len(treedefs_set) != 1:
    raise ValueError(f'non-unique treedef: {treedefs_set}')
  [treedef] = treedefs_set

  zipped = zip(*[jax.tree.leaves(value, is_leaf) for value in data.values()])
  transposed_leaves = [
      DeviceKeyedDict(zip(data.keys(), values)) for values in zipped
  ]
  transposed = jax.tree.unflatten(treedef, transposed_leaves)
  return transposed


def _get_datetime_forecast_starts(
    sample_count: int,
    candidates: pd.DatetimeIndex,
    add_half_day_shifts: bool = True,
) -> np.ndarray:
  """Get equispaced forecast start times for evaluating against ERA5."""

  # To match ECMWF, all forecasts should be initialized at 0z or 12z. To
  # approximate this, we round the candidates to whole days, shift every other
  # one by 12 hours, and then find the closest times in the original candidate
  # list.  This simple heuristic works well if there are a reasonable number
  # more candidates than samples, which should always be true for practical
  # uses.
  if sample_count > len(candidates) / 5:
    logging.warning(
        'Too few candidate times to guarantee good mix of samples.  This is'
        ' fine for tests, but probably not for real use cases.  sample_count=%d'
        '  len(candidates)=%d',
        sample_count,
        len(candidates),
    )

  candidates = pd.to_datetime(candidates)
  rounded_candidates = candidates.ceil('1D').values

  # pick samples evenly spaced from the candidates
  if sample_count > len(rounded_candidates):
    raise ValueError(
        f'Too few candidate times to get {sample_count=}. '
        f'Only {len(rounded_candidates)=} were available.'
    )
  stride = len(rounded_candidates) // sample_count
  idx = np.round(np.arange(sample_count) * stride).astype(int)
  ideal_start_times = rounded_candidates[idx]

  if add_half_day_shifts:
    # increment every other one by 12h
    parity = np.arange(sample_count) % 2
    ideal_start_times = ideal_start_times + parity * pd.Timedelta('12h')

  indices = candidates.get_indexer(
      ideal_start_times, method='nearest', tolerance='24h'
  )
  if (indices < 0).any():
    raise ValueError(
        'no matching times found for some start times: '
        f'{ideal_start_times[indices < 0]}'
    )
  starts = candidates[indices]

  # In (presumably rare) cases where sample_count is close to len(candidates) or
  # there are many items in candidates from the same day we might have
  # duplicates at this point.
  starts = np.unique(starts)

  return starts


def _get_shared_time_axis(
    all_data: dict[str, xarray.Dataset],
) -> pd.DatetimeIndex:
  """Returns the shared time axis across all datasets."""
  time_axes = [data.indexes['time'] for data in all_data.values()]  # pytype: disable=attribute-error  # jax-api-types
  time_axes = cast(list[pd.DatetimeIndex], time_axes)
  time = time_axes[0]
  for time_axis in time_axes[1:]:
    time = time.intersection(time_axis, sort=True)
  return time


def _get_train_sample_origins(
    all_data: dict[str, xarray.Dataset],
    stencil: xreader.TimeStencil | dict[str, xreader.TimeStencil],
    dataset_time_slice: tuple[str, str] | list[tuple[str, str]] | None,
    time_sample_offset: np.timedelta64,
) -> np.ndarray:
  return _get_sample_origins(
      time_axis=_get_shared_time_axis(all_data),
      time_slices=dataset_time_slice,
      stencil=stencil,
      time_sample_offset=time_sample_offset,
  )


def _get_eval_sample_origins(
    all_data: dict[str, xarray.Dataset],
    time_slices: tuple[str, str] | list[tuple[str, str]] | None,
    stencil: xreader.TimeStencil | dict[str, xreader.TimeStencil],
    batch_count: int,
    global_batch_size: int,
    time_sample_offset: np.timedelta64,
) -> np.ndarray:
  """Get sample origins for evaluation."""
  sample_count = global_batch_size * batch_count
  time = _get_shared_time_axis(all_data)
  sample_origins = _get_sample_origins(
      time_axis=time,
      time_slices=time_slices,
      stencil=stencil,
      time_sample_offset=time_sample_offset,
  )
  # If 12hr are not divisible by the valid time sample offset, skip 12hr shifts,
  # as they would lead to invalid sample origins.
  _, reminder = divmod(pd.Timedelta('12h'), time_sample_offset)
  if reminder:
    add_half_day_shifts = False
  else:
    add_half_day_shifts = True
  return _get_datetime_forecast_starts(
      sample_count, sample_origins, add_half_day_shifts
  )


def _make_slice(x: None | slice | Sequence[str]) -> slice:
  """Converts `x` specifying a time slice to a slice object."""
  if x is None:
    return slice(None)  # the whole thing
  if isinstance(x, slice):
    return x
  else:
    return slice(*x)


def _get_sample_origins(
    time_axis: pd.Series,
    time_slices: tuple[str, str] | list[tuple[str, str]] | None,
    stencil: xreader.TimeStencil | dict[str, xreader.TimeStencil],
    time_sample_offset: np.timedelta64,
) -> np.ndarray:
  """Construct an array of origin times for train/eval examples.

  Since different models have varying needs when creating training and eval
  examples, there is a lot of flexibility in how these examples are defined.
  This routine pulls out just the origin times of the examples, while conforming
  to structural requirements defined by time_slices and stencil.  This allows
  later routines to use the origin times to construct the actual examples.

  Args:
    time_axis: The times from which examples are taken.  Typically this will be
      the fully-dense series of all times in the source dataset.
    time_slices: A definition of the regions from time_axis to use when
      constructing examples.  If None, use all of time_axis.
    stencil: The stencil which defines the shape of the training examples, as
      offsets from the origin time.  This determines how much buffer space is
      needed around the origin time.
    time_sample_offset: Time between consecutive sample origins. When multiple
      slices are used, the stride is applied separately within each slice.

  Returns:
    An array of times to use as the origins of examples.
  """

  # Convert to a list of slice objects.
  if time_slices is None:
    time_slices = [time_slices]
  elif all(isinstance(x, str) for x in time_slices) and len(time_slices) == 2:
    time_slices = [time_slices]
  else:
    if not all(isinstance(x, Sequence) and len(x) == 2 for x in time_slices):
      raise ValueError(f'Unsupported time_slices: {time_slices=}')

  time_slices = [_make_slice(x) for x in time_slices]
  time_axis_step = np.unique(np.diff(time_axis.values))
  if time_axis_step.size != 1:
    raise ValueError(
        'Time axis must have a single step, got'
        f' {time_axis_step=}.  time_axis={time_axis.values=}'
    )
  stride_between_windows, reminder = divmod(time_sample_offset, time_axis_step)
  if reminder:
    raise ValueError(
        'Time axis step must be a multiple of time_sample_offset, got'
        f' {time_axis_step=} and {time_sample_offset=}.'
    )
  stride_between_windows = stride_between_windows.item()  # get int value.
  # Collect origin times from each slice.
  sample_origins = []
  for ts in time_slices:
    slice_times = pd.Series(time_axis, index=time_axis).loc[ts].index
    candidates = slice_times[::stride_between_windows]

    if isinstance(stencil, xreader.TimeStencil):
      stencils = [stencil]
    else:
      stencils = list(stencil.values())

    valid_mask = np.ones(len(candidates), dtype=bool)
    for s in stencils:
      valid_origins_for_s = xreader.valid_origin_points(slice_times, s)
      valid_mask &= np.isin(candidates, valid_origins_for_s)
    sample_origins.append(candidates[valid_mask])

  return np.concatenate(sample_origins)


def select_targets(
    inputs: typing.InputFields,
    queries_spec: data_specs.QueriesSpec | None,
) -> typing.InputFields:
  """Selects targets matching queries from inputs."""
  # TODO(dkochkov) This is a temporary simpler solution for the case where no
  # queries include dynamic fields. Eventually those should be excluded here and
  # from the outputs of obseravation operators.
  if queries_spec is None:
    return inputs  # by default we assume that all fields are targets.
  if not isinstance(queries_spec, dict):
    raise ValueError('queries must be a dictionary.')
  targets = {}
  for source, specs in queries_spec.items():
    if specs:
      targets[source] = {}
    for var_name in specs:
      targets[source][var_name] = inputs[source][var_name]
  return targets


def sel_timedelta_fields(
    inputs: PyTree,
    values: np.timedelta64 | slice,
) -> PyTree:
  """Returns a selection of inputs based on timedelta values."""

  def _slice_field(x):
    in_timedelta_axis = x.axes['timedelta']
    if isinstance(in_timedelta_axis, parallelism.CoordinateShard):
      wrap_as_shard = True
      td_axis = in_timedelta_axis.coordinate
    elif isinstance(in_timedelta_axis, coordinates.TimeDelta):
      wrap_as_shard = False
      td_axis = in_timedelta_axis
    else:
      raise ValueError(f'Unexpected type {type(in_timedelta_axis)}')

    deltas = td_axis.deltas

    if isinstance(values, slice):
      if values.step is not None:
        raise ValueError('Stepping is not supported in timedelta selection.')

      start_val = values.start
      stop_val = values.stop

      if start_val is None:
        start_idx = 0
      else:
        start_idx = np.searchsorted(deltas, start_val, side='left')

      if stop_val is None:
        stop_idx = len(deltas)
      else:
        stop_idx = np.searchsorted(deltas, stop_val, side='right')

      get_slice = lambda x: x[start_idx:stop_idx]
      out_td = td_axis[start_idx:stop_idx]

    else:
      # Scalar value
      indices = np.nonzero(deltas == values)[0]
      if indices.size == 0:
        raise KeyError(f'Value {values} not found in timedelta axis.')
      # We take the first match, though typically they are unique.
      idx = indices[0]
      # Using slice [idx:idx+1] to preserve dimension.
      get_slice = lambda x: x[idx : idx + 1]
      out_td = td_axis[idx : idx + 1]

    if wrap_as_shard:
      out_td = parallelism.CoordinateShard(
          out_td,
          in_timedelta_axis.spmd_mesh_shape,
          in_timedelta_axis.dimension_partitions,
      )
    return cx.cpmap(get_slice)(x.untag('timedelta')).tag(out_td)

  return jax.tree.map(_slice_field, inputs, is_leaf=cx.is_field)


def sel_init_fields(inputs: PyTree) -> PyTree:
  """Returns a slice of inputs with timedelta values <= 0."""
  return sel_timedelta_fields(inputs, slice(None, np.timedelta64(0, 's')))


def sel_target_fields(inputs: PyTree) -> PyTree:
  return sel_timedelta_fields(inputs, slice(np.timedelta64(1, 's'), None))


def sel_timedelta_coords(
    coords: PyTree,
    values: np.timedelta64 | slice,
) -> PyTree:
  """Returns `coords` with TimeDelta coordinate sliced to side of `ref`."""
  if_coord = lambda c: isinstance(c, cx.Coordinate)
  fields = jax.tree.map(cx.shape_struct_field, coords, is_leaf=if_coord)
  fn = functools.partial(sel_timedelta_fields, values=values)
  out = jax.eval_shape(fn, fields)
  return jax.tree.map(lambda f: f.coordinate, out, is_leaf=cx.is_field)


def sel_target_timedeltas(inputs: PyTree) -> PyTree:
  return sel_timedelta_coords(inputs, slice(np.timedelta64(1, 's'), None))


def filter_missing_optional(
    specs: dict[
        str, dict[str, cx.Coordinate | data_specs.OptionalSpec[cx.Coordinate]]
    ],
    all_data: dict[str, xarray.Dataset],
) -> dict[str, dict[str, cx.Coordinate]]:
  """Returns `specs` with missing optional entries removed."""
  validated = {}
  for dataset_key, dataset_spec in specs.items():
    if dataset_key not in all_data:
      if not all([
          isinstance(x, data_specs.OptionalSpec) for x in dataset_spec.values()
      ]):
        raise ValueError(
            f'Dataset {dataset_key} not found in all_data and not all values in'
            ' dataset_spec are optional.'
        )
      else:
        continue
    ds = all_data[dataset_key]
    validated[dataset_key] = {}
    for var_name, spec in dataset_spec.items():
      coord, is_optional = data_specs.unwrap_optional(spec)
      if var_name not in ds:
        if is_optional:
          continue
        else:
          raise ValueError(
              f'Variable {var_name} missing in dataset {dataset_key}'
          )
      validated[dataset_key][var_name] = coord
  return validated


def _stencil_from_timedelta(
    timedelta: coordinates.TimeDelta,
) -> xreader.TimeStencil:
  """Converts a TimeDelta coordinate to a TimeStencil."""
  deltas = timedelta.deltas
  if deltas.size < 2:
    raise ValueError(
        f'TimeDelta must be of size >= 2 to infer a TimeStencil, got {deltas=}'
    )

  start = deltas[0]
  steps = np.diff(deltas)
  step = steps[0]
  if not np.all(steps == step):
    raise ValueError(
        'TimeDelta must be uniformly spaced to convert to TimeStencil.'
    )
  stop = deltas[-1]
  closed = 'both'
  return xreader.TimeStencil(start=start, stop=stop, step=step, closed=closed)


def infer_stencils(
    data_spec: dict[str, dict[str, cx.Coordinate]],
) -> dict[str, xreader.TimeStencil]:
  """Inferrs TimeStencils from `data_spec`."""
  stencils = {}

  for k, specs in data_spec.items():
    data_stencils = set()
    for coord in specs.values():
      data_stencils.add(_stencil_from_timedelta(_extract_timedelta(coord)))
    if len(data_stencils) != 1:
      raise ValueError(
          f'Expected exactly 1 unique stencil per dataset, got {data_stencils=}'
          f' for dataset key {k}'
      )
    stencil = data_stencils.pop()
    stencils[k] = stencil
  return stencils


def slice_leading_timedelta(inputs: PyTree, length: int) -> PyTree:
  """Returns a leading slice of `inputs` along timedelta axis of `length`."""

  def _slice_field(x):
    in_timedelta_axis = x.axes['timedelta']
    if isinstance(in_timedelta_axis, parallelism.CoordinateShard):
      out_td_ax = parallelism.CoordinateShard(
          in_timedelta_axis.coordinate[:length],
          in_timedelta_axis.spmd_mesh_shape,
          in_timedelta_axis.dimension_partitions,
      )
    elif isinstance(in_timedelta_axis, coordinates.TimeDelta):
      out_td_ax = in_timedelta_axis[:length]
    else:
      raise ValueError(f'Unexpected type {type(in_timedelta_axis)}')
    return cx.cpmap(lambda y: y[:length])(x.untag('timedelta')).tag(out_td_ax)

  return jax.tree.map(_slice_field, inputs, is_leaf=cx.is_field)


class HostDataBuffer:
  """Helper class to buffer data on host and stream slices to devices."""

  def __init__(
      self,
      data_slice_struct: PyTree,
      local_cpu: jax.Device | None,
      desired_spatial_dims_layout: tuple[str, ...] = (),
  ):
    self.data_slice_struct = data_slice_struct
    self.current_batch = None
    self.desired_dims_layout = ('timedelta',) + desired_spatial_dims_layout
    if local_cpu is not None:
      self.pinned_sharding = jax.sharding.SingleDeviceSharding(
          local_cpu, memory_kind='pinned_host'
      )
    else:
      self.pinned_sharding = None

  def set_data(self, batch):
    """Stores `batch` in a buffer in pre-processed contiguous format."""

    def _optimize_field_for_slice(f):
      new_order = []
      for dim in self.desired_dims_layout:
        if dim in f.dims:
          new_order.append(dim)
      f = f.order_as(*new_order, ...)
      f_data = np.ascontiguousarray(f.data.astype(np.float32))
      if self.pinned_sharding is not None:
        # attempt to move the data to the pinned host memory.
        # TODO(dkochkov): investigate why this does not lead to a speedup.
        f_data = jax.device_put(f_data, self.pinned_sharding)
      # values stored on the buffer are read via pure_callback which should be
      # of numpy array type, ideally as ready as possible for H2D transfer.
      # Steps above attempt to accomplish that by formatting the data to the
      # desired axies order, dtype and memory layout.
      f = cx.Field(np.asarray(f_data), f.dims, f.axes)
      f = f.untag('timedelta')
      return f

    to_set = select_targets(batch, self.data_slice_struct)
    self.current_batch = jax.tree.map(
        _optimize_field_for_slice, to_set, is_leaf=cx.is_field
    )

  def get_slice(self, idx):
    if self.current_batch is None:
      raise ValueError('Trying to access slice before data was set.')
    return jax.tree.map(lambda x: x[idx], self.current_batch)


@dataclasses.dataclass
class DataLoader:
  """Handles data loading and creation of input pipelines.

  Attributes:
    all_data: Dictionary of xarray datasets containing all available data.
    parallelism_mesh: Parallelism mesh describing the device mesh and sharding
      schemas. If specified, the data will be loaded in a sharded manner and
      converted to global jax arrays as a part of the input pipeline. To support
      batching, a sharding rule for 'batch' dimension name should be present in
      `parallelism_mesh.field_partitions[loading_partition_schema]`.
    loading_partition_schema: The partition schema to use when loading data.
    shardable_dims: The dimensions to consider for loading in shards.
  """

  all_data: dict[str, xarray.Dataset]
  parallelism_mesh: parallelism.Mesh | None
  loading_partition_schema: str | None = None
  shardable_dims: tuple[str, ...] = ()

  def __post_init__(self):
    if self.parallelism_mesh and self.loading_partition_schema is None:
      raise ValueError(
          'DataLoader requires `loading_partition_schema` to be specified if'
          ' `parallelism_mesh` is provided.'
      )

  @property
  def spmd_mesh(self) -> jax.sharding.Mesh | None:
    """JAX sharding mesh."""
    return self.parallelism_mesh.spmd_mesh if self.parallelism_mesh else None

  def make_batch_axis(
      self, batch_size_per_device: int | None
  ) -> cx.Coordinate | None:
    """Computes batch axis."""
    if batch_size_per_device is None:
      return None
    spmd_mesh = self.spmd_mesh
    if spmd_mesh is None:
      return cx.SizedAxis('batch', batch_size_per_device)
    deg_of_model_parallelism = math.prod(
        v for k, v in spmd_mesh.shape.items() if k != 'batch'
    )
    global_batch_size = (
        jax.device_count() // deg_of_model_parallelism * batch_size_per_device
    )
    batch_axis = cx.SizedAxis('batch', global_batch_size)
    return batch_axis

  def coord_to_shard(self, coord: cx.Coordinate) -> cx.Coordinate:
    if self.spmd_mesh is None:
      return coord
    mesh = self.parallelism_mesh
    assert isinstance(mesh, parallelism.Mesh)  # guaranteed by spmd_mesh check.
    return cx.coords.compose(*[
        parallelism.CoordinateShard(
            ax,
            mesh.shape,
            mesh.field_partitions[self.loading_partition_schema],
        )
        for ax in coord.axes
    ])

  def _prep_data_stencils_and_shard_count(
      self,
      input_data_specs: typing.Pytree,
      dynamic_input_specs: typing.Pytree,
      batch_size_per_device: int | None,
  ) -> tuple[dict[str, xarray.Dataset], dict[str, xreader.TimeStencil], int]:
    """Prepares data, stencils and shard count for train/eval data loading."""
    get_var_names = lambda k: (
        list(input_data_specs.get(k, [])) + list(dynamic_input_specs.get(k, []))
    )
    all_data = {k: v[get_var_names(k)] for k, v in self.all_data.items()}
    # Remove datasets from which we are not loading any variables.
    all_data = {k: v for k, v in all_data.items() if v.data_vars}

    inputs_stencils = infer_stencils(input_data_specs)
    dynamic_stencils = infer_stencils(dynamic_input_specs)
    stencils = inputs_stencils | dynamic_stencils
    shared_keys = set(inputs_stencils) & set(dynamic_stencils)
    for k in shared_keys:
      if inputs_stencils[k] != dynamic_stencils[k]:
        raise ValueError(
            f'Stencil for {k} differs between inputs and dynamic inputs:'
            f' {inputs_stencils[k]} vs {dynamic_stencils[k]}'
        )

    if batch_size_per_device is None:
      shard_count = 1
    elif self.spmd_mesh is None:
      shard_count = 1
    else:
      deg_of_model_parallelism = math.prod(
          v for k, v in self.spmd_mesh.shape.items() if k != 'batch'
      )
      shard_count = jax.device_count() // deg_of_model_parallelism

    return all_data, stencils, shard_count

  def setup_targets_via_callback(
      self,
      data_slice_struct: PyTree,
      callback_pinned_host: bool = False,
      callback_spatial_dims_layout: tuple[str, ...] = (),
      idx_step: int = 1,
  ) -> tuple[Callable[[int], PyTree], HostDataBuffer]:
    """Sets up data retrieval functions for use in jitted train/eval steps.

    This method sets up the mechanism to load data via a callback. It returns
    a `retrieve_fn` that can be used inside JIT-compiled code to fetch data
    for a given index, and a `data_buffer` that must be updated with the
    corresponding data (e.g. by the iterator returned from `build_train_inputs`).

    Args:
      data_slice_struct: Structure of the data slice to be retrieved.
      callback_pinned_host: Whether to use pinned host memory for the callback
        buffer. This experimental feature should, in principle, lead to faster
        H2D transfers (especially on GPU).
      callback_spatial_dims_layout: The desired spatial dimensions layout for
        the callback buffer.
      idx_step: The step in `idx` passed to `retrieve_fn` that corresponds to
        pulling the next slice of data. `idx` must be a multiple of `idx_step`.

    Returns:
      A tuple containing:
        - retrieve_fn: A function that takes an index and returns the data slice.
        - data_buffer: A HostDataBuffer to store the loaded data.
    """
    spmd_mesh = self.spmd_mesh
    if spmd_mesh is None:
      raise ValueError(
          'setup_targets_via_callback called requires specification of the'
          f' target device mesh, but None is set: {self.parallelism_mesh=}.'
      )
    assert spmd_mesh is not None
    mesh = self.parallelism_mesh
    assert isinstance(mesh, parallelism.Mesh)  # guaranteed by spmd_mesh check.
    device_by_id = {d.id: d for d in jax.devices()}

    devices = jax.local_devices()
    local_cpu = colocated_python.colocated_cpu_devices(devices)[0]

    if callback_pinned_host:
      data_buffer = HostDataBuffer(
          data_slice_struct, local_cpu, tuple(callback_spatial_dims_layout)
      )
    else:
      data_buffer = HostDataBuffer(
          data_slice_struct, None, tuple(callback_spatial_dims_layout)
      )

    # Remove values that won't be needed for the loss to minimize data transfer.
    def host_slice_fn(idx, device_id):
      """Gets a slice from the buffer and selects targets."""
      buffer_idx, reminder = divmod(idx, idx_step)
      if reminder != 0:
        raise ValueError(f'Index {idx} must be a multiple of {idx_step}.')
      device = device_by_id[device_id.item()]
      sliced_data = data_buffer.get_slice(buffer_idx)

      def _is_leaf(x):
        return isinstance(x, dict) and any(isinstance(k, jax.Device) for k in x)

      result = jax.tree.map(lambda x: x[device], sliced_data, is_leaf=_is_leaf)
      return result

    shard_slice_struct = jax.tree.map(
        lambda f: cx.shape_struct_field(self.coord_to_shard(f.coordinate)),
        data_slice_struct,
        is_leaf=cx.is_field,
    )

    def _add_input_shard_optimizations(
        data_shard_struct: PyTree,
    ) -> PyTree:
      """Adds memory layout optimizations to input data shard struct."""

      def _with_pinned_host_mem(shape_dtype) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            shape=shape_dtype.shape,
            dtype=shape_dtype.dtype,
            sharding=jax.sharding.SingleDeviceSharding(
                device=local_cpu, memory_kind='pinned_host'
            ),
        )

      if callback_pinned_host:
        data_shard_struct = jax.tree.map(
            _with_pinned_host_mem, data_shard_struct
        )
      if callback_spatial_dims_layout:
        layout_dims = callback_spatial_dims_layout
        order_as_fn = lambda f: f.order_as(*layout_dims, ...)
        order_as_for_struct = lambda fs: jax.eval_shape(order_as_fn, fs)
        data_shard_struct = jax.tree.map(
            order_as_for_struct, data_shard_struct, is_leaf=cx.is_field
        )
      return data_shard_struct

    def _retrieve_local_shard(idx, device_id):
      """Retrieves a local arrays via callback for each host."""
      shards = jax.pure_callback(
          host_slice_fn,
          _add_input_shard_optimizations(shard_slice_struct),
          idx,
          device_id,
      )
      shards = jax.tree.map(
          lambda f, t: f.order_as(*t.dims),
          shards,
          shard_slice_struct,
          is_leaf=cx.is_field,
      )
      return jax.tree.map(lambda x: x.data, shards, is_leaf=cx.is_field)

    def _retrieve_global_targets(idx):
      """Retrieves a global concrete array by stitching local shards."""
      get_out_spec = lambda f: parallelism.get_partition_spec(
          f.dims,
          mesh.field_partitions[self.loading_partition_schema],
      )
      out_specs = jax.tree.map(
          get_out_spec, data_slice_struct, is_leaf=cx.is_field
      )
      spmd_devices = spmd_mesh.devices
      device_indices = jnp.array([d.id for d in spmd_devices.ravel()]).reshape(
          spmd_devices.shape
      )
      merged_shards = jax.shard_map(
          _retrieve_local_shard,
          mesh=spmd_mesh,
          in_specs=(
              jax.sharding.PartitionSpec(),
              jax.sharding.PartitionSpec(*spmd_mesh.axis_names),
          ),
          out_specs=out_specs,
      )(idx, device_indices)
      coords = jax.tree.map(
          lambda f: parallelism.get_unsharded(f.coordinate),
          data_slice_struct,
          is_leaf=cx.is_field,
      )
      out = jax.tree.map(
          cx.wrap,
          merged_shards,
          coords,
      )
      return mesh.with_sharding_constraint(out, self.loading_partition_schema)

    return _retrieve_global_targets, data_buffer

  def data_slice_struct(
      self,
      input_data_specs: typing.Pytree,
      batch_size_per_device: int | None,
  ):
    """Returns shape struct of a time slice of input data."""
    batch_axis = self.make_batch_axis(batch_size_per_device)
    is_coord = lambda c: isinstance(c, cx.Coordinate)

    def infer_data_slice_struct(c):
      axes = [ax for ax in c.axes if 'timedelta' not in ax.dims]
      if batch_axis is not None:
        axes = [batch_axis] + axes
      return cx.shape_struct_field(cx.coords.compose(*axes))

    return jax.tree.map(
        infer_data_slice_struct,
        input_data_specs,
        is_leaf=is_coord,
    )

  def _read_sharded_fields(
      self, data: dict[str, xarray.Dataset], specs: typing.Pytree
  ) -> typing.Pytree:
    mesh = self.parallelism_mesh
    assert isinstance(mesh, parallelism.Mesh)
    assert mesh.spmd_mesh is not None
    return xarray_utils.read_sharded_from_xarray(
        data,
        specs,
        mesh_shape=mesh.spmd_mesh.shape,
        dim_partitions=mesh.field_partitions[self.loading_partition_schema],
    )

  def _from_xarray_fn(
      self,
      data: dict[str, xarray.Dataset],
      input_data_specs: typing.Pytree,
      dynamic_input_specs: typing.Pytree,
      stencils: dict[str, xreader.TimeStencil],
  ) -> tuple[typing.Pytree, typing.Pytree]:
    """Reads data from xarray datasets."""
    timedelta_origins = {
        k: v.start if v.closed in ['left', 'both'] else v.start + v.step
        for k, v in stencils.items()
    }
    data = xarray_utils.ensure_timedelta_axis(data, timedelta_origins)
    if self.spmd_mesh is None:
      inputs = xarray_utils.read_from_xarray(data, input_data_specs)
      dynamic_inputs = xarray_utils.read_from_xarray(data, dynamic_input_specs)
      return inputs, dynamic_inputs
    return (
        self._read_sharded_fields(data, input_data_specs),
        self._read_sharded_fields(data, dynamic_input_specs),
    )

  def to_global_array(self, pytree: PyTree) -> PyTree:
    """Create a pytree of global JAX arrays from a pytree of NumPy arrays."""
    if self.spmd_mesh is None:
      return pytree

    mesh = self.parallelism_mesh
    assert isinstance(mesh, parallelism.Mesh)  # guaranteed by spmd_mesh check.

    def is_leaf(x):
      return isinstance(x, dict) and any(isinstance(k, jax.Device) for k in x)

    unshard = functools.partial(
        mesh.unshard, schema=self.loading_partition_schema
    )
    unsharded = jax.tree.map(unshard, pytree, is_leaf=is_leaf)
    return unsharded

  def _read_model_parallel_dataset(
      self,
      dataset_dict: dict[str, xarray.Dataset],
      read_shard: Callable[[dict[str, xarray.Dataset], int], grain.IterDataset],
      batch_size_per_device: int | None,
      from_xarray_fn: Callable[[dict[str, xarray.Dataset]], Any],
  ) -> grain.IterDataset:
    """Read a shard of a training dataset into grain.IterDataset."""
    batch_axis = self.make_batch_axis(batch_size_per_device)
    spmd_mesh = self.spmd_mesh
    if spmd_mesh is None:
      data = read_shard(dataset_dict, 0)
      data = data.map(from_xarray_fn)
      if batch_size_per_device is not None:
        data = data.batch(batch_size_per_device)
        data = data.map(lambda x: cx.tag(x, batch_axis))
      return data

    mesh = self.parallelism_mesh
    assert isinstance(mesh, parallelism.Mesh)  # guaranteed by spmd_mesh check.
    partitions = mesh.field_partitions[self.loading_partition_schema]
    if batch_size_per_device is not None:
      spec_names = ('batch',) + self.shardable_dims
    else:
      spec_names = self.shardable_dims

    spec = jax.sharding.PartitionSpec(*(partitions[k] for k in spec_names))
    sharding = jax.sharding.NamedSharding(spmd_mesh, spec)

    spatial_dims = self.shardable_dims

    # Stores datasets groupped by resolution and dataset key.
    # This makes it easier to keep track of resolution-specific index maps.
    datasets_by_res = collections.defaultdict(dict)
    for name, ds in dataset_dict.items():
      res_key = tuple(ds.sizes.get(k, 1) for k in spatial_dims)
      datasets_by_res[res_key][name] = ds

    # Stores shard indices for each device and resolution.
    device_to_indices_by_res = collections.defaultdict(dict)
    for res_key in datasets_by_res:
      sizes = {k: v for k, v in zip(spatial_dims, res_key)}
      # Use .get(k, 1) for dims that may not exist in a particular dataset.
      if batch_axis is not None:
        global_shape = (
            batch_axis.size,
            *(sizes.get(k, 1) for k in spatial_dims),
        )
      else:
        global_shape = tuple(sizes.get(k, 1) for k in spatial_dims)

      indices_map = sharding.addressable_devices_indices_map(global_shape)
      for device, index in indices_map.items():
        device_to_indices_by_res[device][res_key] = index

    # Group devices by their data slices across all resolutions.
    indices_by_res_to_devices = collections.defaultdict(list)
    for device, indices_by_res in device_to_indices_by_res.items():
      # Key must be hashable, so convert dict to a tuple of sorted items.
      # key structure is: tuple[(res_key, device_index), ...].
      key = tuple(sorted(indices_by_res.items()))
      indices_by_res_to_devices[key].append(device)
      # there are devices number of keys in indices_by_res_to_devices.

    def slice_to_shard(slice_: slice) -> int:
      if slice_.start is not None:
        assert batch_size_per_device is not None
        assert slice_.stop - slice_.start == batch_size_per_device
        shard_index = slice_.start // batch_size_per_device
      else:
        shard_index = 0
      return shard_index

    shard_data: list[grain.IterDataset] = []
    # To ensure deterministic order for `zip` in `prep_for_to_global_array`.
    sorted_indices_by_res_keys = sorted(indices_by_res_to_devices.keys())

    for key in sorted_indices_by_res_keys:
      indices_by_res = dict(key)
      if batch_size_per_device is not None:
        # Batch slice is consistent across resolutions for a given device group.
        batch_slice = list(indices_by_res.values())[0][0]
        shard_index = slice_to_shard(batch_slice)
        spatial_slice_start_idx = 1
      else:
        shard_index = 0
        spatial_slice_start_idx = 0

      shard_dataset = {}
      for res_key, fixed_res_datasets in datasets_by_res.items():
        indices_tuple = indices_by_res[res_key]
        spatial_indices_slices = indices_tuple[
            spatial_slice_start_idx:
        ]  # skip the batch index.
        spatial_indices = dict(zip(spatial_dims, spatial_indices_slices))
        for name, ds in fixed_res_datasets.items():
          # we ignore missing dims because some datasets may not have all
          # dimensions e.g. surface data may be missing level dim.
          shard_dataset[name] = ds.isel(spatial_indices, missing_dims='ignore')
      shard_data.append(read_shard(shard_dataset, shard_index))

    if batch_size_per_device is not None:
      batch_axis_shard = parallelism.CoordinateShard(
          batch_axis,
          spmd_mesh.shape,
          mesh.field_partitions[self.loading_partition_schema],
      )

    def prep_for_to_global_array(shards):
      device_to_data = DeviceKeyedDict()
      device_groups = [
          indices_by_res_to_devices[key] for key in sorted_indices_by_res_keys
      ]
      for shard, devices in zip(shards, device_groups):
        for device in devices:
          device_to_data[device] = shard
      return _transpose_for_to_global_array(device_to_data, is_leaf=cx.is_field)

    # shard_data is a list of datasets by resolution and shard indices.
    data = grain.experimental.ZipIterDataset(shard_data)
    # process each (resolution, shard_index) separately.
    data = data.map(lambda shards: [from_xarray_fn(s) for s in shards])
    # shards are converted to coordax Fields and then batched.
    if batch_size_per_device is not None:
      data = data.batch(batch_size_per_device)
      data = data.map(lambda x: cx.tag(x, batch_axis_shard))  # add batch axis.
    # regroup data from resolutions + index to dicts keyed by device, then list
    # of dicts is transposed to the original dict structures keyd by device.
    data = data.map(prep_for_to_global_array)
    return data

  def _iterate_with_updated_buffer(
      self,
      data,
      data_buffer: HostDataBuffer | Sequence[HostDataBuffer] | None,
  ):
    """Iterates over and updates the data buffer inplace, if necessary."""
    if data_buffer is not None:
      if isinstance(data_buffer, HostDataBuffer):
        data_buffer = [data_buffer]

    for batch, dynamic_data in data:
      if data_buffer:
        init_slice = slice_leading_timedelta(batch, 1)
        init_slice = self.to_global_array(init_slice)
        for buffer in data_buffer:
          buffer.set_data(batch)
        inputs = init_slice
      else:
        inputs = batch
      yield inputs, dynamic_data

  def _to_global_dynamic(self, targets_and_dynamic_inputs):
    targets, dynamic_inputs = targets_and_dynamic_inputs
    return targets, self.to_global_array(dynamic_inputs)

  def build_train_inputs(
      self,
      input_data_specs: typing.Pytree,
      dynamic_input_specs: typing.Pytree,
      batch_size_per_device: int | None,
      shuffle_buffer_size_in_bytes: int,
      dataset_rng_seed: int,
      time_sample_offset: np.timedelta64,
      dataset_time_slice: tuple[str, str] | list[tuple[str, str]] | None,
      data_buffer: HostDataBuffer | Sequence[HostDataBuffer] | None = None,
  ) -> Iterator[Any]:
    """Loads the training dataset and returns a data iterator.

    Args:
      input_data_specs: Specifications for input data fields.
      dynamic_input_specs: Specifications for dynamic input fields.
      batch_size_per_device: Number of samples per batch per device.
      shuffle_buffer_size_in_bytes: Size of the shuffle buffer in bytes.
      dataset_rng_seed: Seed for the random number generator used in the
        dataset.
      time_sample_offset: Time between consecutive valid start times.
      dataset_time_slice: Time period(s) to select training data from.
      data_buffer: Buffer(s) to store loaded data. If provided, the data will be
        loaded via callback mechanism.

    Returns:
      An iterator over the training data.
    """
    all_data, stencils, shard_count = self._prep_data_stencils_and_shard_count(
        input_data_specs,
        dynamic_input_specs,
        batch_size_per_device
    )
    if shard_count == 0:
      raise ValueError(
          f'cannot shard training data even once with {batch_size_per_device=}'
      )

    sample_origins = _get_train_sample_origins(
        all_data, stencils, dataset_time_slice, time_sample_offset
    )
    if len(sample_origins) > 0 and len(sample_origins) < shard_count:  # pylint: disable=g-explicit-length-test
      raise ValueError(
          f'Insufficient "{len(sample_origins)=}" for "{shard_count=}".'
      )
    buffer_size_in_bytes = (
        shuffle_buffer_size_in_bytes / jax.local_device_count()
    )
    seed_length_factor = max(len(s.points) for s in stencils.values())
    buffer_diversity = (
        batch_size_per_device if batch_size_per_device is not None else 1
    )

    def read_shard(shard_dataset, shard_index):
      seed = train_utils.combine_rng_seeds(
          dataset_rng_seed, shard_index, seed_length_factor
      )
      return xreader.training_iterator(
          shard_dataset,
          stencils,
          sample_origins,
          num_epochs=None,
          buffer_size_in_bytes=buffer_size_in_bytes,
          buffer_diversity=buffer_diversity,
          shard_index=shard_index,
          shard_count=shard_count,
          seed=seed,
      )

    from_xr = functools.partial(
        self._from_xarray_fn,
        input_data_specs=input_data_specs,
        dynamic_input_specs=dynamic_input_specs,
        stencils=stencils,
    )

    # Read a shard across space and initialization times.
    data = self._read_model_parallel_dataset(
        all_data,
        read_shard,
        batch_size_per_device,
        from_xarray_fn=from_xr,
    )
    if data_buffer is not None:
      data = data.map(self._to_global_dynamic)
    else:
      data = data.map(self.to_global_array)
    data = grain.experimental.ThreadPrefetchIterDataset(
        data, prefetch_buffer_size=1
    )
    return self._iterate_with_updated_buffer(data, data_buffer)

  def build_eval_inputs(
      self,
      input_data_specs: typing.Pytree,
      dynamic_input_specs: typing.Pytree,
      dataset_time_slice: tuple[str, str] | list[tuple[str, str]] | None,
      batch_size_per_device: int | None,
      time_sample_offset: np.timedelta64,
      batch_count: int = 1,
      data_buffer: HostDataBuffer | Sequence[HostDataBuffer] | None = None,
  ) -> list[Any]:
    """Returns an iterable over the data for evaluation.

    Args:
      input_data_specs: Specifications for input data fields.
      dynamic_input_specs: Specifications for dynamic input fields.
      dataset_time_slice: Time period(s) to select evaluation data from.
      batch_size_per_device: Number of samples per batch per device.
      time_sample_offset: Time between consecutive valid sample origins.
      batch_count: Number of batches of evaluation data to return.
      data_buffer: Buffer(s) to store loaded data. If provided, the data will be
        loaded via callback mechanism.

    Returns:
      A list of batches of evaluation data.
    """
    all_data, stencils, shard_count = self._prep_data_stencils_and_shard_count(
        input_data_specs, dynamic_input_specs, batch_size_per_device
    )

    if batch_size_per_device is not None:
      batch_axis = self.make_batch_axis(batch_size_per_device)
      assert batch_axis is not None
      total_sample_count = batch_axis.size
    else:
      batch_axis = None
      total_sample_count = 1

    origins = _get_eval_sample_origins(
        all_data,
        dataset_time_slice,
        stencils,
        batch_count,
        total_sample_count,
        time_sample_offset,
    )

    if shard_count == 0:
      raise ValueError(
          f'cannot shard eval data even once with {batch_axis=} and'
          f' {batch_size_per_device=}'
      )

    def read_shard(shard_dataset, shard_index):
      return xreader.evaluation_iterator(
          shard_dataset,
          stencils,
          origins[shard_index::shard_count],
      )

    from_xr = functools.partial(
        self._from_xarray_fn,
        input_data_specs=input_data_specs,
        dynamic_input_specs=dynamic_input_specs,
        stencils=stencils,
    )

    data = self._read_model_parallel_dataset(
        all_data,
        read_shard,
        batch_size_per_device,  # pytype: disable=attribute-error  # jax-api-types
        from_xarray_fn=from_xr,
    )
    if data_buffer is not None:
      data = data.map(self._to_global_dynamic)
    else:
      data = data.map(self.to_global_array)

    # we call list on the iterator to cache data in memory.
    data = list(self._iterate_with_updated_buffer(data, data_buffer))
    return data
