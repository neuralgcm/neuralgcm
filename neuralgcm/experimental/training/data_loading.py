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
from typing import Any, TypeVar, cast

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
K = TypeVar('K')
V = TypeVar('V')


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
) -> np.ndarray:
  """Get equispaced forecast start times for evaluating against ERA5."""

  # To match ECMWF, all forecasts should be initialized at 0z or 12z. To
  # approximate this, we round the candidates to whole days, shift every other
  # one by 12 hours, and then find the closest times in the orinal candidate
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


def _get_sample_origins(
    time_axis: pd.Series,
    time_slices: tuple[str, str] | list[tuple[str, str]] | None,
    stencil: xreader.TimeStencil,
    stride_between_windows: int,
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
    stride_between_windows: This determines the number of time values between
      consecutive sample origins. When multiple slices are used, the stride is
      applied separately within each slice.

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

  def make_slice(x):
    if x is None:
      return slice(None)  # the whole thing
    if isinstance(x, slice):
      return x
    else:
      return slice(*x)

  time_slices = [make_slice(x) for x in time_slices]

  # Collect origin times from each slice.
  sample_origins = []
  for ts in time_slices:
    slice_times = pd.Series(time_axis, index=time_axis).loc[ts].index
    new_sample_origins = slice_times[::stride_between_windows]
    new_sample_origins = xreader.valid_origin_points(
        new_sample_origins, stencil
    )
    sample_origins.append(new_sample_origins)

  return np.concatenate(sample_origins)


def select_targets(inputs, queries):
  """Selects targets matching queries from inputs."""
  # TODO(dkochkov) This is a temporary simpler solution for the case where no
  # queries include dynamic fields. Eventually those should be excluded here and
  # from the outputs of obseravation operators.
  targets = {}
  for source, specs in queries.items():
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
      local_cpu: jax.Device | None,
      desired_spatial_dims_layout: tuple[str, ...] = (),
  ):
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

    self.current_batch = jax.tree.map(
        _optimize_field_for_slice, batch, is_leaf=cx.is_field
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
    input_data_specs: Specifications for input data fields.
    dynamic_input_specs: Specifications for dynamic input fields.
    queries_spec: Specifications for a subset of input fields to be used as
      targets for the loss.
    training_mesh: Parallel mesh with batch and ensemble sharding for training.
    batch_axis: Axis representing the batch dimension.
    vertical_name: Name of the vertical dimension.
    model_timestep: Timestep of the model.
    inner_steps: Number of inner steps per model step.
    load_data_via_callback: Whether to load data via a callback. If True,
      DataLoader is _stateful_, and updates `data_buffer` inplace using
      each iteration.
    callback_pinned_host: Whether to use pinned host memory for the callback
      buffer.
    callback_spatial_dims_layout: The desired spatial dimensions layout for the
      callback buffer.
  """

  all_data: dict[str, xarray.Dataset]
  input_data_specs: typing.Pytree
  dynamic_input_specs: typing.Pytree
  training_mesh: parallelism.Mesh
  batch_axis: cx.SizedAxis
  vertical_name: str
  model_timestep: pd.Timedelta
  inner_steps: int
  load_data_via_callback: bool
  queries_spec: typing.Pytree
  callback_pinned_host: bool = False
  callback_spatial_dims_layout: tuple[str, ...] = ()

  def __post_init__(self):
    if self.load_data_via_callback:
      devices = jax.local_devices()
      self._local_cpu = colocated_python.colocated_cpu_devices(devices)[0]
      if self.callback_pinned_host:
        self._data_buffer = HostDataBuffer(
            self._local_cpu, tuple(self.callback_spatial_dims_layout)
        )
      else:
        self._data_buffer = HostDataBuffer(
            None, tuple(self.callback_spatial_dims_layout)
        )
    else:
      self._local_cpu = None
      self._data_buffer = None

  @property
  def spmd_mesh(self) -> jax.sharding.Mesh:
    """JAX sharding mesh."""
    mesh = self.training_mesh.spmd_mesh
    assert mesh is not None
    return mesh

  def coord_to_shard(self, coord: cx.Coordinate) -> cx.Coordinate:
    return cx.coords.compose(*[
        parallelism.CoordinateShard(
            ax,
            self.training_mesh.shape,
            self.training_mesh.field_partitions['physics'],
        )
        for ax in coord.axes
    ])

  def add_input_shard_optimizations(
      self,
      data_shard_struct: PyTree,
  ) -> PyTree:
    """Adds memory layout optimizations to input data shard struct."""

    def _with_pinned_host_mem(shape_dtype) -> jax.ShapeDtypeStruct:
      return jax.ShapeDtypeStruct(
          shape=shape_dtype.shape,
          dtype=shape_dtype.dtype,
          sharding=jax.sharding.SingleDeviceSharding(
              device=self._local_cpu, memory_kind='pinned_host'
          ),
      )

    if self.callback_pinned_host:
      data_shard_struct = jax.tree.map(_with_pinned_host_mem, data_shard_struct)
    if self.callback_spatial_dims_layout:
      layout_dims = self.callback_spatial_dims_layout
      order_as_fn = lambda f: f.order_as(*layout_dims, ...)
      order_as_for_struct = lambda fs: jax.eval_shape(order_as_fn, fs)
      data_shard_struct = jax.tree.map(
          order_as_for_struct, data_shard_struct, is_leaf=cx.is_field
      )
    return data_shard_struct

  def setup_targets_via_callback(
      self,
      data_slice_struct: PyTree,
  ) -> Callable[[int], PyTree]:
    """Sets up data retrieval functions for use in jitted train/eval steps."""
    device_by_id = {d.id: d for d in jax.devices()}

    # Remove values that won't be needed for the loss to minimize data transfer.
    def host_slice_fn(idx, device_id):
      """Gets a slice from the buffer and selects targets."""
      device = device_by_id[device_id.item()]
      sliced_data = self._data_buffer.get_slice(idx)  # pytype: disable=attribute-error
      sliced_data = select_targets(sliced_data, self.queries_spec)

      def _is_leaf(x):
        return isinstance(x, dict) and any(isinstance(k, jax.Device) for k in x)

      result = jax.tree.map(lambda x: x[device], sliced_data, is_leaf=_is_leaf)
      return result

    target_slice_struct = select_targets(data_slice_struct, self.queries_spec)
    shard_slice_struct = jax.tree.map(
        lambda f: cx.shape_struct_field(self.coord_to_shard(f.coordinate)),
        target_slice_struct,
        is_leaf=cx.is_field,
    )

    def _retrieve_local_shard(idx, device_id):
      """Retrieves a local arrays via callback for each host."""
      shards = jax.pure_callback(
          host_slice_fn,
          self.add_input_shard_optimizations(shard_slice_struct),
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
          f.dims, self.training_mesh.field_partitions['physics']
      )
      out_specs = jax.tree.map(
          get_out_spec, target_slice_struct, is_leaf=cx.is_field
      )
      spmd_devices = self.spmd_mesh.devices
      device_indices = jnp.array([d.id for d in spmd_devices.ravel()]).reshape(
          spmd_devices.shape
      )
      merged_shards = jax.shard_map(
          _retrieve_local_shard,
          mesh=self.spmd_mesh,
          in_specs=(
              jax.sharding.PartitionSpec(),
              jax.sharding.PartitionSpec(*self.spmd_mesh.axis_names),
          ),
          out_specs=out_specs,
      )(idx, device_indices)
      coords = jax.tree.map(
          lambda f: parallelism.get_unsharded(f.coordinate),
          target_slice_struct,
          is_leaf=cx.is_field,
      )
      result = jax.tree.map(
          cx.wrap,
          merged_shards,
          coords,
      )
      return self.training_mesh.with_sharding_constraint(result, 'physics')

    return _retrieve_global_targets

  def data_slice_struct(self):
    """Returns shape struct of a time slice of input data."""
    is_coord_or_spec = lambda x: isinstance(x, cx.Coordinate) or isinstance(
        x, data_specs.CoordSpec
    )
    # Prepend batch axis which is not in inputs_spec and drop timedelta.
    infer_data_slice_struct = lambda c: cx.shape_struct_field(
        cx.coords.compose(
            self.batch_axis,
            *(ax for ax in c.coord.axes if 'timedelta' not in ax.dims),
        )
    )
    return jax.tree.map(
        infer_data_slice_struct,
        self.input_data_specs,
        is_leaf=is_coord_or_spec,
    )

  def _read_sharded_fields(
      self, data: dict[str, xarray.Dataset], specs: typing.Pytree
  ) -> typing.Pytree:
    return xarray_utils.read_sharded_from_xarray(
        data,
        specs,
        mesh_shape=self.spmd_mesh.shape,
        dim_partitions=self.training_mesh.field_partitions['physics'],
    )

  def from_xarray_fn(
      self, data: dict[str, xarray.Dataset]
  ) -> tuple[typing.Pytree, typing.Pytree]:
    data = xarray_utils.ensure_timedelta_axis(data)
    return (
        self._read_sharded_fields(data, self.input_data_specs),  # pytype: disable=attribute-error  # jax-api-types
        self._read_sharded_fields(data, self.dynamic_input_specs),  # pytype: disable=attribute-error  # jax-api-types
    )

  def to_global_array(self, pytree: PyTree) -> PyTree:
    """Create a pytree of global JAX arrays from a pytree of NumPy arrays."""

    def is_leaf(x):
      return isinstance(x, dict) and any(isinstance(k, jax.Device) for k in x)

    unshard = functools.partial(self.training_mesh.unshard, schema='physics')
    unsharded = jax.tree.map(unshard, pytree, is_leaf=is_leaf)
    return unsharded

  def _read_model_parallel_dataset(
      self,
      dataset_dict: dict[str, xarray.Dataset],
      read_shard: Callable[[dict[str, xarray.Dataset], int], grain.IterDataset],
      global_batch_size: int,
      batch_size_per_device: int,
  ) -> grain.IterDataset:
    """Read a shard of a training dataset into grain.IterDataset."""
    partitions = self.training_mesh.field_partitions['physics']
    spec = jax.sharding.PartitionSpec(
        *(partitions[k] for k in ('batch', 'level', 'longitude', 'latitude'))
    )
    sharding = jax.sharding.NamedSharding(self.spmd_mesh, spec)

    spatial_dims = (self.vertical_name, 'longitude', 'latitude')

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
      global_shape = (
          global_batch_size,
          *(sizes.get(k, 1) for k in spatial_dims),
      )
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
      # Batch slice is consistent across resolutions for a given device group.
      batch_slice = list(indices_by_res.values())[0][0]
      shard_index = slice_to_shard(batch_slice)

      shard_dataset = {}
      for res_key, fixed_res_datasets in datasets_by_res.items():
        indices_tuple = indices_by_res[res_key]
        spatial_indices_slices = indices_tuple[1:]  # skip the batch index.
        spatial_indices = dict(zip(spatial_dims, spatial_indices_slices))
        for name, ds in fixed_res_datasets.items():
          # we ignore missing dims because some datasets may not have all
          # dimensions e.g. surface data may be missing level dim.
          shard_dataset[name] = ds.isel(spatial_indices, missing_dims='ignore')
      shard_data.append(read_shard(shard_dataset, shard_index))

    batch_axis = parallelism.CoordinateShard(
        self.batch_axis,
        self.spmd_mesh.shape,
        self.training_mesh.field_partitions['physics'],
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
    data = data.map(lambda shards: [self.from_xarray_fn(s) for s in shards])
    # shards are converted to coordax Fields and then batched.
    data = data.batch(batch_size_per_device)
    data = data.map(lambda x: cx.tag(x, batch_axis))  # add batch axis.
    # regroup data from resolutions + index to dicts keyed by device, then list
    # of dicts is transposed to the original dict structures keyd by device.
    data = data.map(prep_for_to_global_array)
    return data

  def _get_shared_time_axis(self) -> pd.DatetimeIndex:
    time_axes = [data.indexes['time'] for data in self.all_data.values()]  # pytype: disable=attribute-error  # jax-api-types
    time_axes = cast(list[pd.DatetimeIndex], time_axes)
    time = time_axes[0]
    for time_axis in time_axes[1:]:
      time = time.intersection(time_axis, sort=True)
    return time

  def _get_train_sample_origins(
      self,
      stencil: xreader.TimeStencil,
      dataset_time_slice: tuple[str, str] | list[tuple[str, str]] | None,
      time_sample_offset: int,
  ) -> np.ndarray:
    return _get_sample_origins(
        time_axis=self._get_shared_time_axis(),
        time_slices=dataset_time_slice,
        stencil=stencil,
        stride_between_windows=time_sample_offset,
    )

  def _get_eval_sample_origins(
      self,
      time_slices: tuple[str, str] | list[tuple[str, str]] | None,
      stencil: xreader.TimeStencil,
      batch_count: int,
      global_batch_size: int,
  ) -> np.ndarray:
    """Get sample origins for evaluation."""

    sample_count = global_batch_size * batch_count

    time = self._get_shared_time_axis()
    sample_origins = _get_sample_origins(
        time_axis=time,
        time_slices=time_slices,
        stencil=stencil,
        stride_between_windows=1,
    )
    return _get_datetime_forecast_starts(sample_count, sample_origins)

  def _iterate_with_updated_buffer(self, data):
    """Iterates over and updates the data buffer inplace, if necessary."""
    for batch, dynamic_data in data:
      if self.load_data_via_callback:
        init_slice = slice_leading_timedelta(batch, 1)
        init_slice = self.to_global_array(init_slice)
        # TODO(dkochkov): figure out a saner way to write this
        self._data_buffer.set_data(  # pytype: disable=attribute-error
            select_targets(batch, self.queries_spec)
        )
        inputs = init_slice
      else:
        inputs = batch
      yield inputs, dynamic_data

  def build_train_inputs(
      self,
      time_series_length: int,
      batch_size_per_device: int,
      global_batch_size: int,
      shuffle_buffer_size_in_bytes: int,
      dataset_rng_seed: int,
      time_sample_offset: int,
      dataset_time_slice: tuple[str, str] | list[tuple[str, str]] | None,
  ) -> Iterator[Any]:
    """Loads the training dataset and returns a data iterator.

    Args:
      time_series_length: Length of the time series for each sample.
      batch_size_per_device: Number of samples per batch per device.
      global_batch_size: Total number of samples per batch across all devices.
      shuffle_buffer_size_in_bytes: Size of the shuffle buffer in bytes.
      dataset_rng_seed: Seed for the random number generator used in the
        dataset.
      time_sample_offset: Stride for sampling start times of training
        trajectories.
      dataset_time_slice: Time period(s) to select training data from.

    Returns:
      An iterator over the training data.
    """
    shard_count = global_batch_size // batch_size_per_device
    if shard_count == 0:
      raise ValueError(
          'cannot shard training data even once with'
          f' {global_batch_size=} and {batch_size_per_device=}'
      )

    dt = self.model_timestep * self.inner_steps  # pytype: disable=attribute-error  # jax-api-types
    stencil = xreader.TimeStencil(
        start='0h',
        stop=time_series_length * dt,
        step=dt,
    )

    sample_origins = self._get_train_sample_origins(
        stencil, dataset_time_slice, time_sample_offset
    )

    buffer_size_in_bytes = (
        shuffle_buffer_size_in_bytes / jax.local_device_count()
    )

    def read_shard(shard_dataset, shard_index):
      seed = train_utils.combine_rng_seeds(
          dataset_rng_seed, shard_index, time_series_length
      )
      return xreader.training_iterator(
          shard_dataset,
          stencil,
          sample_origins,
          num_epochs=None,
          buffer_size_in_bytes=buffer_size_in_bytes,
          buffer_diversity=batch_size_per_device,
          shard_index=shard_index,
          shard_count=shard_count,
          seed=seed,
      )

    # Read a shard across space and initialization times.
    data = self._read_model_parallel_dataset(
        self.all_data,  # pytype: disable=attribute-error  # jax-api-types
        read_shard,
        global_batch_size,
        batch_size_per_device,
    )
    if self.load_data_via_callback:

      def _to_global_dynamic(targets_and_dynamic_inputs):
        targets, dynamic_inputs = targets_and_dynamic_inputs
        return targets, self.to_global_array(dynamic_inputs)

      data = data.map(_to_global_dynamic)
    else:
      data = data.map(self.to_global_array)
    data = grain.experimental.ThreadPrefetchIterDataset(
        data, prefetch_buffer_size=1
    )
    return self._iterate_with_updated_buffer(data)

  def build_eval_inputs(
      self,
      dataset_time_slices: tuple[str, str] | list[tuple[str, str]] | None,
      train_trajectory_length: int,
      num_init_frames: int,
      eval_trajectory_length: int,
      batch_size_per_device: int,
      global_batch_size: int,
      batch_count: int = 1,
  ) -> list[Any]:
    """Returns an iterable over the data for evaluation.

    Args:
      dataset_time_slices: Time period(s) to select evaluation data from.
      train_trajectory_length: Length of the training trajectory.
      num_init_frames: Number of frames to use for initialization.
      eval_trajectory_length: Length of the evaluation trajectory.
      batch_size_per_device: Number of samples per batch per device.
      global_batch_size: Total number of samples per batch across all devices.
      batch_count: Number of batches of evaluation data to return.

    Returns:
      A list of batches of evaluation data.
    """

    time_series_length = max(
        eval_trajectory_length + num_init_frames - 1,  # pytype: disable=attribute-error  # jax-api-types
        train_trajectory_length,
    )

    dt = self.model_timestep * self.inner_steps  # pytype: disable=attribute-error  # jax-api-types
    stencil = xreader.TimeStencil(
        start='0h', stop=time_series_length * dt, step=dt
    )
    origins = self._get_eval_sample_origins(
        dataset_time_slices, stencil, batch_count, global_batch_size
    )

    shard_count = global_batch_size // batch_size_per_device

    if shard_count == 0:
      raise ValueError(
          f'cannot shard eval data even once with {global_batch_size=} and'
          f' {batch_size_per_device=}'
      )

    def read_shard(shard_dataset, shard_index):
      return xreader.evaluation_iterator(
          shard_dataset,
          stencil,
          origins[shard_index::shard_count],
      )

    data = self._read_model_parallel_dataset(
        self.all_data,
        read_shard,
        global_batch_size,
        batch_size_per_device,  # pytype: disable=attribute-error  # jax-api-types
    )
    if self.load_data_via_callback:

      def _to_global_dynamic(targets_and_dynamic_inputs):
        targets, dynamic_inputs = targets_and_dynamic_inputs
        return targets, self.to_global_array(dynamic_inputs)

      data = data.map(_to_global_dynamic)
    else:
      data = data.map(self.to_global_array)

    data = list(self._iterate_with_updated_buffer(data))  # cache in memory
    return data
