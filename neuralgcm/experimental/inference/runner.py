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
"""High performance inference API for NeuralGCM models."""
import contextlib
import dataclasses
import logging
import math
import pickle
from typing import Any
import uuid

import dask.array
from etils import epath
from flax import nnx
import jax
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.inference import dynamic_inputs as dynamic_inputs_lib
from neuralgcm.experimental.inference import streaming
import numpy as np
import numpy.typing as npt
import xarray

# pylint: disable=logging-fstring-interpolation

NestedData = dict[str, xarray.Dataset]  # eventually, use xarray.DataTree


def _query_to_dummy_datatree(query: typing.Query) -> xarray.DataTree:
  """Create a DataTree of dummy dask.array objects, matching the given query."""
  outputs = {}
  for group, sub_query in query.items():
    ds = xarray.Dataset()
    for var, coord in sub_query.items():
      ds[var] = xarray.DataArray(
          # TODO(shoyer): include dtype as part of the query spec?
          data=dask.array.zeros(coord.shape, np.float32),
          dims=coord.dims,
          coords=coord.to_xarray(),
      )
    outputs[group] = ds
  return xarray.DataTree.from_dict(outputs)


# TODO(shoyer): Add these methods upstream onto xarray.DataTree.


def _datatree_expand_dims(
    tree: xarray.DataTree, **kwargs: int | list[Any] | np.ndarray
) -> xarray.DataTree:
  """Expand dimensions in a DataTree."""
  return tree.map_over_datasets(lambda ds: ds.expand_dims(**kwargs))


def _datatree_rename(
    tree: xarray.DataTree, renames: dict[str, str]
) -> xarray.DataTree:
  """Rename variables and coordinates in a DataTree."""
  return tree.map_over_datasets(
      lambda ds: ds.rename({k: v for k, v in renames.items() if k in ds})
  )


def _coordinate_to_root(tree: xarray.DataTree, name: str) -> xarray.DataTree:
  """Move a coordinate to the root of a DataTree."""
  tree = tree.copy()
  for node in tree.subtree:
    if name in node.coords:
      coord = node.coords[name]
      del node.coords[name]
      if name in tree.coords:
        if not tree.coords[name].equals(coord):
          raise ValueError(
              f'Coordinate {name=} {tree.coords[name]=} != {coord=}'
          )
      else:
        tree.coords[name] = coord
  return tree


@nnx.jit
def _assimilate(model, input_obs, dynamic_inputs, rng):
  return model.assimilate(input_obs, dynamic_inputs=dynamic_inputs, rng=rng)


@nnx.jit(static_argnames=['steps_per_unroll', 'output_freq'])
def _unroll(
    model, state, dynamic_inputs, output_query, steps_per_unroll, output_freq
):
  def observe(state, model):
    return model.observe(
        state,
        output_query,
        dynamic_inputs=dynamic_inputs,
    )

  return model.unroll(
      state,
      dynamic_inputs=dynamic_inputs,
      outer_steps=steps_per_unroll,
      timedelta=output_freq,
      post_process_fn=observe,
  )


def _tmp_file_path(path: str) -> epath.Path:
  return epath.Path(path).with_suffix(f'.{uuid.uuid4().hex}.tmp')


@contextlib.contextmanager
def _atomic_write(path):
  tmp_path = _tmp_file_path(path)
  try:
    with tmp_path.open('wb') as f:
      yield f
    tmp_path.replace(path)
  except Exception:  # pylint: disable=broad-except
    tmp_path.unlink(missing_ok=True)
    raise


@dataclasses.dataclass
class InferenceRunner:
  """High performance inference runner for NeuralGCM models."""

  model: api.ForecastSystem
  inputs: NestedData
  dynamic_inputs: dynamic_inputs_lib.DynamicInputs
  init_times: npt.NDArray[np.datetime64]
  ensemble_size: int | None  # ensemble_size=None for deterministic models
  output_path: str
  output_query: typing.Query
  output_freq: np.timedelta64
  output_duration: np.timedelta64
  output_chunks: dict[str, int]  # should have a sane defaults
  unroll_duration: np.timedelta64
  checkpoint_duration: np.timedelta64
  zarr_format: int | None = None
  random_seed: int = 0

  def __post_init__(self):
    if self.output_duration % self.output_freq != np.timedelta64(0):
      raise ValueError(
          f'{self.output_duration=} must be a multiple of {self.output_freq=}'
      )
    if self.unroll_duration % self.output_freq != np.timedelta64(0):
      raise ValueError(
          f'{self.unroll_duration=} must be a multiple of {self.output_freq=}'
      )
    if self.output_chunks['lead_time'] % self.steps_per_unroll != 0:
      raise ValueError(
          f"output_chunks['lead_time'] ({self.output_chunks['lead_time']}) must"
          ' be a multiple of unroll_duration in steps'
          f' ({self.steps_per_unroll})'
      )
    write_freq = self.output_freq * self.max_steps_per_write
    if self.checkpoint_duration % write_freq != np.timedelta64(0):
      raise ValueError(
          f'{self.checkpoint_duration=} must be a multiple of {write_freq=}'
      )

  @property
  def total_steps(self) -> int:
    return math.ceil(self.output_duration / self.output_freq)

  @property
  def steps_per_unroll(self) -> int:
    return self.unroll_duration // self.output_freq

  @property
  def steps_per_checkpoint(self) -> int:
    return self.checkpoint_duration // self.output_freq

  @property
  def max_steps_per_write(self) -> int:
    return self.output_chunks['lead_time']

  def _checkpoints_path(self) -> epath.Path:
    # names beginning with __ are reserved for Zarr v3 extensions:
    # https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#node-names
    return epath.Path(self.output_path) / '__inference_checkpoints'

  def setup(self) -> None:
    """Call once to setup metadata in output Zarr(s)."""
    checkpoints_path = self._checkpoints_path()
    if checkpoints_path.exists():
      # Already setup.
      return

    # Our strategy here is to create a DataTree where all variables are empty
    # dask.array objects, similar xarray_beam.make_template()
    template = _query_to_dummy_datatree(self.output_query)
    lead_times = np.arange(0, self.output_duration, self.output_freq)
    expanded_dims = {}
    if self.ensemble_size is not None:
      expanded_dims['realization'] = np.arange(self.ensemble_size)
    expanded_dims['init_time'] = self.init_times.astype('datetime64[ns]')
    expanded_dims['lead_time'] = lead_times.astype('timedelta64[ns]')

    # The use of expand_dims() here replicates
    # xarray_beam.replace_template_dims().
    template = _datatree_expand_dims(template, **expanded_dims)
    template = template.chunk(self.output_chunks)

    # TODO(shoyer): Determine if there's anything we need to do to work around
    # idempotency issues in Zarr, or if using Zarr v3 is sufficient:
    # https://github.com/zarr-developers/zarr-python/issues/1435

    # TODO(shoyer): Switch to Zarr v3, which will require setting a default fill
    # value explicitly: https://github.com/pydata/xarray/issues/10646
    template.to_zarr(self.output_path, compute=False, zarr_format=2)

    # Add directory for inference checkpoints. The existance of this directory
    # is used as a marker to indicate that setup() has been called.
    checkpoints_path.mkdir(exist_ok=True)

  @property
  def task_count(self) -> int:
    """Total number of simulation tasks."""
    ensemble_count = 1 if self.ensemble_size is None else self.ensemble_size
    return len(self.init_times) * ensemble_count

  def _task_path(self, task_id: int) -> epath.Path:
    return self._checkpoints_path() / f'{task_id}.pkl'

  def run(self, task_id: int) -> None:
    """Simulate a single task, saving outputs to the Zarr file.

    Each task corresponds to an initial condition and ensemble RNG.

    Args:
      task_id: integer task ID to run, from the range [0, task_count).

    Raises:
      SomeException: if setup() has not been completed first.
    """
    init_time_index = (
        task_id if self.ensemble_size is None else task_id // self.ensemble_size
    )
    init_time = self.init_times[init_time_index].astype('datetime64[ns]')
    dynamic_inputs_forecast = self.dynamic_inputs.get_forecast(init_time)

    # Initialize state from checkpoint or from scratch.
    task_path = self._task_path(task_id)
    if task_path.exists():
      logging.info(f'state checkpoint already exists for task {task_id=}')
      with task_path.open('rb') as f:
        state, commited_steps = pickle.load(f)
    else:
      logging.info(f'no state checkpoint found for task {task_id=}')
      if self.ensemble_size is not None:
        base_key = jax.random.key(self.random_seed)
        rng = jax.random.fold_in(base_key, task_id)
      else:
        rng = None

      # TODO(shoyer): Can we get the number of time-steps to include in model
      # inputs from the model (or at least an InferenceRunner property) instead
      # of hard-coding it here?
      selected_inputs = {
          k: v.sel(time=[init_time]) for k, v in self.inputs.items()
      }
      input_obs = self.model.inputs_from_xarray(selected_inputs)
      dynamic_inputs = self.model.dynamic_inputs_from_xarray(
          dynamic_inputs_forecast.get_data(
              np.timedelta64(0), self.model.timestep
          )
      )
      logging.info('assimilating state')
      state = _assimilate(self.model, input_obs, dynamic_inputs, rng)
      commited_steps = 0

    # Unroll simulation forward in time.

    def get_dynamic_inputs(output_step):
      lead_start = output_step * self.output_freq
      lead_stop = (output_step + self.steps_per_unroll) * self.output_freq
      return self.model.dynamic_inputs_from_xarray(
          dynamic_inputs_forecast.get_data(lead_start, lead_stop)
      )

    dynamic_inputs_task = streaming.SingleTaskExecutor(get_dynamic_inputs)
    dynamic_inputs_task.submit(commited_steps)

    commit_chunk_task = streaming.SingleTaskExecutor(self._commit_chunk)

    for steps_written in range(
        commited_steps, self.total_steps, self.max_steps_per_write
    ):
      output_buffer = []
      chunk_end_step = min(
          steps_written + self.max_steps_per_write, self.total_steps
      )

      for step_start in range(
          steps_written, chunk_end_step, self.steps_per_unroll
      ):
        dynamic_inputs = dynamic_inputs_task.get()
        if step_start + self.steps_per_unroll < self.total_steps:
          dynamic_inputs_task.submit(step_start + self.steps_per_unroll)

        state, trajectory_slice = _unroll(
            self.model,
            state,
            dynamic_inputs,
            output_query=self.output_query,
            steps_per_unroll=self.steps_per_unroll,
            output_freq=self.output_freq,
        )
        output_buffer.append(jax.device_get(trajectory_slice))

      commit_chunk_task.wait()
      commit_chunk_task.submit(state, output_buffer, task_id, steps_written)
    commit_chunk_task.wait()

  def _commit_chunk(
      self,
      state: typing.Pytree,
      output_buffer: list[typing.Pytree],
      task_id: int,
      steps_written: int,
  ) -> None:
    """Write outputs to disk, checking state if necessary."""
    logging.info(f'committing chunk for {task_id=} and {steps_written=}')
    self._write_zarr_chunk(output_buffer, task_id, steps_written)
    self._maybe_update_state_checkpoint(state, task_id, steps_written)
    logging.info('commit chunk complete')

  def _write_zarr_chunk(
      self, output_buffer: list[typing.Pytree], task_id: int, steps_written: int
  ) -> None:
    """Write Zarr chunk to disk."""
    if self.ensemble_size is None:
      init_time_index = task_id
      realization_index = None
    else:
      init_time_index = task_id // self.ensemble_size
      realization_index = task_id % self.ensemble_size

    trees = []
    for trajectory_slice in output_buffer:
      outputs = self.model.data_to_xarray(trajectory_slice)
      outputs_tree = xarray.DataTree.from_dict(outputs)
      # TODO(shoyer): update the name of the TimeDelta coordinate?
      outputs_tree = _datatree_rename(outputs_tree, {'timedelta': 'lead_time'})
      trees.append(outputs_tree)

    tree = xarray.map_over_datasets(
        lambda *xs: xarray.concat(xs, dim='lead_time'), *trees
    )
    tree = _coordinate_to_root(tree, 'lead_time')
    num_steps_in_chunk = tree.sizes['lead_time']

    init_time = self.init_times[init_time_index].astype('datetime64[ns]')
    tree = _datatree_expand_dims(tree, init_time=[init_time])
    region = {'init_time': slice(init_time_index, init_time_index + 1)}
    region['lead_time'] = slice(
        steps_written, steps_written + num_steps_in_chunk
    )
    if realization_index is not None:
      tree = _datatree_expand_dims(tree, realization=[realization_index])
      region['realization'] = slice(realization_index, realization_index + 1)

    # Remove variables that don't have an init_time dimension. These shouldn't
    # be written to disk again.
    for node in tree.subtree:
      for name, variable in node.variables.items():
        if not any(dim in variable.dims for dim in region):
          del node[name]

    logging.info('writing zarr chunk')
    tree.to_zarr(self.output_path, mode='r+', region=region)

  def _maybe_update_state_checkpoint(
      self, state: typing.Pytree, task_id: int, steps_written: int
  ) -> None:
    """Update the state checkpoint, if necessary."""
    if steps_written + self.max_steps_per_write >= self.total_steps:
      logging.info('task complete, removing temporary state checkpoint')
      self._task_path(task_id).unlink(missing_ok=True)
    elif steps_written % self.steps_per_checkpoint == 0:
      logging.info('writing state checkpoint')
      task_path = self._task_path(task_id)
      state = jax.device_get(state)
      with _atomic_write(task_path) as f:
        pickle.dump((state, steps_written + self.max_steps_per_write), f)
    else:
      logging.info('no state checkpoint update for this step')
