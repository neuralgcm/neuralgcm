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
from typing import Any, TypeVar
import uuid

import dask.array
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.inference import dynamic_inputs as dynamic_inputs_lib
from neuralgcm.experimental.inference import streaming
from neuralgcm.experimental.inference import timing
import numpy as np
import numpy.typing as npt
import pandas as pd
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


def device_put_to_cpu(x: typing.Pytree) -> typing.Pytree:
  """Non-blocking transfer of JAX arrays to CPU."""
  cpu_device = jax.devices('cpu')[0]
  return jax.device_put(x, cpu_device)


class BadStateError(Exception):
  """Error raised when an invalid state value (i.e., NaN) is encountered."""


@jax.jit
def _any_nans(x: typing.Pytree) -> jax.Array:
  return jnp.any(jnp.array([jnp.isnan(x).any() for x in jax.tree.leaves(x)]))


def check_pytree_for_bad_state(state: typing.Pytree, msg: str) -> None:
  """Check a pytree for bad state (i.e., NaN).

  This computation is always performed on the CPU, to avoid blocking computation
  on accelerators.

  Args:
    state: pytree to check.
    msg: message to include in the exception if a bad state is found.

  Raises:
    BadStateError: if a bad state is found.
  """
  if _any_nans(device_put_to_cpu(state)):
    raise BadStateError(msg)


T = TypeVar('T')


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

  model: api.ForecastSystem | None
  inputs: NestedData
  dynamic_inputs: dynamic_inputs_lib.DynamicInputs
  init_times: npt.NDArray[np.datetime64] | pd.DatetimeIndex
  ensemble_size: int | None  # ensemble_size=None for deterministic models
  output_path: str
  output_query: typing.Query
  output_freq: np.timedelta64
  output_duration: np.timedelta64
  unroll_duration: np.timedelta64
  write_duration: np.timedelta64
  checkpoint_duration: np.timedelta64
  zarr_attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
  zarr_chunks: dict[str, int] = dataclasses.field(default_factory=dict)
  zarr_shards: dict[str, int] | None = None
  bad_state_strategy: str = 'raise'
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
    if self.write_duration % self.output_freq != np.timedelta64(0):
      raise ValueError(
          f'{self.write_duration=} must be a multiple of {self.output_freq=}'
      )
    if self.max_steps_per_write % self.steps_per_unroll != 0:
      raise ValueError(
          f'write_duration in steps ({self.max_steps_per_write}) must be a '
          f'multiple of unroll_duration in steps ({self.steps_per_unroll})'
      )
    if self.max_steps_per_write % self.zarr_chunks['lead_time'] != 0:
      raise ValueError(
          f'write_duration in steps ({self.max_steps_per_write}) must be a'
          " multiple of zarr_chunks['lead_time']"
          f" ({self.zarr_chunks['lead_time']})"
      )
    if self.checkpoint_duration % self.write_duration != np.timedelta64(0):
      raise ValueError(
          f'{self.checkpoint_duration=} must be a multiple of'
          f' {self.write_duration=}'
      )
    if self.bad_state_strategy not in ('raise', 'impute_nan'):
      raise ValueError(f'Unknown bad_state_strategy: {self.bad_state_strategy}')
    if isinstance(self.init_times, pd.DatetimeIndex):
      self.init_times = self.init_times.to_numpy()

  @property
  def total_steps(self) -> int:
    return math.ceil(self.output_duration / self.output_freq)

  @property
  def steps_per_unroll(self) -> int:
    return self.unroll_duration // self.output_freq

  @property
  def max_steps_per_write(self) -> int:
    return self.write_duration // self.output_freq

  @property
  def steps_per_checkpoint(self) -> int:
    return self.checkpoint_duration // self.output_freq

  def _checkpoints_path(self) -> epath.Path:
    # names beginning with __ are reserved for Zarr v3 extensions:
    # https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#node-names
    return epath.Path(self.output_path) / '__inference_checkpoints'

  def _get_array_encoding(
      self, array: xarray.DataArray
  ) -> dict[str, tuple[int, ...]]:
    """Return encoding for a DataArray."""
    chunks = tuple(
        self.zarr_chunks.get(dim, size) for dim, size in array.sizes.items()
    )
    # TODO(shoyer): Remove fill_value after Xarray defaults to NaN:
    # https://github.com/pydata/xarray/pull/10757
    encoding = {'chunks': chunks, 'fill_value': np.nan}
    zarr_shards = self.zarr_shards
    if zarr_shards is not None:
      encoding['shards'] = tuple(
          zarr_shards.get(dim, chunksize)
          for dim, chunksize in zip(array.dims, chunks)
      )
    return encoding

  def setup(self) -> None:
    """Call once to setup metadata in output Zarr(s)."""
    checkpoints_path = self._checkpoints_path()
    if checkpoints_path.exists():
      logging.info('InferenceRunner.setup() already called, skipping')
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
    template.attrs.update(self.zarr_attrs)

    # TODO(shoyer): Determine if there's anything we need to do to work around
    # idempotency issues in Zarr, or if using Zarr v3 is sufficient:
    # https://github.com/zarr-developers/zarr-python/issues/1435

    encoding = {}
    for node in template.subtree:
      encoding[node.path] = {
          k: self._get_array_encoding(v) for k, v in node.data_vars.items()
      }

    logging.info(
        f'writing template to {self.output_path}:\n{encoding=}\n{template}'
    )
    template.to_zarr(
        self.output_path,
        compute=False,
        encoding=encoding,
        mode='w',  # override if exists, to ensure robustness to preemptions
        zarr_format=3,
    )

    # Add directory for inference checkpoints. The existance of this directory
    # is used as a marker to indicate that setup() has been called.
    checkpoints_path.mkdir(exist_ok=True, mode=0o775)
    logging.info('InferenceRunner.setup() complete')

  @property
  def task_count(self) -> int:
    """Total number of simulation tasks."""
    ensemble_count = 1 if self.ensemble_size is None else self.ensemble_size
    return len(self.init_times) * ensemble_count

  def _task_path(self, task_id: int) -> epath.Path:
    return self._checkpoints_path() / f'task_{task_id}.pkl'

  def run(self, task_id: int) -> None:
    """Simulate a single task, saving outputs to the Zarr file.

    Each task corresponds to an initial condition and ensemble RNG.

    Args:
      task_id: integer task ID to run, from the range [0, task_count).

    Raises:
      SomeException: if setup() has not been completed first.
    """
    if self.model is None:
      raise ValueError('model must be set before calling InferenceRunner.run()')
    init_time_index = (
        task_id if self.ensemble_size is None else task_id // self.ensemble_size
    )
    init_time = self.init_times[init_time_index].astype('datetime64[ns]')
    dynamic_inputs_forecast = self.dynamic_inputs.get_forecast(init_time)

    # TODO(shoyer): Find a less hacky way to ensure that calling
    # dynamic_inputs_from_xarray() is thread-safe.
    model2 = nnx.clone(self.model)

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
      input_obs = model2.inputs_from_xarray(selected_inputs)
      dynamic_inputs = model2.dynamic_inputs_from_xarray(
          dynamic_inputs_forecast.get_data(
              np.timedelta64(0), self.model.timestep
          )
      )
      logging.info('assimilating initial state')
      state = _assimilate(self.model, input_obs, dynamic_inputs, rng)
      check_pytree_for_bad_state(state, f'initial state for {task_id=}')
      commited_steps = 0

    # Unroll simulation forward in time.

    def get_dynamic_inputs(output_step):
      lead_start = output_step * self.output_freq
      lead_stop = (output_step + self.steps_per_unroll) * self.output_freq
      logging.info(f'getting dynamic inputs for {output_step=}')
      xarray_inputs = dynamic_inputs_forecast.get_data(lead_start, lead_stop)
      logging.info(f'xarray_inputs: {xarray_inputs}')
      data = model2.dynamic_inputs_from_xarray(xarray_inputs)
      check_pytree_for_bad_state(data, f'dynamic inputs at {output_step=}')
      keys = {k: list(v) for k, v in data.items()}
      logging.info(f'dynamic inputs ready: {keys}')
      return data

    @timing.Timed
    def unroll(state, dynamic_inputs):
      return _unroll(
          self.model,
          state,
          dynamic_inputs,
          output_query=self.output_query,
          steps_per_unroll=self.steps_per_unroll,
          output_freq=self.output_freq,
      )

    dynamic_inputs_task = streaming.SingleTaskExecutor(get_dynamic_inputs)
    commit_chunk_task = streaming.SingleTaskExecutor(self._commit_chunk)

    dynamic_inputs_task.submit(commited_steps)
    write_step_timer = timing.Timer()

    try:
      for steps_written in range(
          commited_steps, self.total_steps, self.max_steps_per_write
      ):
        write_step_timer.begin_step()
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

          logging.info(f'{dynamic_inputs=}')
          state, trajectory_slice = unroll(state, dynamic_inputs)
          output_buffer.append(device_put_to_cpu(trajectory_slice))

        commit_chunk_task.wait()
        commit_chunk_task.submit(state, output_buffer, task_id, steps_written)
        write_step_timer.finish_step()

        logging.info(
            f'write step took {write_step_timer.last:.3g}s '
            f'({unroll.timer.total:.3g}s for compute, '
            f'{dynamic_inputs_task.timer.total:.3g}s for dynamic inputs, '
            f'{commit_chunk_task.timer.last:.3g}s for commit chunk)'
        )
        dynamic_inputs_task.timer.reset_total()
        unroll.timer.reset_total()

      commit_chunk_task.wait()

    except BadStateError as error:
      if self.bad_state_strategy == 'raise':
        error.add_note(f'{task_id=} failed')
        raise
      else:
        logging.warning(
            f'Bad state encountered, imputing NaN for remainder of {task_id=}:'
            f' {error}'
        )

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
    if self.model is None:
      raise ValueError('model must be set before calling InferenceRunner.run()')
    if self.ensemble_size is None:
      init_time_index = task_id
      realization_index = None
    else:
      init_time_index = task_id // self.ensemble_size
      realization_index = task_id % self.ensemble_size

    logging.info('preparing DataTree for writing to zarr')
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

    logging.info('setting up zarr outputs')
    delayed = tree.chunk().to_zarr(
        self.output_path, mode='r+', region=region, compute=False
    )

    logging.info(f'writing {tree.nbytes/1e6:.1f} MB to zarr')
    delayed.compute(num_workers=128)

    check_pytree_for_bad_state(
        output_buffer, f'unroll outputs at {steps_written=}'
    )

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
      state = device_put_to_cpu(state)
      saved_step = steps_written + self.max_steps_per_write
      with _atomic_write(task_path) as f:
        contents = (state, saved_step)
        pickle.dump(contents, f, protocol=pickle.HIGHEST_PROTOCOL)
      # Check after dumping to pickle file to aid in debugging.
      check_pytree_for_bad_state(state, f'unroll state at {saved_step=}')
    else:
      logging.info('no state checkpoint update for this step')
