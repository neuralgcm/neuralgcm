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
import dataclasses
import math
from typing import Any

import dask.array
from flax import nnx
import jax
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.inference import dynamic_inputs as dynamic_inputs_lib
import numpy as np
import numpy.typing as npt
import xarray

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
  zarr_format: int | None = None
  random_seed: int = 0

  def setup(self) -> None:
    """Call once to setup metadata in output Zarr(s)."""
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

    # TODO(shoyer): Consider checking if the output Zarr file with the correct
    # metadata already exists, and if so, skipping setup.

    # TODO(shoyer): Determine if there's anything we need to do to work around
    # idempotency issues in Zarr, or if using Zarr v3 is sufficient:
    # https://github.com/zarr-developers/zarr-python/issues/1435

    # TODO(shoyer): Switch to Zarr v3, which will require setting a default fill
    # value explicitly: https://github.com/pydata/xarray/issues/10646
    template.to_zarr(self.output_path, compute=False, zarr_format=2)

  @property
  def task_count(self) -> int:
    """Total number of simulation tasks."""
    ensemble_count = 1 if self.ensemble_size is None else self.ensemble_size
    return len(self.init_times) * ensemble_count

  def run(self, task_id: int) -> None:
    """Simulate a single task, saving outputs to the Zarr file.

    Each task corresponds to an initial condition and ensemble RNG.

    Args:
      task_id: integer task ID to run, from the range [0, task_count).

    Raises:
      SomeException: if setup() has not been completed first.
    """
    if self.ensemble_size is None:
      init_time_index = task_id
      realization_index = None
      rng = None
    else:
      init_time_index = task_id // self.ensemble_size
      realization_index = task_id % self.ensemble_size
      base_key = jax.random.key(self.random_seed)
      rng = jax.random.fold_in(base_key, task_id)

    init_time = self.init_times[init_time_index].astype('datetime64[ns]')
    # TODO(shoyer): Can we get the number of time-steps to include in model
    # inputs from the model (or at least an InferenceRunner property) instead of
    # hard-coding it here?
    selected_inputs = {
        k: v.sel(time=[init_time]) for k, v in self.inputs.items()
    }
    forecast_inputs = self.dynamic_inputs.get_forecast(init_time).get_data(
        lead_start=np.timedelta64(0, 'h'), lead_stop=self.output_duration
    )

    @nnx.jit
    def assimilate(model, input_obs, dynamic_inputs, rng):
      return model.assimilate(input_obs, dynamic_inputs=dynamic_inputs, rng=rng)

    @nnx.jit
    def unroll(model, state, dynamic_inputs):

      def observe(state, model):
        return model.observe(
            state,
            self.output_query,
            dynamic_inputs=dynamic_inputs,
        )

      return model.unroll(
          state,
          dynamic_inputs=dynamic_inputs,
          outer_steps=math.ceil(self.output_duration / self.output_freq),
          timedelta=self.output_freq,
          post_process_fn=observe,
      )

    input_obs = self.model.inputs_from_xarray(selected_inputs)
    dynamic_inputs = self.model.dynamic_inputs_from_xarray(forecast_inputs)
    state = assimilate(self.model, input_obs, dynamic_inputs, rng)
    # TODO(shoyer): Call unroll() in a loop, writing outputs to Zarr as we go
    # and checkpointing `state`.
    _, trajectory_slice = unroll(self.model, state, dynamic_inputs)
    outputs = self.model.data_to_xarray(trajectory_slice)

    outputs_tree = xarray.DataTree.from_dict(outputs)
    outputs_tree = _datatree_expand_dims(outputs_tree, init_time=[init_time])
    # TODO(shoyer): Rename TimeDelta to LeadTime in
    # neuralgcm.experimental.core.coordinates.
    outputs_tree = _datatree_rename(outputs_tree, {'timedelta': 'lead_time'})

    # Remove variables that don't have an init_time dimension. These shouldn't
    # be written to disk again.
    for node in outputs_tree.subtree:
      for name, variable in node.variables.items():
        if 'init_time' not in variable.dims:
          del node[name]

    # TODO(shoyer): See if we can get region='auto' to work.
    region = {'init_time': slice(init_time_index, init_time_index + 1)}
    if realization_index is not None:
      outputs_tree = _datatree_expand_dims(
          outputs_tree, realization=[realization_index]
      )
      region['realization'] = slice(realization_index, realization_index + 1)

    outputs_tree.to_zarr(self.output_path, mode='r+', region=region)
