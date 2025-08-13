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

import dask.array
from neuralgcm.experimental.core import api
from neuralgcm.experimental.core import typing
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


@dataclasses.dataclass
class InferenceRunner:
  model: api.ForecastSystem
  inputs: NestedData
  dynamic_inputs: NestedData  # if needed separately from inputs
  init_times: npt.NDArray[np.datetime64]
  ensemble_size: int | None  # ensemble_size=None for deterministic models
  output_path: str
  output_query: typing.Query
  output_freq: (
      np.timedelta64
  )  # or potentially dict[str, timedelta] for nested outputs
  output_duration: np.timedelta64
  output_chunks: dict[str, int]  # should have a sane defaults

  def setup(self) -> None:
    """Call once to setup metadata in output Zarr(s)."""
    # Our strategy here is to create a DataTree where all variables are empty
    # dask.array objects, similar xarray_beam.make_template()
    dummy_outputs = _query_to_dummy_datatree(self.output_query)
    lead_times = np.arange(0, self.output_duration, self.output_freq)
    expanded_dims = {
        'init_time': self.init_times.astype('datetime64[ns]'),
        'lead_time': lead_times.astype('timedelta64[ns]'),
    }
    if self.ensemble_size is not None:
      expanded_dims['realization'] = np.arange(self.ensemble_size)

    # The use of expand_dims() here replicates
    # xarray_beam.replace_template_dims().
    expanded_outputs: xarray.DataTree = dummy_outputs.map_over_datasets(  # type: ignore
        lambda ds: ds.expand_dims(expanded_dims).chunk(self.output_chunks)
    )
    # TODO(shoyer): Consider checking if the output Zarr file with the correct
    # metadata already exists, and if so, skipping setup.
    # TODO(shoyer): Determine if there's anything we need to do to work around
    # idempotency issues in Zarr, or if using Zarr v3 is sufficient:
    # https://github.com/zarr-developers/zarr-python/issues/1435
    expanded_outputs.to_zarr(self.output_path, compute=False)

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
