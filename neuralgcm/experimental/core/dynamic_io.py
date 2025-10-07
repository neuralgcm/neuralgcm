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

"""API for providing dynamic inputs to NeuralGCM models."""

import abc

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import typing
import numpy as np


DynamicInput = typing.DynamicInput


class DynamicInputModule(nnx.Module, abc.ABC):
  """Base class for modules that interface with dynamically supplied data."""

  @abc.abstractmethod
  def update_dynamic_inputs(self, dynamic_inputs):
    """Ingests relevant data from `dynamic_inputs` onto the internal state."""
    raise NotImplementedError()

  @abc.abstractmethod
  def __call__(self, time: jdt.Datetime) -> typing.Pytree:
    """Returns dynamic data at the specified time."""
    raise NotImplementedError()


class DynamicInputSlice(DynamicInputModule):
  """Exposes inputs from the most recent available time slice."""

  def __init__(
      self,
      keys_to_coords: dict[str, cx.Coordinate],
      observation_key: str,
      time_axis: int = 0,
  ):
    self.keys_to_coords = keys_to_coords
    self.observation_key = observation_key
    self.time_axis = time_axis
    mock_dt = coordinates.TimeDelta(np.array([np.timedelta64(1, 'h')]))
    self.time = DynamicInput(
        cx.wrap(jdt.to_datetime('1970-01-01T00')[np.newaxis], mock_dt)
    )
    dummy_data = {}
    for k, v in self.keys_to_coords.items():
      value = jnp.nan * jnp.zeros(mock_dt.shape + v.shape)
      dummy_data[k] = cx.wrap(value, mock_dt, v)
    self.data = DynamicInput(dummy_data)

  def update_dynamic_inputs(
      self, dynamic_inputs: dict[str, dict[str, cx.Field]]
  ) -> None:
    if self.observation_key not in dynamic_inputs:
      # TODO(dkochkov): Consider allowing partial updates.
      raise ValueError(
          f'Observation key {self.observation_key!r} not found in dynamic'
          f' inputs: {dynamic_inputs.keys()}'
      )
    inputs = dynamic_inputs[self.observation_key]
    if 'time' not in inputs:
      raise ValueError(
          f'Dynamic inputs under key {self.observation_key!r} do not have the'
          f" required 'time' variable: {inputs.keys()}"
      )
    time = inputs['time']
    self.time.value = time
    data_dict = {}
    for k in self.keys_to_coords:
      if k not in inputs:
        # TODO(dkochkov): Consider allowing partial updates.
        raise ValueError(
            f'Key {k!r} not found in dynamic inputs: {inputs.keys()}'
        )
      v = inputs[k]
      if v.axes.get('timedelta', None) != time.axes['timedelta']:
        raise ValueError(f'{v.axes=} does not contain {time.axes=}.')
      data_coord = cx.compose_coordinates(
          *[v.axes[d] for d in v.dims if d != 'timedelta']
      )
      expected_coord = cx.compose_coordinates(
          *[self.data.value[k].axes[d] for d in v.dims if d != 'timedelta']
      )
      if data_coord != expected_coord:
        raise ValueError(
            f'Coordinate mismatch for key {k!r}: {data_coord=} !='
            f' {expected_coord=}'
        )
      data_dict[k] = v
    self.data.value = data_dict

  def _slice_data_at_time(
      self,
      time: typing.Array,
      available_time: typing.Array,
      array: typing.Array,
  ) -> typing.Array:
    """Returns slice of array ."""
    time_indices = jnp.arange(available_time.size)
    approx_index = jdt.interp(time, available_time, time_indices)
    # TODO(shoyer): switch to jnp.floor?
    index = jnp.round(approx_index).astype(int)
    return jax.lax.dynamic_index_in_dim(array, index, keepdims=False)

  def __call__(self, time: cx.Field) -> dict[str, cx.Field]:
    """Returns covariates at the specified time."""
    outputs = {}
    for k, v in self.data.value.items():
      field_index_fn = cx.cmap(self._slice_data_at_time)
      outputs[k] = field_index_fn(
          time, self.time.value.untag('timedelta'), v.untag('timedelta')
      )
    return outputs
