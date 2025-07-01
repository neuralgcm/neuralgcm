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
import functools

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import typing
import numpy as np


class DynamicInputValue(nnx.Intermediate):
  ...


class DynamicInputModule(nnx.Module, abc.ABC):
  """Base class for modules that interface with dynamically supplied data."""

  @abc.abstractmethod
  def update_dynamic_inputs(self, dynamic_inputs):
    """Ingests relevant data from `dynamic_inputs` onto the internal state."""
    raise NotImplementedError()

  @abc.abstractmethod
  def output_shapes(self) -> typing.Pytree:
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
    self.time = DynamicInputValue(jdt.to_datetime('1970-01-01T00')[np.newaxis])
    mock_dt = coordinates.TimeDelta(np.array([np.timedelta64(1, 'h')]))
    dummy_data = {}
    for k, v in self.keys_to_coords.items():
      value = jnp.nan * jnp.zeros(mock_dt.shape + v.shape)
      dummy_data[k] = cx.wrap(value, mock_dt, v)
    self.data = DynamicInputValue(dummy_data)

  def update_dynamic_inputs(self, dynamic_inputs):
    if self.observation_key not in dynamic_inputs:
      raise ValueError(
          f'Observation key {self.observation_key} not found in dynamic inputs'
          f' {dynamic_inputs.keys()}'
      )
    inputs = dynamic_inputs[self.observation_key]
    if 'time' not in inputs:
      raise ValueError(
          f'Dynamic inputs under key {self.observation_key} do not have a'
          f' required time variable {inputs.keys()}'
      )
    time = inputs['time']
    if time.ndim != 1 or time.dims[0] != 'timedelta':
      raise ValueError(f'Expected time to be 1D timedelta, got {time.dims=}')
    self.time = DynamicInputValue(time.data)
    data_dict = {}
    for k, expected_coord in self.keys_to_coords.items():
      if k not in inputs:
        raise ValueError(f'Key {k} not found in dynamic inputs {inputs.keys()}')
      v = inputs[k]
      if v.axes.get('timedelta', None) != time.axes['timedelta']:
        raise ValueError(f'{v.axes=} does not contain {time.axes=}.')
      data_coord = cx.compose_coordinates(
          *[v.axes[d] for d in v.dims if d != 'timedelta']
      )
      if data_coord != expected_coord:
        raise ValueError(
            f'Coordinate mismatch for key {k}: {data_coord=} !='
            f' {expected_coord=}'
        )
      data_dict[k] = v
    self.data = DynamicInputValue(data_dict)

  def output_shapes(self) -> typing.Pytree:
    return {
        k: typing.ShapeFloatStruct(coord.shape)
        for k, coord in self.keys_to_coords.items()
    }

  def __call__(self, time: cx.Field) -> dict[str, cx.Field]:
    """Returns covariates at the specified time."""
    time = time.unwrap()
    time_indices = jnp.arange(self.time.size)
    approx_index = jdt.interp(time, self.time.value, time_indices)
    index = jnp.round(approx_index).astype(int)
    field_index_fn = functools.partial(
        jax.lax.dynamic_index_in_dim,
        index=index,
        keepdims=False,
    )
    outputs = {}
    for k, v in self.data.value.items():  # pylint: disable=attribute-error
      out_axes = {k: i for i, k in enumerate(self.keys_to_coords[k].dims)}
      outputs[k] = cx.cmap(field_index_fn, out_axes)(v.untag('timedelta'))
    return outputs
