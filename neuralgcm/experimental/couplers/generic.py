# Copyright 2026 Google LLC

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
"""Generic coupler components that implement typing  API."""

from __future__ import annotations

import dataclasses

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import dynamic_io
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import typing
import numpy as np


@nnx_compat.dataclass
class InstantDataCoupler(nnx.Module, pytree=False):
  """Coupler that stores instant field values copied from model components."""

  field_coords: dict[str, cx.Coordinate]  # coordinates for stored variables.
  var_to_data_keys: dict[str, str]  # field_name to data_key.
  coupling_fields: nnx.Dict = dataclasses.field(init=False)

  def __post_init__(self):
    missing_data_keys = set(self.field_coords) - set(self.var_to_data_keys)
    if missing_data_keys:
      raise ValueError(
          f'InstantDataCoupler missing data keys {missing_data_keys}.'
      )
    self.coupling_fields = nnx.Dict({
        k: typing.Coupling(cx.field(jnp.nan * jnp.zeros(c.shape), c))
        for k, c in self.field_coords.items()
    })

  @module_utils.ensure_unchanged_state_structure
  def update_fields(self, nested_inputs: dict[str, dict[str, cx.Field]]):
    """Updates the fields stored on the coupler from the observations."""
    for k, _ in (self.field_coords).items():
      data_key = self.var_to_data_keys[k]
      if data_key not in nested_inputs:
        raise ValueError(
            f'missing Fields for "{data_key}" needed for "{k}"'
        )
      self.coupling_fields[k].set_value(nested_inputs[data_key][k])

  def __call__(self, time: jdt.Datetime) -> dict[str, cx.Field]:
    del time
    return {k: v.get_value() for k, v in self.coupling_fields.items()}


@nnx_compat.dataclass
class MultiTimeDataCoupler(nnx.Module, pytree=False):
  """Coupler that stores coupling field over multiple time slices.

  Attributes:
    field_coords: Maps coupling field names to their spatial coordinates.
    var_to_data_keys: Maps field names to the top-level key in the nested input.
    time_data_key: The top-level key for 'time' field in the nested inputs.
    n_time_slices: Number of time slices to store in the rolling buffer.
  """

  field_coords: dict[str, cx.Coordinate]  # coordinates for stored variables.
  var_to_data_keys: dict[str, str]  # field_name to data_key.
  time_data_key: str  # data_key for time.
  n_time_slices: int
  coupling_fields: nnx.Dict = dataclasses.field(init=False)
  times: typing.Coupling = dataclasses.field(init=False)

  def __post_init__(self):
    missing_data_keys = set(self.field_coords) - set(self.var_to_data_keys)
    if missing_data_keys:
      raise ValueError(
          f'MultiTimeDataCoupler missing data keys {missing_data_keys}.'
      )
    self.time_slice_axis = cx.SizedAxis('time_slice_idx', self.n_time_slices)
    full_coords = {
        k: cx.coords.compose(self.time_slice_axis, c)
        for k, c in self.field_coords.items()
    }
    self.coupling_fields = nnx.Dict({
        k: typing.Coupling(cx.field(jnp.nan * jnp.zeros(c.shape), c))
        for k, c in full_coords.items()
    })
    dummy_times = jdt.to_datetime('1970-01-01T00') + jdt.to_timedelta(
        np.arange(self.n_time_slices) - self.n_time_slices, 's'
    )
    self.times = typing.Coupling(cx.field(dummy_times, self.time_slice_axis))

  @module_utils.ensure_unchanged_state_structure
  def update_fields(self, nested_inputs: dict[str, dict[str, cx.Field]]):
    """Updates the fields stored on the coupler from the observations."""
    if self.time_data_key not in nested_inputs:
      raise ValueError(
          f'missing Fields for "{self.time_data_key}" needed for time.'
      )
    idx_ax = self.time_slice_axis
    update = lambda arr, new_arr: jnp.concat([arr[1:], new_arr[None, ...]])
    update_tree = lambda tree, new_tree: jax.tree.map(update, tree, new_tree)
    update_fn = cx.cmap(update_tree)
    # Update time.
    time = nested_inputs[self.time_data_key]['time']
    times = update_fn(self.times.get_value().untag(idx_ax), time).tag(idx_ax)
    self.times.set_value(times)
    # Update fields.
    for k, _ in self.field_coords.items():
      data_key = self.var_to_data_keys[k]
      if data_key not in nested_inputs:
        raise ValueError(
            f'missing Fields for "{data_key}" needed for "{k}"'
        )
      previous = self.coupling_fields[k].get_value()
      latest_f = nested_inputs[data_key][k]
      new_values = update_fn(previous.untag(idx_ax), latest_f).tag(idx_ax)
      self.coupling_fields[k].set_value(new_values)

  def __call__(self, time: cx.Field) -> dict[str, cx.Field]:
    idx_ax = self.time_slice_axis
    field_index_fn = cx.cmap(dynamic_io.slice_data_by_time)
    times = self.times.get_value().untag(idx_ax)
    outputs = {}
    for k, v in self.coupling_fields.items():
      outputs[k] = field_index_fn(time, times, v.get_value().untag(idx_ax))
    return outputs
