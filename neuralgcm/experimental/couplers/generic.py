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
import jax.numpy as jnp
import jax_datetime as jdt
from neuralgcm.experimental.core import module_utils
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import typing


@nnx_compat.dataclass
class InstantDataCoupler(nnx.Module):
  """Coupler that stores instant field values copied from model components."""

  field_coords: dict[str, cx.Coordinate]  # coordinates for stored variables.
  var_to_data_keys: dict[str, str]  # field_name to data_key.
  coupling_fields: nnx.Data = dataclasses.field(init=False)

  def __post_init__(self):
    missing_data_keys = set(self.field_coords) - set(self.var_to_data_keys)
    if missing_data_keys:
      raise ValueError(
          f'InstantDataCoupler missing data keys {missing_data_keys}.'
      )
    self.coupling_fields = {
        k: typing.Coupling(cx.field(jnp.nan * jnp.zeros(c.shape), c))
        for k, c in self.field_coords.items()
    }

  @module_utils.ensure_unchanged_state_structure
  def update_fields(self, nested_inputs: dict[str, dict[str, cx.Field]]):
    """Updates the fields stored on the coupler from the observations."""
    for k, _ in (self.field_coords).items():
      data_key = self.var_to_data_keys[k]
      if data_key not in nested_inputs:
        raise ValueError(
            f'DataCoupler missing Fields for "{data_key}" needed for "{k}"'
        )
      self.coupling_fields[k].value = nested_inputs[data_key][k]

  def __call__(self, time: jdt.Datetime) -> dict[str, cx.Field]:
    del time
    return {k: v.value for k, v in self.coupling_fields.items()}
