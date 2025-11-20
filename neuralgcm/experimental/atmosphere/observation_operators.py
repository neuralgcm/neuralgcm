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

"""Atmosphere-specific observation operators."""

import copy
import dataclasses

import coordax as cx
from neuralgcm.experimental.atmosphere import state_conversion
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import units


@dataclasses.dataclass
class StandardVariablesObservationOperator(
    observation_operators.ObservationOperator
):
  """Operator that predicts atmospheric observations in "u,v,t,z,..." vars.

  This operator is compatible with models that use primitive equation
  representation of the atmospheric prognostics in spherical harmonics basis.
  The observations are computed by converting the state to
  velocity/temperature/geopotential representation and then linearly
  interpolating to the desired vertical levels using combination of linear and
  constant extrapolation strategies. If `observation_correction` mapping is
  specified, an additional correction is added to the predicted observations in
  lon-lat space.
  """

  ylm_map: spherical_harmonics.FixedYlmMapping
  orography: orographies.ModalOrography
  levels: coordinates.PressureLevels | coordinates.SigmaLevels
  sim_units: units.SimUnits
  observation_correction: transforms.Transform | None
  mesh: parallelism.Mesh

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Observes fields in `query` on pressure/lon/lat coords from YLM `inputs`.

    Args:
      inputs: A dictionary of fields in spherical harmonic representation.
      query: A dictionary specifying coordinates of the queried fields.

    Returns:
      Predicted fields matching the `query`.
    """
    inputs = copy.copy(inputs)  # avoid mutating inputs.
    time = inputs.pop('time')
    interpolated = state_conversion.primitive_equations_to_uvtz(
        inputs=inputs,
        ylm_map=self.ylm_map,
        levels=self.levels,
        orography=self.orography,
        sim_units=self.sim_units,
    )
    interpolated = parallelism.with_physics_sharding(self.mesh, interpolated)
    if self.observation_correction is not None:
      inputs = parallelism.with_dycore_sharding(self.mesh, inputs)
      correction = self.observation_correction(inputs | {'time': time})
      correction = pytree_utils.replace_with_matching_or_default(
          interpolated, correction
      )
      correction = parallelism.with_physics_sharding(self.mesh, correction)
      add_fn = lambda x, y: x + y if y is not None else x
      interpolated = {
          k: add_fn(v, correction.get(k, None))
          for k, v in interpolated.items()
      }

    return observation_operators.DataObservationOperator(
        interpolated | {'time': time}
    ).observe(inputs, query)
