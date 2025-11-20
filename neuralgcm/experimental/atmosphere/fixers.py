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

"""Module-based API for calculating diagnostics of NeuralGCM models."""

from typing import Protocol

import coordax as cx
from flax import nnx
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import units


class TendencyFixerModule(Protocol):
  """Protocol for energy balance modules that adjust tendencies."""

  def __call__(
      self,
      imbalance: dict[str, cx.Field],
      tendencies: dict[str, cx.Field],
      *args,
      **kwargs,
  ) -> dict[str, cx.Field]:
    """Adjusts tendencies based on energy imbalance."""


@nnx_compat.dataclass
class TemperatureAdjustmentForEnergyBalance(nnx.Module):
  """Adjusts temperature tendency to conserve energy based on imbalance.

  This module adjusts temperature tendency to account for energy imbalance,
  assuming that dE/dt due to this adjustment is p_s/g * Cp * delta_T, and
  that this should be equal to `imbalance`.
  The adjustment delta_T is computed as:
  delta_T = imbalance * g / (p_s * Cp)
  This adjustment is added to `tendencies['temperature']`.
  """

  ylm_map: spherical_harmonics.FixedYlmMapping
  levels: coordinates.SigmaLevels
  sim_units: units.SimUnits
  transform: transforms.TransformABC | None = None
  prognostics_arg_key: str | int = 'prognostics'
  imbalance_diagnostic_key: str = 'imbalance'

  @property
  def imbalance_query(self) -> dict[str, cx.Coordinate]:
    return {self.imbalance_diagnostic_key: self.ylm_map.nodal_grid}

  def __call__(
      self,
      imbalance: dict[str, cx.Field],
      tendencies: dict[str, cx.Field],
      *args,
      **kwargs,
  ) -> dict[str, cx.Field]:
    """Returns tendencies with d_temperature/d_t adjusted to conserve energy."""
    tendencies = tendencies.copy()
    if isinstance(self.prognostics_arg_key, int):
      prognostics = args[self.prognostics_arg_key]
    else:
      prognostics = kwargs.get(self.prognostics_arg_key)
    if not isinstance(prognostics, dict):
      raise ValueError(
          f'Prognostics must be a dictionary, got {type(prognostics)=} instead.'
      )
    if self.transform is not None:
      imbalance = self.transform(imbalance)
    # We want to add delta_T_tendency such that:
    # integral(Cp * delta_T_tendency) dp / g = imbalance
    # If delta_T_tendency is constant in vertical:
    # Cp * delta_T_tendency / g * integral() dp = imbalance
    # Cp * delta_T_tendency / g * (p_surface - p_top) = imbalance
    # in sigma coords integral(1)dp ~ p_surface if p_top=0.
    # With sigma_integral, integral(f dp) ~ sigma_integral(f) * p_surface.
    # So integral(1 dp) ~ p_surface.

    # If we add uniform dTemp_tend, its integrated effect is:
    # integral(Cp * dTemp_tend) dp / g = Cp * dTemp_tend / g * p_surface
    # so: Cp * dTemp_tend / g * p_surface = imbalance
    # dTemp_tend = imbalance * g / (Cp * p_surface)
    to_nodal = self.ylm_map.to_nodal
    p_surface = cx.cmap(jnp.exp)(to_nodal(prognostics['log_surface_pressure']))
    cp = self.sim_units.Cp
    g = self.sim_units.gravity_acceleration
    delta_t_tendency = imbalance['imbalance'] * g / (cp * p_surface)

    delta_t_tendency_modal = self.ylm_map.to_modal(delta_t_tendency)
    tendencies['temperature'] += delta_t_tendency_modal
    return tendencies
