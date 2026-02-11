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

import dataclasses
from typing import Protocol, Sequence

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.atmosphere import diagnostics
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units


class FixerModule(Protocol):
  """Protocol for energy balance modules that adjust tendencies."""

  @property
  def diagnostics_query(self) -> dict[str, cx.Coordinate]:
    ...

  def __call__(
      self,
      diagnostic: dict[str, cx.Field],
      targets: dict[str, cx.Field],
      *args,
      **kwargs,
  ) -> dict[str, cx.Field]:
    """Adjusts targets (state or tendencies) based on diagnostics."""


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
  transform: typing.Transform | None = None
  prognostics_arg_key: str | int = 'prognostics'
  imbalance_diagnostic_key: str = 'imbalance'

  @property
  def diagnostics_query(self) -> dict[str, cx.Coordinate]:
    return {self.imbalance_diagnostic_key: self.ylm_map.nodal_grid}

  def __call__(
      self,
      diagnostic: dict[str, cx.Field],
      targets: dict[str, cx.Field],
      *args,
      **kwargs,
  ) -> dict[str, cx.Field]:
    """Returns tendencies with d_temperature/d_t adjusted to conserve energy."""
    tendencies = targets.copy()
    if isinstance(self.prognostics_arg_key, int):
      prognostics = args[self.prognostics_arg_key]
    else:
      prognostics = kwargs.get(self.prognostics_arg_key)
    if not isinstance(prognostics, dict):
      raise ValueError(
          f'Prognostics must be a dictionary, got {type(prognostics)=} instead.'
      )
    if self.transform is not None:
      imbalance = self.transform(diagnostic)
    else:
      imbalance = diagnostic
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
    delta_t_tendency = (
        imbalance[self.imbalance_diagnostic_key] * g / (cp * p_surface)
    )

    delta_t_tendency_modal = self.ylm_map.to_modal(delta_t_tendency)
    tendencies['temperature'] += delta_t_tendency_modal
    return tendencies


@nnx_compat.dataclass
class GlobalEnergyFixer(nnx.Module):
  """Adjusts temperature to conserve global energy based on budget prediction.

  This module adjusts temperature based on discrepancy between predicted
  global energy and actual energy of the evolved state.
  The predicted energy E_col*(t1) is obtained from `ExtractColumnEnergyBudget`
  module, which computes it based on state at t0 and integrated fluxes:
  E_col*(t1) = E_col(t0) + RT - FS.
  The energy of state at t1 is E_col(t1).
  We calculate the global energy deficit:
  Delta_E = integral(E_col*(t1) - E_col(t1))dS
  and adjust temperature uniformly to compensate for it.
  """

  ylm_map: spherical_harmonics.FixedYlmMapping
  levels: coordinates.SigmaLevels
  sim_units: units.SimUnits
  model_orography: orographies.ModalOrography
  use_liquid_ice_moist_static_energy: bool = False
  predict_energy_budget: diagnostics.ExtractColumnEnergyBudget = (
      dataclasses.field(init=False)
  )

  def __post_init__(self):
    self.predict_energy_budget = diagnostics.ExtractColumnEnergyBudget(
        self.ylm_map,
        self.levels,
        self.sim_units,
        self.model_orography,
        use_liquid_ice_moist_static_energy=self.use_liquid_ice_moist_static_energy,
    )

  @property
  def diagnostics_query(self) -> dict[str, cx.Coordinate]:
    return {'column_energy_budget': self.ylm_map.nodal_grid}

  def __call__(
      self,
      diagnostic: dict[str, cx.Field],
      targets: dict[str, cx.Field],
      *args,
      **kwargs,
  ) -> dict[str, cx.Field]:
    """Computes temperature tendency adjustment to conserve energy."""
    column_energy_t1 = self.predict_energy_budget(targets)['column_energy']
    column_energy_star_t1 = diagnostic['column_energy_budget']

    # Calculate total mass to distribute energy deficit uniformly per unit mass.
    to_nodal = self.ylm_map.to_nodal
    p_surface = cx.cmap(jnp.exp)(to_nodal(targets['log_surface_pressure']))
    g = self.sim_units.gravity_acceleration
    column_mass = p_surface / g
    total_mass = self.ylm_map.nodal_grid.integrate(column_mass, radius=1.0)

    # delta_T = (Target_E - Current_E) / (Cp * Total_Mass)
    # We compute (Target_E - Current_E) by integrating the difference of column
    # energies, to avoid precision issues.
    energy_diff = self.ylm_map.nodal_grid.integrate(
        column_energy_star_t1 - column_energy_t1, radius=1.0
    )
    cp = self.sim_units.Cp
    delta_t = energy_diff / (cp * total_mass) # uniform
    t_nodal = to_nodal(targets['temperature'])
    t_adj_nodal = t_nodal + delta_t

    state_adj = targets.copy()
    state_adj['temperature'] = self.ylm_map.to_modal(t_adj_nodal)
    return state_adj


@nnx_compat.dataclass
class GlobalDryAirMassFixer(nnx.Module):
  """Adjusts surface pressure to conserve global dry air mass.

  This module adjusts surface pressure based on discrepancy between dry air mass
  at t0 and t1: M_d(t1) such that M_d_adj(t1) = M_d(t0).
  The adjustment factor gamma = M_d(t0) / M_d(t1) is used to adjust
  surface pressure: p_s_adj(t1) = gamma * p_s(t1).
  """

  ylm_map: spherical_harmonics.FixedYlmMapping
  levels: coordinates.SigmaLevels | coordinates.HybridLevels
  sim_units: units.SimUnits
  extract_column_dry_air_mass: diagnostics.ExtractColumnDryAirMass = (
      dataclasses.field(init=False)
  )

  def __post_init__(self):
    self.extract_column_dry_air_mass = diagnostics.ExtractColumnDryAirMass(
        self.ylm_map,
        self.levels,
        self.sim_units,
    )

  @property
  def diagnostics_query(self) -> dict[str, cx.Coordinate]:
    return {'column_dry_air_mass': self.ylm_map.nodal_grid}

  def __call__(
      self,
      diagnostic: dict[str, cx.Field],
      targets: dict[str, cx.Field],
      *args,
      **kwargs,
  ) -> dict[str, cx.Field]:
    """Computes surface pressure adjustment to conserve dry air mass."""
    column_dry_air_mass_t1 = self.extract_column_dry_air_mass(targets)[
        'column_dry_air_mass'
    ]
    column_dry_air_mass_t0 = diagnostic['column_dry_air_mass']

    # gamma = total_mass_t0 / total_mass_t1
    #       = (total_mass_t1 + delta_mass) / total_mass_t1
    #       = 1 + delta_mass / total_mass_t1
    # We compute delta_mass by integrating the difference of column masses,
    # to avoid precision issues.
    mass_diff = self.ylm_map.nodal_grid.integrate(
        column_dry_air_mass_t0 - column_dry_air_mass_t1, radius=1.0
    )
    total_mass_t1 = self.ylm_map.nodal_grid.integrate(
        column_dry_air_mass_t1, radius=1.0
    )
    gamma = 1.0 + mass_diff / total_mass_t1

    targets_adj = targets.copy()
    log_sp_nodal = self.ylm_map.to_nodal(
        targets['log_surface_pressure']
    )
    log_sp_adj_nodal = log_sp_nodal + cx.cmap(jnp.log)(gamma)
    targets_adj['log_surface_pressure'] = self.ylm_map.to_modal(
        log_sp_adj_nodal
    )
    return targets_adj
