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

"""Module-based API for calculating diagnostics of NeuralGCM models."""

import dataclasses
from typing import Literal, Protocol

import coordax as cx
from flax import nnx
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units


class EnergyBalanceModule(Protocol):
  """Protocol for energy balance modules that adjust tendencies."""

  def __call__(
      self,
      imbalance: cx.Field,
      tendencies: dict[str, cx.Field],
      *args,
      **kwargs,
  ) -> dict[str, cx.Field]:
    """Adjusts tendencies based on energy imbalance."""


@nnx_compat.dataclass
class ExtractPrecipitationPlusEvaporation(nnx.Module):
  """Diagnoses precipitation plus evaporation rate from physics tendencies.

  The computation of P + E is based on the integration of non-dynamical moisture
  tendency over the vertical column. We define precipitation and evaporation
  rates as the rate of change of non-atmospheric moisture, i.e. resulting in
  positive values for precipitation and negative values for evaporation. This
  is in line with how these quantities are often defined in datasets like ERA5
  or IMERG. This is also in line with the convention of having "downward"
  fluxes as positive and "upward" fluxes as negative.
  """

  ylm_map: spherical_harmonics.FixedYlmMapping
  levels: coordinates.SigmaLevels | coordinates.HybridLevels
  sim_units: units.SimUnits
  moisture_species: tuple[str, ...] = (
      'specific_humidity',
      'specific_cloud_ice_water_content',
      'specific_cloud_liquid_water_content',
  )
  prognostics_arg_key: str | int = 'prognostics'

  def _compute_p_plus_e_rate(
      self,
      tendencies: dict[str, cx.Field],
      prognostics: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    to_nodal = self.ylm_map.to_nodal
    p_surface = cx.cmap(jnp.exp)(to_nodal(prognostics['log_surface_pressure']))
    scale = 1 / self.sim_units.gravity_acceleration
    moisture_tendencies_nodal = [
        to_nodal(v) for k, v in tendencies.items() if k in self.moisture_species
    ]
    moisture_tendencies_sum = sum(moisture_tendencies_nodal)
    assert isinstance(moisture_tendencies_sum, cx.Field)
    p_plus_e = -scale * self.levels.integrate_over_pressure(
        moisture_tendencies_sum,
        p_surface,
        self.sim_units,
    )
    return p_plus_e

  def __call__(self, inputs, *args, **kwargs) -> dict[str, cx.Field]:
    tendencies = inputs
    if isinstance(self.prognostics_arg_key, int):
      prognostics = args[self.prognostics_arg_key]
    else:
      prognostics = kwargs.get(self.prognostics_arg_key)
    p_plus_e_rate = self._compute_p_plus_e_rate(tendencies, prognostics)
    return {'precipitation_plus_evaporation_rate': p_plus_e_rate}


PrecipitationScales = Literal['rate', 'cumulative', 'mass_rate']


@nnx_compat.dataclass
class ExtractPrecipitationAndEvaporation(nnx.Module):
  """Extracts balanced precipitation and evaporation values.

  This module can be attached in diagnostics that have access to both
  parameterization tendencies and model state to infer balanced precipitation
  and evaporation. We use `observation_operator` to predict on of the
  two (either `precipitation` or `evaporation`) and infer the other from the
  precipitation_plus_evaporation calculation. The mode is defined by the
  provided operator, query and inference variable indicating which variable
  will be computed from the balance equations.

  Attributes:
    observation_operator: Observation operator used to predict one of the two
      variables from the balance equations.
    operator_query: Query used for the observation operator.
    extract_p_plus_e: Module that extracts precipitation plus evaporation from
      tendencies and prognostics.
    prognostics_arg_key: Key or index of the prognostics argument in the call
      signature.
    precipitation_scaling: Scaling strategy for the precipitation field. Must be
      one of `rate`, `mass_rate` or `cumulative`. If using `cumulative` scaling,
      `dt` must be set.
    evaporation_scaling: Scaling strategy for the evaporation field. Must be one
      of `rate`, `mass_rate` or `cumulative`. If using `cumulative` scaling,
      `dt` must be set.
    dt: Timestep by which the precipitation is scaled (only used when
      `precipitation_scaling` is set to `cumulative`).
    sim_units: Object defining nondimensionalization and physical constants.
    precipitation_key: Key under which the precipitation field is stored in the
      output.
    evaporation_key: Key under which the evaporation field is stored in the
      output.
  """

  observation_operator: observation_operators.ObservationOperatorABC
  operator_query: dict[str, cx.Coordinate]
  extract_p_plus_e: ExtractPrecipitationPlusEvaporation
  prognostics_arg_key: str | int = 'prognostics'
  precipitation_scaling: PrecipitationScales = 'rate'
  evaporation_scaling: PrecipitationScales = 'rate'
  dt: float | None = None
  precipitation_key: str = 'precipitation'
  evaporation_key: str = 'evaporation'
  sim_units: units.SimUnits = dataclasses.field(kw_only=True)

  def __post_init__(self):
    valid_keys = set([self.precipitation_key, self.evaporation_key])
    query_keys = set(self.operator_query.keys())
    if len(query_keys.intersection(valid_keys)) != 1:
      raise ValueError(
          f'{self.operator_query=} should contain exactly on of {valid_keys=}.'
      )
    [self.observe_key] = valid_keys.intersection(query_keys)
    [self.diagnosed_key] = valid_keys.difference(query_keys)

  def _extract_prognostics(self, *args, **kwargs):
    if isinstance(self.prognostics_arg_key, int):
      prognostics = args[self.prognostics_arg_key]
    else:
      prognostics = kwargs.get(self.prognostics_arg_key)
    if not isinstance(prognostics, dict):
      raise ValueError(
          f'Prognostics must be a dictionary, got {type(prognostics)=} instead.'
      )
    return prognostics

  def _apply_scaling(self, precipitation_and_evaporation):
    water_density = self.sim_units.water_density
    for key, scaling in zip(
        [self.precipitation_key, self.evaporation_key],
        [self.precipitation_scaling, self.evaporation_scaling],
    ):
      if scaling == 'cumulative':
        if self.dt is None:
          raise ValueError(
              'dt must be provided when using cumulative precipitation scaling.'
          )
        precipitation_and_evaporation[key] *= self.dt / water_density
      elif scaling == 'rate':
        precipitation_and_evaporation[key] *= 1 / water_density
      elif scaling == 'mass_rate':
        continue
      else:
        raise ValueError(
            f'{scaling=} should be one of rate, mass_rate or cumulative.'
        )
    return precipitation_and_evaporation

  def __call__(self, result, *args, **kwargs):
    tendencies = result
    [p_plus_e] = self.extract_p_plus_e(tendencies, *args, **kwargs).values()
    prognostics = self._extract_prognostics(*args, **kwargs)
    observation = self.observation_operator.observe(
        prognostics, query=self.operator_query
    )
    observation = observation[self.observe_key]
    precipitation_and_evaporation = {
        self.diagnosed_key: p_plus_e - observation,
        self.observe_key: observation,
    }
    return self._apply_scaling(precipitation_and_evaporation)


@nnx_compat.dataclass
class ExtractPrecipitationAndEvaporationWithConstraints(
    ExtractPrecipitationAndEvaporation
):
  """Extracts balanced precipitation and evaporation values.

  This module can be attached in diagnostics that have access to both
  parameterization tendencies and model state to infer balanced precipitation
  and evaporation. We use `observation_operator` to predict on of the
  two (either `precipitation` or `evaporation`) and infer the other from the
  precipitation_plus_evaporation calculation. The mode is defined by the
  provided operator, query and inference variable indicating which variable
  will be computed from the balance equations. Evaporation is constrained to
  be non-positive and precipitation is constrained to be non-negative.
  """

  def __call__(self, result, *args, **kwargs):
    tendencies = result
    [p_plus_e] = self.extract_p_plus_e(tendencies, *args, **kwargs).values()
    prognostics = self._extract_prognostics(*args, **kwargs)
    observation = self.observation_operator.observe(
        prognostics, query=self.operator_query
    )
    observation = observation[self.observe_key]
    if self.observe_key == self.precipitation_key:
      constrained_observation = cx.cmap(
          lambda x, a, b: jnp.maximum(x, jnp.maximum(a, b))
      )(observation, p_plus_e, 0)
      precipitation_and_evaporation = {
          self.observe_key: constrained_observation,
          self.diagnosed_key: p_plus_e - constrained_observation,
      }
    elif self.observe_key == self.evaporation_key:
      constrained_observation = cx.cmap(
          lambda x, a, b: jnp.minimum(x, jnp.minimum(a, b))
      )(observation, p_plus_e, 0)
      precipitation_and_evaporation = {
          self.observe_key: constrained_observation,
          self.diagnosed_key: p_plus_e - constrained_observation,
      }
    else:
      raise ValueError(
          f'{self.observe_key=} should be either {self.precipitation_key=} or'
          f' {self.evaporation_key=}.'
      )
    return self._apply_scaling(precipitation_and_evaporation)


@nnx_compat.dataclass
class ExtractColumnDryAirMass(nnx.Module):
  """Extracts column dry air mass from prognostics."""

  ylm_map: spherical_harmonics.FixedYlmMapping
  levels: coordinates.SigmaLevels | coordinates.HybridLevels
  sim_units: units.SimUnits
  moisture_species: tuple[str, ...] = (
      'specific_humidity',
      'specific_cloud_ice_water_content',
      'specific_cloud_liquid_water_content',
  )

  def __call__(
      self, prognostics: dict[str, cx.Field], *args, **kwargs
  ) -> dict[str, cx.Field]:
    """Computes column dry air mass."""
    del args, kwargs  # Unused.
    to_nodal = self.ylm_map.to_nodal
    p_surface_field = cx.cmap(jnp.exp)(
        to_nodal(prognostics['log_surface_pressure'])
    )
    g = self.sim_units.gravity_acceleration

    missing_keys = [k for k in self.moisture_species if k not in prognostics]
    if missing_keys:
      raise KeyError(
          f'Moisture species {missing_keys} not found in prognostics.'
      )
    q_fields = [to_nodal(prognostics[k]) for k in self.moisture_species]
    if q_fields:
      q_total = sum(q_fields)
    else:
      q_total = 0.0
    assert isinstance(q_total, (cx.Field, float, int))

    column_dry_air_mass = (1 / g) * self.levels.integrate_over_pressure(
        1.0 - q_total, p_surface_field, self.sim_units
    )
    return {'column_dry_air_mass': column_dry_air_mass}


@nnx_compat.dataclass
class ExtractEnergyResiduals(nnx.Module):
  """Computes column energy imbalance based on moist enthalpy formulation.

  This module calculates the imbalance between surface and TOA fluxes (RT - FS)
  and the column energy tendency due to parameterizations dE/dt|_NN based on
  E = phi_s*p_s/g + p_s/g * integral(Cp*T + Lv*q - Lf*qi + k)dsigma
  (See Durran's book section 8.6.4 where it is shown that this is equivalent to
  E = p_s/g * integral(Cv*T + Lv*q + phi - Lf*qi + k)dsigma
  albeit it does not have moisture species there):
  The tendency dE/dt|_NN is computed as:
  dE/dt|_NN = p_s/g * [
      (phi_s + integral(Cp*T + Lv*q - Lf*qi + k)dsigma) * d(log p_s)/dt|_NN +
      integral(Cp*dT/dt|_NN + Lv*dq/dt|_NN - Lf*dqi/dt|_NN + dk/dt|_NN)dsigma
  ]
  The module returns the imbalance: (RT - FS) - dE/dt|_NN
  where RT and FS are TOA and surface fluxes obtained from
  observation_operator.
  If use_evaporation_for_latent_heat is True, FS uses latent heat flux
  derived from mean_evaporation_rate (by multiplying by Lv which is inaccurate
  in ice covered regions), otherwise it uses surface_latent_heat_flux
  from net_energy_terms.
  If use_liquid_ice_moist_static_energy is True, qi is included in the
  column energy integral and we need to predict also snowfall to close budget,
  otherwise it is excluded.
  """

  ylm_map: spherical_harmonics.FixedYlmMapping
  levels: coordinates.SigmaLevels | coordinates.HybridLevels
  sim_units: units.SimUnits
  model_orography: orographies.ModalOrography
  observation_operator: observation_operators.ObservationOperatorABC
  energy_fluxes_query: dict[str, cx.Coordinate]
  prognostics_arg_key: str | int = 'prognostics'
  use_evaporation_for_latent_heat: bool = False
  use_liquid_ice_moist_static_energy: bool = False

  def __post_init__(self):
    self.rt_keys = ['top_net_thermal_radiation', 'top_net_solar_radiation']
    self.fs_keys = [
        'surface_sensible_heat_flux',
        'surface_net_solar_radiation',
        'surface_net_thermal_radiation',
    ]
    if self.use_evaporation_for_latent_heat:
      required_keys = self.rt_keys + self.fs_keys + ['mean_evaporation_rate']
    else:
      required_keys = self.rt_keys + self.fs_keys + ['surface_latent_heat_flux']

    if self.use_liquid_ice_moist_static_energy:
      required_keys.append('snowfall')
    missing_keys = [
        k for k in required_keys if k not in self.energy_fluxes_query
    ]
    if missing_keys:
      raise ValueError(
          f'Missing energy terms in energy_fluxes_query: {missing_keys}'
      )

  def _compute_ke_and_tendency(
      self,
      tendencies: dict[str, cx.Field],
      prognostics: dict[str, cx.Field],
  ) -> tuple[cx.Field, cx.Field]:
    """Computes nodal kinetic energy and its tendency."""
    velocity_from_div_curl = transforms.VelocityFromModalDivCurl(
        self.ylm_map
    )
    winds = velocity_from_div_curl({
        'vorticity': prognostics['vorticity'],
        'divergence': prognostics['divergence'],
    })
    u_nodal = winds['u_component_of_wind']
    v_nodal = winds['v_component_of_wind']
    k_nodal = 0.5 * (u_nodal**2 + v_nodal**2)
    wind_tends = velocity_from_div_curl({
        'vorticity': tendencies['vorticity'],
        'divergence': tendencies['divergence'],
    })
    du_dt_nodal = wind_tends['u_component_of_wind']
    dv_dt_nodal = wind_tends['v_component_of_wind']
    dk_dt_nodal = u_nodal * du_dt_nodal + v_nodal * dv_dt_nodal
    return k_nodal, dk_dt_nodal

  def _compute_vertically_integrated_tendency(
      self,
      tendencies: dict[str, cx.Field],
      prognostics: dict[str, cx.Field],
  ) -> cx.Field:
    """Computes column energy tendency due to parameterization."""
    to_nodal = self.ylm_map.to_nodal
    p_surface_field = cx.cmap(jnp.exp)(
        to_nodal(prognostics['log_surface_pressure'])
    )
    cp = self.sim_units.Cp
    lv = self.sim_units.Lv
    g = self.sim_units.gravity_acceleration
    lf = self.sim_units.Lf

    t_nodal_field = to_nodal(prognostics['temperature'])
    q_nodal_field = to_nodal(prognostics['specific_humidity'])

    if self.use_liquid_ice_moist_static_energy:
      qi_nodal_field = to_nodal(prognostics['specific_cloud_ice_water_content'])
      dqi_dt_nodal_field = to_nodal(
          tendencies['specific_cloud_ice_water_content']
      )
    else:
      qi_nodal_field = 0.0
      dqi_dt_nodal_field = 0.0

    k_nodal_field, dk_dt_nodal_field = self._compute_ke_and_tendency(
        tendencies, prognostics
    )
    temp_tend_nodal_field = to_nodal(tendencies['temperature'])
    hum_tend_nodal_field = to_nodal(tendencies['specific_humidity'])

    phi_s = self.model_orography.nodal_orography * g
    log_sp_tend = tendencies.get('log_surface_pressure')
    if log_sp_tend is not None:
      log_sp_tend_nodal_field = to_nodal(log_sp_tend)
    else:
      log_sp_tend_nodal_field = p_surface_field * 0

    integrand1 = (
        cp * t_nodal_field
        + lv * q_nodal_field
        - lf * qi_nodal_field
        + k_nodal_field
    )
    i1 = self.levels.integrate_over_pressure(
        integrand1, p_surface_field, self.sim_units
    )
    integrand2 = (
        cp * temp_tend_nodal_field
        + lv * hum_tend_nodal_field
        - lf * dqi_dt_nodal_field
        + dk_dt_nodal_field
    )
    i2 = self.levels.integrate_over_pressure(
        integrand2, p_surface_field, self.sim_units
    )

    energy_tendency_data = (1 / g) * (
        (p_surface_field * phi_s + i1) * log_sp_tend_nodal_field + i2
    )

    return energy_tendency_data

  def __call__(
      self,
      inputs: dict[str, cx.Field],
      *args,
      **kwargs,
  ) -> dict[str, cx.Field]:
    """Computes temperature tendency adjustment to conserve energy."""
    tendencies = inputs
    if isinstance(self.prognostics_arg_key, int):
      prognostics = args[self.prognostics_arg_key]
    else:
      prognostics = kwargs.get(self.prognostics_arg_key)
    if not isinstance(prognostics, dict):
      raise ValueError(
          f'Prognostics must be a dictionary, got {type(prognostics)=} instead.'
      )

    e_tendency_nn = self._compute_vertically_integrated_tendency(
        tendencies, prognostics
    )

    net_energy_terms = self.observation_operator.observe(
        prognostics, query=self.energy_fluxes_query
    )

    # Assuming observation_operator returns RT and FS fluxes in J/m^2
    # accumulated over an hour time and need to be converted to W/m^2.
    # RT is TOA flux into atm, and FS is surface flux from atm.
    # The user-provided formula is dE/dt = RT - FS.
    sec_in_hour_inv = 1 / self.sim_units.nondimensionalize(
        3600 * typing.units.seconds
    )
    rt = sum(net_energy_terms[k] for k in self.rt_keys) * sec_in_hour_inv
    fs = sum(net_energy_terms[k] for k in self.fs_keys) * sec_in_hour_inv

    if self.use_evaporation_for_latent_heat:
      # mean_evaporation_rate is mass rate per second, in SI: (kg/m^2/s).
      # Multiplying by Lv gives nondim equivalent of W/m^2.
      fs += (
          net_energy_terms['mean_evaporation_rate']
          * self.sim_units.Lv
      )
    else:
      fs += net_energy_terms['surface_latent_heat_flux'] * sec_in_hour_inv
    if self.use_liquid_ice_moist_static_energy:
      # snowfall is in m of water equivalent (accumulated), so we multiply by
      # density and Lf to get J/m^2 and then divide by time to get W/m^2.
      fs += (
          net_energy_terms['snowfall']
          * self.sim_units.water_density
          * self.sim_units.Lf
          * sec_in_hour_inv
      )

    # Energy imbalance: difference between required tendency (rt - fs) and
    # tendency from NN (e_tendency_nn).
    imbalance = (rt - fs) - e_tendency_nn
    return {'imbalance': imbalance}


@nnx_compat.dataclass
class ExtractColumnEnergyBudget(nnx.Module):
  """Computes column energy and adds TOA and surface fluxes.

  This module calculates column energy based on
  E_col = phi_s*p_s/g + p_s/g * integral(Cp*T + Lv*q - Lf*qi + k)dsigma
  and adds TOA and surface fluxes (RT - FS) obtained from
  observation_operator to compute budget = E_col + RT - FS, representing
  the column energy based on fluxes.
  The result is integrated horizontally to obtain a global energy budget:
  E = horizontal_integral(E_col + RT - FS).
  If use_evaporation_for_latent_heat is True, FS uses latent heat flux
  derived from mean_evaporation_rate (by multiplying by Lv which is inaccurate
  in ice covered regions), otherwise it uses surface_latent_heat_flux
  from net_energy_terms.
  If use_liquid_ice_moist_static_energy is True, qi is included in the
  column energy integral and we need to predict also snowfall to close budget,
  otherwise it is excluded.
  """

  ylm_map: spherical_harmonics.FixedYlmMapping
  levels: coordinates.SigmaLevels | coordinates.HybridLevels
  sim_units: units.SimUnits
  model_orography: orographies.ModalOrography
  observation_operator: observation_operators.ObservationOperatorABC | None = (
      None
  )
  energy_fluxes_query: dict[str, cx.Coordinate] | None = None
  dt: float | None = None
  use_evaporation_for_latent_heat: bool = False
  use_liquid_ice_moist_static_energy: bool = False
  prognostics_arg_key: str | int | None = None

  def __post_init__(self):
    if self.energy_fluxes_query is not None:
      self.rt_keys = ['top_net_thermal_radiation', 'top_net_solar_radiation']
      self.fs_keys = [
          'surface_sensible_heat_flux',
          'surface_net_solar_radiation',
          'surface_net_thermal_radiation',
      ]
      if self.use_evaporation_for_latent_heat:
        required_keys = self.rt_keys + self.fs_keys + ['mean_evaporation_rate']
      else:
        required_keys = (
            self.rt_keys + self.fs_keys + ['surface_latent_heat_flux']
        )

      if self.use_liquid_ice_moist_static_energy:
        required_keys.append('snowfall')
      missing_keys = [
          k for k in required_keys if k not in self.energy_fluxes_query
      ]
      if missing_keys:
        raise ValueError(
            f'Missing energy terms in energy_fluxes_query: {missing_keys}'
        )

  def _compute_column_energy(
      self, prognostics: dict[str, cx.Field]
  ) -> cx.Field:
    to_nodal = self.ylm_map.to_nodal
    p_surface_field = cx.cmap(jnp.exp)(
        to_nodal(prognostics['log_surface_pressure'])
    )
    cp = self.sim_units.Cp
    lv = self.sim_units.Lv
    g = self.sim_units.gravity_acceleration
    lf = self.sim_units.Lf

    t_nodal_field = to_nodal(prognostics['temperature'])
    q_nodal_field = to_nodal(prognostics['specific_humidity'])

    if self.use_liquid_ice_moist_static_energy:
      qi_nodal_field = to_nodal(
          prognostics['specific_cloud_ice_water_content']
      )
    else:
      qi_nodal_field = 0.0

    velocity_from_div_curl = transforms.VelocityFromModalDivCurl(self.ylm_map)
    winds = velocity_from_div_curl({
        'vorticity': prognostics['vorticity'],
        'divergence': prognostics['divergence'],
    })
    u_nodal = winds['u_component_of_wind']
    v_nodal = winds['v_component_of_wind']
    k_nodal_field = 0.5 * (u_nodal**2 + v_nodal**2)

    phi_s = self.model_orography.nodal_orography * g

    integrand = (
        cp * t_nodal_field
        + lv * q_nodal_field
        - lf * qi_nodal_field
        + k_nodal_field
    )
    vert_integrated_energy = self.levels.integrate_over_pressure(
        integrand, p_surface_field, self.sim_units
    )
    column_energy = (1 / g) * (p_surface_field * phi_s + vert_integrated_energy)
    return column_energy

  def __call__(
      self,
      inputs: dict[str, cx.Field],
      *args,
      **kwargs,
  ) -> dict[str, cx.Field]:
    """Computes total energy budget."""
    if self.prognostics_arg_key is None:
      prognostics = inputs
    elif isinstance(self.prognostics_arg_key, int):
      prognostics = args[self.prognostics_arg_key]
    else:
      prognostics = kwargs[self.prognostics_arg_key]

    if not isinstance(prognostics, dict):
      raise ValueError(
          f'Prognostics must be a dictionary, got {type(prognostics)=} instead.'
      )

    column_energy = self._compute_column_energy(prognostics)
    results = {'column_energy': column_energy}

    if self.observation_operator is not None:
      if self.energy_fluxes_query is None:
        raise ValueError(
            'energy_fluxes_query must be provided if observation_operator is'
            ' provided.'
        )
      net_energy_terms = self.observation_operator.observe(
          prognostics, query=self.energy_fluxes_query
      )

      # Assuming observation_operator returns RT and FS fluxes in J/m^2
      # accumulated over an hour time.
      rt = sum(net_energy_terms[k] for k in self.rt_keys)
      fs = sum(net_energy_terms[k] for k in self.fs_keys)

      if self.use_evaporation_for_latent_heat:
        # mean_evaporation_rate is a rate (kg/m^2/s), so we multiply by Lv and
        # dt to get J/m^2.
        fs += (
            net_energy_terms['mean_evaporation_rate']
            * self.sim_units.Lv
            * self.dt
        )
      else:
        fs += net_energy_terms['surface_latent_heat_flux']
      if self.use_liquid_ice_moist_static_energy:
        # snowfall is in m of water equivalent (accumulated), so we multiply by
        # density and Lf to get J/m^2.
        fs += (
            net_energy_terms['snowfall']
            * self.sim_units.water_density
            * self.sim_units.Lf
        )
      column_budget = column_energy + rt - fs
      results['column_energy_budget'] = column_budget
    return results
