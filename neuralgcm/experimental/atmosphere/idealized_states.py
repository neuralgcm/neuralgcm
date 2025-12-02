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
"""Functions that generate atmospheric test case states."""

from typing import Any, Literal

import coordax as cx
from dinosaur import coordinate_systems
from dinosaur import primitive_equations as dinosaur_primitive_equations
from dinosaur import primitive_equations_states
from dinosaur import spherical_harmonic as dino_spherical_harmonic
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np


def _dino_state_to_neuralgcm_dict(
    state: dinosaur_primitive_equations.State,
    ylm_map: spherical_harmonics.FixedYlmMapping,
    levels: coordinates.SigmaLevels | coordinates.HybridLevels,
    aux_features: dict[str, Any] | None,
    as_nodal: bool = False,
    temperature_format: Literal['absolute', 'variation'] = 'absolute',
) -> dict[str, cx.Field]:
  """Converts dinosaur state to neuralgcm state dict."""
  state_dict = state.asdict()
  state_dict |= state_dict.pop('tracers')
  if temperature_format == 'absolute':
    if not aux_features or 'ref_temperatures' not in aux_features:
      raise ValueError('ref_temperatures missing in aux_features.')
    ref_temperatures = aux_features['ref_temperatures']
    temperature = dino_spherical_harmonic.add_constant(
        state_dict.pop('temperature_variation'), ref_temperatures
    )
    state_dict['temperature'] = temperature
  elif temperature_format != 'variation':
    raise ValueError(f'Unknown temperature format: {temperature_format}')
  # Remove dummy dimension used in dinosaur codebase.
  state_dict['log_surface_pressure'] = state_dict['log_surface_pressure'][0]
  grid, ylm_grid = ylm_map.nodal_grid, ylm_map.modal_grid
  coords_map = {
      2: ylm_grid,  # for log_surface_pressure.
      3: cx.compose_coordinates(levels, ylm_grid),  # other fields.
  }
  state_dict = {
      k: cx.wrap(v, coords_map[v.ndim]) for k, v in state_dict.items()
  }
  if as_nodal:
    state_dict = transforms.ToNodal(ylm_map)(state_dict)
  if 'orography' in aux_features:
    state_dict['orography'] = cx.wrap(aux_features['orography'], grid)
  if 'ref_temperatures' in aux_features:
    ref_temperatures = cx.wrap(aux_features['ref_temperatures'], levels)
    state_dict['ref_temperatures'] = ref_temperatures
  if 'geopotential' in aux_features:
    geopotential = cx.wrap(aux_features['geopotential'], levels, grid)
    state_dict['geopotential'] = geopotential
  return state_dict


def to_si_units(
    inputs: dict[str, cx.Field], sim_units: units.SimUnits
) -> dict[str, cx.Field]:
  """Converts inputs to SI units."""
  inputs_to_units_mapping = {
      'surface_pressure': 'pascal',
      'log_surface_pressure': 'dimensionless',  # consistent when nondim.
      'divergence': '1 / second',
      'vorticity': '1 / second',
      'temperature': 'kelvin',
      'temperature_variation': 'kelvin',
      'orography': 'meter',
      'ref_temperatures': 'kelvin',
      'u_component_of_wind': 'meter / second',
      'v_component_of_wind': 'meter / second',
      'geopotential': 'm**2 s**-2',
      'specific_cloud_ice_water_content': 'dimensionless',
      'specific_cloud_liquid_water_content': 'dimensionless',
  }
  redim = transforms.Redimensionalize(sim_units, inputs_to_units_mapping)
  return redim(inputs)


def isothermal_rest_atmosphere(
    ylm_map: spherical_harmonics.FixedYlmMapping,
    levels: coordinates.SigmaLevels | coordinates.HybridLevels,
    rng: typing.PRNGKeyArray,
    sim_units: units.SimUnits = units.SI_UNITS,
    tref: typing.Quantity = 288.0 * typing.units.kelvin,
    p0: typing.Quantity = 1e5 * typing.units.pascal,
    p1: typing.Quantity = 0.0 * typing.units.pascal,
    surface_height: typing.Quantity | None = None,
    as_nodal: bool = False,
    in_si_units: bool = False,
    temperature_format: Literal['absolute', 'variation'] = 'absolute',
) -> dict[str, cx.Field]:
  """Returns initial state of isothermal atmosphere at rest."""
  if isinstance(levels, coordinates.SigmaLevels):
    vertical_coords = levels.sigma_levels
  elif isinstance(levels, coordinates.HybridLevels):
    vertical_coords = levels.hybrid_levels
  else:
    raise ValueError(f'Unsupported vertical coordinate system: {levels}')
  dinosaur_coords = coordinate_systems.CoordinateSystem(
      ylm_map.dinosaur_grid, vertical_coords
  )
  init_state_fn, aux = primitive_equations_states.isothermal_rest_atmosphere(
      coords=dinosaur_coords,
      physics_specs=sim_units,
      tref=tref,
      p0=p0,
      p1=p1,
      surface_height=surface_height,
      meter_quantity=typing.units.meter,
  )
  dino_state = init_state_fn(rng)
  state = _dino_state_to_neuralgcm_dict(
      dino_state, ylm_map, levels, aux, as_nodal, temperature_format
  )
  if in_si_units:
    state = to_si_units(state, sim_units)
  return state


def steady_state_jw(
    ylm_map: spherical_harmonics.FixedYlmMapping,
    levels: coordinates.SigmaLevels | coordinates.HybridLevels,
    rng: typing.PRNGKeyArray,
    sim_units: units.SimUnits = units.SI_UNITS,
    u0: typing.Quantity = 35.0 * typing.units.m / typing.units.s,
    p0: typing.Quantity = 1e5 * typing.units.pascal,
    t0: typing.Quantity = 288.0 * typing.units.kelvin,
    delta_t: typing.Quantity = 4.8e5 * typing.units.kelvin,
    gamma: typing.Quantity = 0.005 * typing.units.kelvin / typing.units.m,
    eta_tropo: float = 0.2,
    eta0: float = 0.252,
    as_nodal: bool = False,
    in_si_units: bool = False,
    temperature_format: Literal['absolute', 'variation'] = 'absolute',
) -> dict[str, cx.Field]:
  """Returns Jablonowski and Williamson steady state."""
  if isinstance(levels, coordinates.SigmaLevels):
    vertical_coords = levels.sigma_levels
  elif isinstance(levels, coordinates.HybridLevels):
    vertical_coords = levels.hybrid_levels
  else:
    raise ValueError(f'Unsupported vertical coordinate system: {levels}')
  dinosaur_coords = coordinate_systems.CoordinateSystem(
      ylm_map.dinosaur_grid, vertical_coords
  )
  init_state_fn, aux = primitive_equations_states.steady_state_jw(
      dinosaur_coords,  # coords
      sim_units,  # physics_specs
      u0,  # u0
      p0,  # p0
      t0,  # t0
      delta_t,  # delta_t
      gamma,  # gamma
      eta_tropo,  # eta_tropo
      eta0,  # eta0
      typing.units.hPa,  # hPa quantity
  )
  dino_state = init_state_fn(rng)
  state = _dino_state_to_neuralgcm_dict(
      dino_state, ylm_map, levels, aux, as_nodal, temperature_format
  )
  if in_si_units:
    state = to_si_units(state, sim_units)
  return state


def perturbed_jw(
    ylm_map: spherical_harmonics.FixedYlmMapping,
    levels: coordinates.SigmaLevels | coordinates.HybridLevels,
    rng: typing.PRNGKeyArray,
    sim_units: units.SimUnits = units.SI_UNITS,
    u0: typing.Quantity = 35.0 * typing.units.m / typing.units.s,
    p0: typing.Quantity = 1e5 * typing.units.pascal,
    t0: typing.Quantity = 288.0 * typing.units.kelvin,
    delta_t: typing.Quantity = 4.8e5 * typing.units.kelvin,
    gamma: typing.Quantity = 0.005 * typing.units.kelvin / typing.units.m,
    eta_tropo: float = 0.2,
    eta0: float = 0.252,
    u_perturb: typing.Quantity = 1.0 * typing.units.m / typing.units.s,
    lon_location: float = np.pi / 9,
    lat_location: float = 2 * np.pi / 9,
    perturbation_radius: float = 0.1,
    as_nodal: bool = False,
    in_si_units: bool = False,
    temperature_format: Literal['absolute', 'variation'] = 'absolute',
) -> dict[str, cx.Field]:
  """Returns Jablonowski and Williamson steady state with perturbation."""
  if isinstance(levels, coordinates.SigmaLevels):
    vertical_coords = levels.sigma_levels
  elif isinstance(levels, coordinates.HybridLevels):
    vertical_coords = levels.hybrid_levels
  else:
    raise ValueError(f'Unsupported vertical coordinate system: {levels}')
  dinosaur_coords = coordinate_systems.CoordinateSystem(
      ylm_map.dinosaur_grid, vertical_coords
  )
  init_state_fn, aux = primitive_equations_states.steady_state_jw(
      dinosaur_coords,  # coords
      sim_units,  # physics_specs
      u0,  # u0
      p0,  # p0
      t0,  # t0
      delta_t,  # delta_t
      gamma,  # gamma
      eta_tropo,  # eta_tropo
      eta0,  # eta0
      typing.units.hPa,  # hPa quantity
  )
  dino_steady_state = init_state_fn(rng)
  dino_perturbation = primitive_equations_states.baroclinic_perturbation_jw(
      dinosaur_coords,
      sim_units,
      u_perturb=u_perturb,
      lon_location=lon_location,
      lat_location=lat_location,
      perturbation_radius=perturbation_radius,
      hpa_quantity=typing.units.hPa,
  )
  dino_full_state = jax.tree.map(
      lambda x, y: x + y, dino_steady_state, dino_perturbation
  )
  state = _dino_state_to_neuralgcm_dict(
      dino_full_state,
      ylm_map,
      levels,
      aux,
      as_nodal,
      temperature_format,
  )
  if in_si_units:
    state = to_si_units(state, sim_units)
  return state
