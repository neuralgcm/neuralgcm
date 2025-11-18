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
from dinosaur import scales as dino_scales
from dinosaur import spherical_harmonic as dino_spherical_harmonic
from dinosaur import typing as dino_typing
import jax
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np


def _as_dino_qty(q: typing.Quantity) -> dino_typing.Quantity:
  """Returns a dinosaur Quantity from a neuralgcm Quantity."""
  # We perform round trip through str representation to convert quantities
  # between two unit registries to avoid incompatibility errors.
  return dino_typing.Quantity(str(q))


def _get_dino_specs(
    sim_units: units.SimUnits,
) -> dinosaur_primitive_equations.PrimitiveEquationsSpecs:
  """Returns dinosaur PrimitiveEquationsSpecs from sim_units."""
  ngcm_scales_dict = sim_units.scale._scales  # pylint: disable=protected-access
  dino_scales_dict = {k: _as_dino_qty(v) for k, v in ngcm_scales_dict.items()}
  dino_scale = dino_scales.Scale(*dino_scales_dict.values())
  return dinosaur_primitive_equations.PrimitiveEquationsSpecs.from_si(
      radius_si=_as_dino_qty(units.scales.RADIUS),
      angular_velocity_si=_as_dino_qty(units.scales.ANGULAR_VELOCITY),
      gravity_acceleration_si=_as_dino_qty(units.scales.GRAVITY_ACCELERATION),
      ideal_gas_constant_si=_as_dino_qty(units.scales.IDEAL_GAS_CONSTANT),
      water_vapor_gas_constant_si=_as_dino_qty(
          units.scales.IDEAL_GAS_CONSTANT_H20
      ),
      water_vapor_isobaric_heat_capacity_si=_as_dino_qty(
          units.scales.WATER_VAPOR_CP
      ),
      kappa_si=_as_dino_qty(units.scales.KAPPA),
      scale=dino_scale,
  )


def _dino_state_to_neuralgcm_dict(
    state: dinosaur_primitive_equations.State,
    ylm_map: spherical_harmonics.FixedYlmMapping,
    sigma: coordinates.SigmaLevels,
    aux_features: dict[str, Any] | None,
    as_nodal: bool = False,
    temperature_format: Literal['absolute', 'variation'] = 'variation',
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
  coords_map = {
      2: ylm_map.modal_grid,  # for log_surface_pressure.
      3: cx.compose_coordinates(sigma, ylm_map.modal_grid),  # other fields.
  }
  state_dict = {
      k: cx.wrap(v, coords_map[v.ndim]) for k, v in state_dict.items()
  }
  if as_nodal:
    return transforms.ToNodal(ylm_map)(state_dict)
  return state_dict


def isothermal_rest_atmosphere(
    ylm_map: spherical_harmonics.FixedYlmMapping,
    sigma: coordinates.SigmaLevels,
    rng: typing.PRNGKeyArray,
    sim_units: units.SimUnits = units.SI_UNITS,
    tref: typing.Quantity = 288.0 * typing.units.kelvin,
    p0: typing.Quantity = 1e5 * typing.units.pascal,
    p1: typing.Quantity = 0.0 * typing.units.pascal,
    surface_height: typing.Quantity | None = None,
    as_nodal: bool = False,
    temperature_format: Literal['absolute', 'variation'] = 'variation',
) -> dict[str, cx.Field]:
  """Returns initial state of isothermal atmosphere at rest."""
  dinosaur_coords = coordinate_systems.CoordinateSystem(
      ylm_map.dinosaur_grid, sigma.sigma_levels
  )
  dino_specs = _get_dino_specs(sim_units)
  init_state_fn, aux = primitive_equations_states.isothermal_rest_atmosphere(
      coords=dinosaur_coords,
      physics_specs=dino_specs,
      tref=_as_dino_qty(tref),
      p0=_as_dino_qty(p0),
      p1=_as_dino_qty(p1),
      surface_height=_as_dino_qty(surface_height) if surface_height else None,
  )
  dino_state = init_state_fn(rng)
  return _dino_state_to_neuralgcm_dict(
      dino_state, ylm_map, sigma, aux, as_nodal, temperature_format
  )


def steady_state_jw(
    ylm_map: spherical_harmonics.FixedYlmMapping,
    sigma: coordinates.SigmaLevels,
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
    temperature_format: Literal['absolute', 'variation'] = 'variation',
) -> dict[str, cx.Field]:
  """Returns Jablonowski and Williamson steady state."""
  dinosaur_coords = coordinate_systems.CoordinateSystem(
      ylm_map.dinosaur_grid, sigma.sigma_levels
  )
  dino_specs = _get_dino_specs(sim_units)
  init_state_fn, aux = primitive_equations_states.steady_state_jw(
      dinosaur_coords,  # coords
      dino_specs,  # physics_specs
      _as_dino_qty(u0),  # u0
      _as_dino_qty(p0),  # p0
      _as_dino_qty(t0),  # t0
      _as_dino_qty(delta_t),  # delta_t
      _as_dino_qty(gamma),  # gamma
      eta_tropo,  # eta_tropo
      eta0,  # eta0
  )
  dino_state = init_state_fn(rng)
  return _dino_state_to_neuralgcm_dict(
      dino_state, ylm_map, sigma, aux, as_nodal, temperature_format
  )


def perturbed_jw(
    ylm_map: spherical_harmonics.FixedYlmMapping,
    sigma: coordinates.SigmaLevels,
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
    temperature_format: Literal['absolute', 'variation'] = 'variation',
) -> dict[str, cx.Field]:
  """Returns Jablonowski and Williamson steady state with perturbation."""
  dinosaur_coords = coordinate_systems.CoordinateSystem(
      ylm_map.dinosaur_grid, sigma.sigma_levels
  )
  dino_specs = _get_dino_specs(sim_units)
  init_state_fn, aux = primitive_equations_states.steady_state_jw(
      dinosaur_coords,  # coords
      dino_specs,  # physics_specs
      _as_dino_qty(u0),  # u0
      _as_dino_qty(p0),  # p0
      _as_dino_qty(t0),  # t0
      _as_dino_qty(delta_t),  # delta_t
      _as_dino_qty(gamma),  # gamma
      eta_tropo,  # eta_tropo
      eta0,  # eta0
  )
  dino_steady_state = init_state_fn(rng)
  dino_perturbation = primitive_equations_states.baroclinic_perturbation_jw(
      dinosaur_coords,
      dino_specs,
      u_perturb=_as_dino_qty(u_perturb),
      lon_location=lon_location,
      lat_location=lat_location,
      perturbation_radius=perturbation_radius,
  )
  dino_full_state = jax.tree.map(
      lambda x, y: x + y, dino_steady_state, dino_perturbation
  )
  return _dino_state_to_neuralgcm_dict(
      dino_full_state,
      ylm_map,
      sigma,
      aux,
      as_nodal,
      temperature_format,
  )
