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

"""Helpers for converting between different atmospheric states."""

import functools

import coordax as cx
from dinosaur import primitive_equations as dinosaur_primitive_equations
import jax.numpy as jnp
from neuralgcm.experimental.atmosphere import interpolators
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import units


def get_geopotential(
    inputs: dict[str, cx.Field],
    orography: orographies.Orography,
    sim_units: units.SimUnits,
):
  """Computes geopotential from temperature and moisture species."""
  temperature = inputs['temperature']
  specific_humidity = inputs['specific_humidity']
  clouds = None
  if (
      'specific_cloud_ice_water_content' in inputs
      and 'specific_cloud_liquid_water_content' in inputs
  ):
    clouds = (
        inputs['specific_cloud_ice_water_content'].data
        + inputs['specific_cloud_liquid_water_content'].data
    )
  canonical = cx.coords.canonicalize(temperature.coordinate)
  levels = [c for c in canonical if isinstance(c, coordinates.SigmaLevels)]
  if len(levels) != 1:
    raise ValueError(
        f'Expected exactly one sigma in {temperature.coordinate}, got {levels}'
    )
  [levels] = levels
  # TODO(dkochkov): Simplify and generalize this function in dinosaur, also
  # consider exposing this function elsewhere in the codebase.
  dino_get_geopotential = functools.partial(
      dinosaur_primitive_equations.get_geopotential_with_moisture,
      nodal_orography=orography.nodal_orography.data,
      coordinates=levels.sigma_levels,
      gravity_acceleration=sim_units.gravity_acceleration,
      ideal_gas_constant=sim_units.ideal_gas_constant,
      water_vapor_gas_constant=sim_units.water_vapor_gas_constant,
  )
  coord = temperature.coordinate
  geopotential = dino_get_geopotential(
      temperature=temperature.data,
      specific_humidity=specific_humidity.data,
      clouds=clouds,
  )
  geopotential = cx.field(geopotential, coord)
  return geopotential


def uvtz_to_primitive_equations(
    inputs: dict[str, cx.Field],
    levels: coordinates.SigmaLevels | coordinates.HybridLevels,
    orography: orographies.Orography,
    sim_units: units.SimUnits,
) -> dict[str, cx.Field]:
  """Converts velocity/temperature/geopotential to primitive equations state."""
  if 'geopotential' not in inputs and 'surface_pressure' not in inputs:
    raise ValueError(
        'Missing `geopotential` and `surface_pressure` in source data keys'
        f' {inputs.keys()}, at least one is needed to obtain surface pressure.'
    )

  inputs = inputs.copy()  # avoid mutating inputs.
  geopotential = inputs.pop('geopotential')
  input_levels = {
      geopotential.axes.get(k) for k in ['sigma', 'hybrid', 'pressure']
  }
  input_levels.discard(None)
  if len(input_levels) != 1:
    raise ValueError('expected only one type of level type, got {levels}.')
  [input_levels] = input_levels
  if isinstance(
      input_levels, (coordinates.SigmaLevels, coordinates.HybridLevels)
  ):
    surface_pressure = inputs.pop('surface_pressure')
  else:
    geopotential_at_surface = orography.nodal_orography * sim_units.g
    surface_pressure = interpolators.get_surface_pressure(
        geopotential, geopotential_at_surface, sim_units
    )
  regrid = interpolators.LinearOnPressure(levels, sim_units=sim_units)
  on_levels = regrid(inputs | {'surface_pressure': surface_pressure})
  on_levels['log_surface_pressure'] = cx.cpmap(jnp.log)(surface_pressure)
  return on_levels


def primitive_equations_to_uvtz(
    inputs: dict[str, cx.Field],
    ylm_map: spherical_harmonics.FixedYlmMapping,
    levels: coordinates.PressureLevels | coordinates.SigmaLevels,
    orography: orographies.Orography,
    sim_units: units.SimUnits,
    include_surface_pressure: bool = False,
) -> dict[str, cx.Field]:
  """Converts primitive equations state to pressure level representation.

  This function transforms an atmospheric state described in terms of
  temperature variation, divergence, vorticity, surface pressure and tracers
  on sigma levels to wind components, temperature, geopotential and tracers
  on fixed pressure-level coordinates.

  Args:
    inputs: State represented using primitive equations variables.
    ylm_map: Spherical harmonics mapping that defines modal-nodal conversion.
    levels: Vertical levels to interpolate "uvtz, ..." variables to.
    orography: Orography module.
    sim_units: Simulation units object.
    include_surface_pressure: Whether to include surface pressure in the output.

  Returns:
    State represented as zonal, medidional wind, temperature, geopotential and
    tracers interpolated to vertical `levels`.
  """
  inputs = inputs.copy()  # avoid mutating inputs.
  vorticity, divergence = inputs.pop('vorticity'), inputs.pop('divergence')
  u, v = spherical_harmonics.vor_div_to_uv_nodal(vorticity, divergence, ylm_map)
  log_surface_pressure = inputs.pop('log_surface_pressure')
  surface_pressure = cx.cpmap(jnp.exp)(ylm_map.to_nodal(log_surface_pressure))

  nodal_inputs = ylm_map.to_nodal(inputs)  # includes temperature and tracers.
  geopotential = get_geopotential(nodal_inputs, orography, sim_units)
  temperature = nodal_inputs.pop('temperature')

  surface_pressure, temperature, geopotential, nodal_inputs = (
      ylm_map.mesh.with_sharding_constraint(
          (surface_pressure, temperature, geopotential, nodal_inputs),
          ('physics', 'dycore'),
      )
  )

  regrid_constant = interpolators.LinearOnPressure(
      levels, 'constant', sim_units=sim_units
  )
  regrid_linear = interpolators.LinearOnPressure(
      levels, 'linear', sim_units=sim_units
  )
  # closest regridding options to those used in ERA5.
  # use constant extrapolation for `u, v, tracers`.
  # use linear extrapolation for `z, t`.
  # google reference: http://shortn/_X09ZAU1jsx.
  ps_dict = {'surface_pressure': surface_pressure}
  winds = {'u_component_of_wind': u, 'v_component_of_wind': v}
  outputs = regrid_constant(winds | nodal_inputs | ps_dict)
  outputs |= regrid_linear(
      {'temperature': temperature, 'geopotential': geopotential} | ps_dict
  )
  if include_surface_pressure:
    outputs['surface_pressure'] = surface_pressure
  return outputs
