# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for converting between different state representations."""

import functools

from dinosaur import primitive_equations as dinosaur_primitive_equations
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
from dinosaur import xarray_utils as dinosaur_xarray_utils
import jax.numpy as jnp

from neuralgcm.experimental import coordinates
from neuralgcm.experimental import equations
from neuralgcm.experimental import orographies
from neuralgcm.experimental import typing
from neuralgcm.experimental import units


def uvtz_to_primitive_equations(
    source_data: dict[str, typing.Array],
    source_coords: coordinates.DinosaurCoordinates,
    primitive_equations: equations.PrimitiveEquations,
    orography,
    sim_units: units.SimUnits,
):
  """Converts velocity/temperature/geopotential to primitive equations state."""
  surface_pressure = vertical_interpolation.get_surface_pressure(
      source_coords.vertical,
      source_data['geopotential'],
      orography.nodal_orography,
      sim_units.g,
  )
  interpolate_fn = vertical_interpolation.vectorize_vertical_interpolation(
      vertical_interpolation.vertical_interpolation
  )
  vertical = primitive_equations.coords.vertical
  assert isinstance(vertical, coordinates.SigmaLevels)
  # TODO(dkochkov): Consider having explicit args that used differently, eg z.
  data_on_sigma = vertical_interpolation.interp_pressure_to_sigma(
      {k: v for k, v in source_data.items() if k != 'geopotential'},
      pressure_coords=source_coords.vertical,
      sigma_coords=vertical.sigma_levels,
      surface_pressure=surface_pressure,
      interpolate_fn=interpolate_fn,
  )
  data_on_sigma['temperature_variation'] = (
      data_on_sigma.pop('temperature') - primitive_equations.T_ref
  )
  data_on_sigma['log_surface_pressure'] = jnp.log(surface_pressure)
  return data_on_sigma


def primitive_equations_to_uvtz(
    source_state: dinosaur_primitive_equations.StateWithTime,
    input_coords: coordinates.DinosaurCoordinates,
    primitive_equations: equations.PrimitiveEquations,
    orography: orographies.Orography,
    target_coords: coordinates.DinosaurCoordinates,
    sim_units: units.SimUnits,
):
  """Converts primitive equations state to pressure level representation.

  This function transforms an atmospheric state described in terms of
  temperature variation, divergence, vorticity, surface pressure and tracers
  on sigma levels to wind components, temperature, geopotential and tracers
  on fixed pressure-level coordinates.

  Args:
    source_state: State in primitive equations representation.
    input_coords: Coordinates of the source state.
    primitive_equations: Primitive equations module.
    orography: Orography module.
    target_coords: Pressure-level dinosaur coordinates to convert to.
    sim_units: Simulation units object.

  Returns:
    Dictionary of state components interpolated to target_coords.
  """
  to_nodal_fn = input_coords.horizontal.ylm_grid.to_nodal
  velocity_fn = functools.partial(
      spherical_harmonic.vor_div_to_uv_nodal,
      input_coords.horizontal.ylm_grid,
  )
  u, v = velocity_fn(  # returned in nodal space.
      vorticity=source_state.vorticity, divergence=source_state.divergence
  )
  t = dinosaur_xarray_utils.temperature_variation_to_absolute(
      to_nodal_fn(source_state.temperature_variation),
      ref_temperature=primitive_equations.T_ref.squeeze())
  tracers = to_nodal_fn(source_state.tracers)
  vertical = input_coords.vertical
  assert isinstance(vertical, coordinates.SigmaLevels)
  z = dinosaur_primitive_equations.get_geopotential_with_moisture(
      temperature=t,
      specific_humidity=tracers['specific_humidity'],
      nodal_orography=orography.nodal_orography,
      coordinates=vertical.sigma_levels,
      gravity_acceleration=sim_units.gravity_acceleration,
      ideal_gas_constant=sim_units.ideal_gas_constant,
      water_vapor_gas_constant=sim_units.water_vapor_gas_constant,
  )
  surface_pressure = jnp.exp(to_nodal_fn(source_state.log_surface_pressure))
  # u, v, t, z, tracers, surface_pressure = (
  #     self.coords.dycore_to_physics_sharding(
  #         (u, v, t, z, tracers, surface_pressure)
  #     )
  # )
  interpolate_with_linear_extrap_fn = (
      vertical_interpolation.vectorize_vertical_interpolation(
          vertical_interpolation.linear_interp_with_linear_extrap
      )
  )
  vertical = primitive_equations.coords.vertical
  assert isinstance(vertical, coordinates.SigmaLevels)
  regrid_with_linear_fn = functools.partial(
      vertical_interpolation.interp_sigma_to_pressure,
      pressure_coords=target_coords.vertical,
      sigma_coords=vertical.sigma_levels,
      surface_pressure=surface_pressure,
      interpolate_fn=interpolate_with_linear_extrap_fn,
  )
  interpolate_with_constant_extrap_fn = (
      vertical_interpolation.vectorize_vertical_interpolation(
          vertical_interpolation.vertical_interpolation
      )
  )
  regrid_with_constant_fn = functools.partial(
      vertical_interpolation.interp_sigma_to_pressure,
      pressure_coords=target_coords.vertical,
      sigma_coords=vertical.sigma_levels,
      surface_pressure=surface_pressure,
      interpolate_fn=interpolate_with_constant_extrap_fn,
  )
  # closest regridding options to those used in ERA5.
  # use constant extrapolation for `u, v, tracers`.
  # use linear extrapolation for `z, t`.
  # google reference: http://shortn/_X09ZAU1jsx.
  outputs = dict(
      u_component_of_wind=regrid_with_constant_fn(u),
      v_component_of_wind=regrid_with_constant_fn(v),
      temperature=regrid_with_linear_fn(t),
      geopotential=regrid_with_linear_fn(z),
  )
  for k, v in regrid_with_constant_fn(tracers).items():
    outputs[k] = v
  return outputs
