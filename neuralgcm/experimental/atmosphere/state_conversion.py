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

import dataclasses
import functools

import coordax as cx
from dinosaur import primitive_equations as dinosaur_primitive_equations
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
from dinosaur import xarray_utils as dinosaur_xarray_utils
import jax.numpy as jnp
from neuralgcm.experimental.atmosphere import equations
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units


def uvtz_to_primitive_equations(
    source_data: dict[str, cx.Field],
    source_coords: coordinates.DinosaurCoordinates,
    primitive_equations: equations.PrimitiveEquations,
    orography,
    sim_units: units.SimUnits,
) -> dict[str, cx.Field]:
  """Converts velocity/temperature/geopotential to primitive equations state."""
  # TODO(dkochkov): Update this method to leverage coordax.
  surface_pressure = vertical_interpolation.get_surface_pressure(
      source_coords.vertical,
      source_data['geopotential'].data,
      orography.nodal_orography,
      sim_units.g,
  )[
      0, ...
  ]  # Dinosaur uses dummy surface dimension for which we remove.
  interpolate_fn = vertical_interpolation.vectorize_vertical_interpolation(
      vertical_interpolation.vertical_interpolation
  )
  vertical = primitive_equations.sigma_levels
  # TODO(dkochkov): Consider having explicit args that used differently, eg z.
  data_on_sigma = vertical_interpolation.interp_pressure_to_sigma(
      {k: v.data for k, v in source_data.items() if k != 'geopotential'},
      pressure_coords=source_coords.vertical,
      sigma_coords=vertical.sigma_levels,
      surface_pressure=surface_pressure,
      interpolate_fn=interpolate_fn,
  )
  data_on_sigma['temperature_variation'] = (
      data_on_sigma.pop('temperature') - primitive_equations.T_ref
  )
  grid = source_coords.horizontal
  out_coords = coordinates.DinosaurCoordinates(
      horizontal=grid, vertical=vertical
  )
  data_on_sigma = {k: cx.wrap(v, out_coords) for k, v in data_on_sigma.items()}
  data_on_sigma['log_surface_pressure'] = cx.wrap(
      jnp.log(surface_pressure), grid
  )
  return data_on_sigma


def primitive_equations_to_uvtz(
    source_state: dinosaur_primitive_equations.StateWithTime,
    ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
    sigma_levels: coordinates.SigmaLevels,
    pressure_levels: coordinates.PressureLevels,
    primitive_equations: equations.PrimitiveEquations,
    orography: orographies.Orography,
    sim_units: units.SimUnits,
) -> dict[str, cx.Field]:
  """Converts primitive equations state to pressure level representation.

  This function transforms an atmospheric state described in terms of
  temperature variation, divergence, vorticity, surface pressure and tracers
  on sigma levels to wind components, temperature, geopotential and tracers
  on fixed pressure-level coordinates.

  Args:
    source_state: State in primitive equations representation.
    ylm_transform: Spherical transform that defines modal-nodal conversion.
    sigma_levels: Sigma coordinates used by the primitive equations module.
    pressure_levels: Pressure coordinates on which to comupte the outputs.
    primitive_equations: Primitive equations module.
    orography: Orography module.
    sim_units: Simulation units object.

  Returns:
    Dictionary of state components interpolated to target_coords.
  """
  # TODO(dkochkov): Update this method to leverage coordax.
  nondim_pressure = dataclasses.replace(
      pressure_levels.pressure_levels,
      centers=sim_units.nondimensionalize(
          pressure_levels.pressure_levels.centers * typing.units.millibar
      ),
  )
  to_nodal_fn = ylm_transform.to_nodal_array
  velocity_fn = functools.partial(
      spherical_harmonic.vor_div_to_uv_nodal,
      ylm_transform.dinosaur_grid,
  )
  u, v = velocity_fn(  # returned in nodal space.
      vorticity=source_state.vorticity, divergence=source_state.divergence
  )
  t = dinosaur_xarray_utils.temperature_variation_to_absolute(
      to_nodal_fn(source_state.temperature_variation),
      ref_temperature=primitive_equations.T_ref.squeeze(),
  )
  tracers = {k: to_nodal_fn(v) for k, v in source_state.tracers.items()}
  if (
      'specific_cloud_ice_water_content' in tracers.keys()
      and 'specific_cloud_liquid_water_content' in tracers.keys()
  ):
    clouds = (
        tracers['specific_cloud_ice_water_content']
        + tracers['specific_cloud_liquid_water_content']
    )
  else:
    clouds = None

  z = dinosaur_primitive_equations.get_geopotential_with_moisture(
      temperature=t,
      specific_humidity=tracers['specific_humidity'],
      nodal_orography=orography.nodal_orography,
      coordinates=sigma_levels.sigma_levels,
      gravity_acceleration=sim_units.gravity_acceleration,
      ideal_gas_constant=sim_units.ideal_gas_constant,
      water_vapor_gas_constant=sim_units.water_vapor_gas_constant,
      clouds=clouds,
  )
  surface_pressure = jnp.exp(to_nodal_fn(source_state.log_surface_pressure))
  u, v, t, z, tracers, surface_pressure = (
      parallelism.with_dycore_to_physics_sharding(
          ylm_transform.mesh, (u, v, t, z, tracers, surface_pressure)
      )
  )
  interpolate_with_linear_extrap_fn = (
      vertical_interpolation.vectorize_vertical_interpolation(
          vertical_interpolation.linear_interp_with_linear_extrap
      )
  )
  regrid_with_linear_fn = functools.partial(
      vertical_interpolation.interp_sigma_to_pressure,
      pressure_coords=nondim_pressure,
      sigma_coords=sigma_levels.sigma_levels,
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
      pressure_coords=nondim_pressure,
      sigma_coords=sigma_levels.sigma_levels,
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
  target_coords = coordinates.DinosaurCoordinates(
      horizontal=ylm_transform.nodal_grid, vertical=pressure_levels
  )
  outputs = {k: cx.wrap(v, target_coords) for k, v in outputs.items()}
  return outputs


def sigma_to_primitive_equations(
    source_data: dict[str, cx.Field],
    source_coords: coordinates.DinosaurCoordinates,
    primitive_equations: equations.PrimitiveEquations,
    orography,
    sim_units: units.SimUnits,
) -> dict[str, cx.Field]:
  """Converts v/u/temp/geopot from sigma levels to primitive equations state."""
  del orography, sim_units
  # TODO(dkochkov): Update this method to leverage coordax.
  data_on_new_sigma = vertical_interpolation.interp_sigma_to_sigma(
      {k: v.data for k, v in source_data.items() if k != 'geopotential'},
      source_sigma=source_coords.vertical,
      target_sigma=primitive_equations.sigma_levels,
  )

  surface_pressure = data_on_new_sigma.pop('surface_pressure')
  data_on_new_sigma['temperature_variation'] = (
      data_on_new_sigma.pop('temperature') - primitive_equations.T_ref
  )
  grid = source_coords.horizontal
  target_coords = coordinates.DinosaurCoordinates(
      horizontal=grid,
      vertical=primitive_equations.sigma_levels,
  )
  data_on_new_sigma = {
      k: cx.wrap(v, target_coords) for k, v in data_on_new_sigma.items()
  }
  data_on_new_sigma['log_surface_pressure'] = cx.wrap(
      jnp.log(surface_pressure), grid
  )
  return data_on_new_sigma


def primitive_equations_to_sigma(
    source_state: dinosaur_primitive_equations.StateWithTime,
    ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
    sigma_levels: coordinates.SigmaLevels,
    primitive_equations: equations.PrimitiveEquations,
    orography: orographies.Orography,
    target_coords: coordinates.DinosaurCoordinates,
    sim_units: units.SimUnits,
) -> dict[str, cx.Field]:
  """Converts primitive equations state to sigma level data representation."""
  # TODO(dkochkov): Update this method to leverage coordax.
  to_nodal_fn = ylm_transform.to_nodal_array
  velocity_fn = functools.partial(
      spherical_harmonic.vor_div_to_uv_nodal,
      ylm_transform.dinosaur_grid,
  )
  u, v = velocity_fn(  # returned in nodal space.
      vorticity=source_state.vorticity, divergence=source_state.divergence
  )
  t = dinosaur_xarray_utils.temperature_variation_to_absolute(
      to_nodal_fn(source_state.temperature_variation),
      ref_temperature=primitive_equations.T_ref.squeeze(),
  )
  tracers = {k: to_nodal_fn(v) for k, v in source_state.tracers.items()}
  if (
      'specific_cloud_ice_water_content' in tracers.keys()
      and 'specific_cloud_liquid_water_content' in tracers.keys()
  ):
    clouds = (
        tracers['specific_cloud_ice_water_content']
        + tracers['specific_cloud_liquid_water_content']
    )
  else:
    clouds = None

  z = dinosaur_primitive_equations.get_geopotential_with_moisture(
      temperature=t,
      specific_humidity=tracers['specific_humidity'],
      nodal_orography=orography.nodal_orography,
      coordinates=sigma_levels.sigma_levels,
      gravity_acceleration=sim_units.gravity_acceleration,
      ideal_gas_constant=sim_units.ideal_gas_constant,
      water_vapor_gas_constant=sim_units.water_vapor_gas_constant,
      clouds=clouds,
  )

  surface_pressure = jnp.exp(to_nodal_fn(source_state.log_surface_pressure))

  u, v, t, z, tracers, surface_pressure = (
      parallelism.with_dycore_to_physics_sharding(
          ylm_transform.mesh, (u, v, t, z, tracers, surface_pressure)
      )
  )

  interpolate_with_linear_extrap_fn = (
      vertical_interpolation.linear_interp_with_linear_extrap
  )
  regrid_with_linear_fn = functools.partial(
      vertical_interpolation.interp_sigma_to_sigma,
      source_sigma=sigma_levels.sigma_levels,
      target_sigma=target_coords.vertical.sigma_levels,
      interpolate_fn=interpolate_with_linear_extrap_fn,
  )

  outputs = dict(
      u_component_of_wind=regrid_with_linear_fn(u),
      v_component_of_wind=regrid_with_linear_fn(v),
      temperature=regrid_with_linear_fn(t),
      geopotential=regrid_with_linear_fn(z),
  )
  for k, v in regrid_with_linear_fn(tracers).items():
    outputs[k] = v

  outputs = {k: cx.wrap(v, target_coords) for k, v in outputs.items()}
  outputs['surface_pressure'] = cx.wrap(
      surface_pressure, target_coords.horizontal
  )
  return outputs
