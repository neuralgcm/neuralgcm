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

"""Modules parameterizing PDEs describing atmospheric processes."""

from typing import Callable, Sequence

import coordax as cx
from dinosaur import coordinate_systems
from dinosaur import held_suarez
from dinosaur import hybrid_coordinates
from dinosaur import primitive_equations
from dinosaur import sigma_coordinates
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import time_integrators
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np


def get_temperature_linearization_transform(
    ref_temperatures: Sequence[float] | cx.Field,
    levels: coordinates.SigmaLevels | coordinates.HybridLevels |None,
    abs_temperature_key: str = 'temperature',
    del_temperature_key: str = 'temperature_variation',
) -> transforms.Transform:
  """Constructs transform for linearizing temperature around `ref_temperature`."""
  if isinstance(ref_temperatures, cx.Field):
    if levels is not None and cx.get_coordinate(ref_temperatures) != levels:
      raise ValueError(
          f'ref_temperatures coordinate {cx.get_coordinate(ref_temperatures)}'
          f' does not match levels coordinate {levels}.'
      )
    ref_temp_field = ref_temperatures
  else:  # Sequence[float]
    if levels is None:
      raise ValueError(
          '`levels` must be provided for sequence `ref_temperatures`'
      )
    ref_temp_field = cx.field(np.array(ref_temperatures), levels)

  def linearize_fn(abs_temp: cx.Field) -> cx.Field:
    ylm_dims = ('longitude_wavenumber', 'total_wavenumber')
    if all(d in abs_temp.dims for d in ylm_dims):
      ylm_grid = cx.coords.extract(
          abs_temp.coordinate, coordinates.SphericalHarmonicGrid
      )
      del_temp = ylm_grid.add_constant(abs_temp, -ref_temp_field)
    else:
      del_temp = abs_temp - ref_temp_field
    return del_temp

  return transforms.Sequential([
      transforms.ApplyFnToKeys(
          fn=linearize_fn,
          keys=[abs_temperature_key],
          include_remaining=True,
      ),
      transforms.Rename(rename_dict={abs_temperature_key: del_temperature_key}),
  ])


def get_temperature_delinearization_transform(
    ref_temperatures: Sequence[float] | cx.Field,
    levels: coordinates.SigmaLevels | coordinates.HybridLevels | None,
    abs_temperature_key: str = 'temperature',
    del_temperature_key: str = 'temperature_variation',
) -> transforms.Transform:
  """Constructs transform for reversing temperature linearization."""
  if isinstance(ref_temperatures, cx.Field):
    if levels is not None and cx.get_coordinate(ref_temperatures) != levels:
      raise ValueError(
          f'ref_temperatures coordinate {cx.get_coordinate(ref_temperatures)}'
          f' does not match levels coordinate {levels}.'
      )
    ref_temp_field = ref_temperatures
  else:  # Sequence[float]
    if levels is None:
      raise ValueError(
          '`levels` must be provided for sequence `ref_temperatures`'
      )
    ref_temp_field = cx.field(np.array(ref_temperatures), levels)

  def delinearize_fn(del_temp: cx.Field) -> cx.Field:
    """Applies delinearization to `del_temp` field."""
    ylm_dims = ('longitude_wavenumber', 'total_wavenumber')
    if all(d in del_temp.dims for d in ylm_dims):
      ylm_grid = cx.coords.extract(
          del_temp.coordinate, coordinates.SphericalHarmonicGrid
      )
      abs_temp = ylm_grid.add_constant(del_temp, ref_temp_field)
    else:
      abs_temp = del_temp + ref_temp_field
    return abs_temp

  return transforms.Sequential([
      transforms.ApplyFnToKeys(
          fn=delinearize_fn,
          keys=[del_temperature_key],
          include_remaining=True,
      ),
      transforms.Rename(rename_dict={del_temperature_key: abs_temperature_key}),
  ])


class PrimitiveEquations(time_integrators.ImplicitExplicitODE):
  """Equation module for primitive equations.

  This module wraps methods of an appropriate primitive equations class from
  `dinosaur` and converts between dict[str, cx.Field] and dinosaur convention
  representations. The type of primitive equation solver is selected by the
  type of the vertical coordinate system. Supported vertical coordinates include
  SigmaLevels and HybridLevels for which spectral solvers are available. Other
  arguments control the additional features of the primitive equations solver,
  such as vertical advection and account for moisture species.

  Attributes:
    ylm_map: Spherical harmonics mapping for the horizontal grid.
    levels: Vertical levels coordinate.
    sim_units: Physical constants and units for nondimensionalization.
    reference_temperatures: Reference temperatures used for linearization.
    tracer_names: A sequence of names of tracers to be evolved by dynamics.
    orography_module: Orography module that provides modal orography data.
    vertical_advection: A optional custom function that implements vertical
      advection scheme. If None, a default centered difference scheme will be
      used based on the type of `levels`.
    include_vertical_advection: Whether to include vertical advection terms.
    humidity_key: Key in tracers names that corresponds to specific humidity.
      If the key is not present in `tracer_names`, uses dry primitive equations.
    cloud_keys: Keys in tracers names that corresponds to cloud species. Uses
      only keys that are present in `tracer_names`. If at least one of the cloud
      species is present, humidity key must be present in `tracer_names`.
  """

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      levels: coordinates.SigmaLevels | coordinates.HybridLevels,
      sim_units: units.SimUnits,
      reference_temperatures: Sequence[float],
      tracer_names: Sequence[str],
      orography_module: orographies.ModalOrography,
      vertical_advection: Callable[..., typing.Array] | None = None,
      include_vertical_advection: bool = True,
      humidity_key: str = 'specific_humidity',
      cloud_keys: tuple[str, ...] = (
          'specific_cloud_ice_water_content',
          'specific_cloud_liquid_water_content',
      ),
  ):
    self.ylm_map = ylm_map
    self.levels = levels
    self.orography_module = orography_module
    self.sim_units = sim_units
    self.orography = orography_module
    self.reference_temperatures = reference_temperatures
    self.tracer_names = tracer_names
    self.include_vertical_advection = include_vertical_advection
    self.linearize_transform = get_temperature_linearization_transform(
        ref_temperatures=reference_temperatures, levels=levels
    )
    self.delinearize_transform = get_temperature_delinearization_transform(
        ref_temperatures=reference_temperatures, levels=levels
    )
    self.linear_to_absolute_rename = transforms.Rename(
        rename_dict={'temperature_variation': 'temperature'}
    )
    if isinstance(levels, coordinates.SigmaLevels):
      self.equation_cls = primitive_equations.PrimitiveEquationsSigma
      self.dinosaur_coords = coordinate_systems.CoordinateSystem(
          horizontal=self.ylm_map.dinosaur_grid,
          vertical=self.levels.sigma_levels,
          spmd_mesh=self.ylm_map.dinosaur_spmd_mesh,
      )
      if vertical_advection is None:
        vertical_advection = sigma_coordinates.centered_vertical_advection
      self.vertical_advection = vertical_advection
      self.unit_kwargs = {}
    elif isinstance(levels, coordinates.HybridLevels):
      self.equation_cls = primitive_equations.PrimitiveEquationsHybrid
      self.dinosaur_coords = coordinate_systems.CoordinateSystem(
          horizontal=self.ylm_map.dinosaur_grid,
          vertical=self.levels.hybrid_levels,
          spmd_mesh=self.ylm_map.dinosaur_spmd_mesh,
      )
      if vertical_advection is None:
        vertical_advection = hybrid_coordinates.centered_vertical_advection
      self.vertical_advection = vertical_advection
      self.unit_kwargs = {
          'hpa_quantity': typing.units.hPa,
          'reference_surface_pressure': 101325.0 * typing.units.pascal,
      }
    else:
      raise ValueError(f'Unsupported vertical coordinate system: {levels}')
    if humidity_key in tracer_names:
      self.humidity_key = humidity_key
    else:
      self.humidity_key = None
    present_cloud_keys = tuple(k for k in cloud_keys if k in tracer_names)
    if present_cloud_keys:
      self.cloud_keys = present_cloud_keys
    else:
      self.cloud_keys = None

  @property
  def primitive_equation(self):
    return self.equation_cls(
        coords=self.dinosaur_coords,
        physics_specs=self.sim_units,
        reference_temperature=np.asarray(self.reference_temperatures),
        orography=self.orography_module.modal_orography.data,
        vertical_advection=self.vertical_advection,
        include_vertical_advection=self.include_vertical_advection,
        humidity_key=self.humidity_key,
        cloud_keys=self.cloud_keys,
        **self.unit_kwargs,
    )

  @property
  def T_ref(self) -> typing.Array:
    return self.primitive_equation.T_ref

  def _to_primitive_equations_state(
      self, inputs: dict[str, cx.Field]
  ) -> primitive_equations.State:
    """Converts a dict of fields to a primitive equations state."""
    inputs = self.linearize_transform(inputs)
    tracers_dict = {k: inputs[k].data for k in self.tracer_names}
    log_surface_pressure = inputs['log_surface_pressure'].data[np.newaxis]
    return primitive_equations.State(
        divergence=inputs['divergence'].data,
        vorticity=inputs['vorticity'].data,
        temperature_variation=inputs['temperature_variation'].data,
        tracers=tracers_dict,
        log_surface_pressure=log_surface_pressure,
    )

  def _from_primitive_equations_state(
      self, state: primitive_equations.State, is_tendency: bool = True
  ) -> dict[str, cx.Field]:
    sigma_levels, ylm_grid = self.levels, self.ylm_map.modal_grid
    tracers = {
        k: cx.field(state.tracers[k], sigma_levels, ylm_grid)
        for k in self.tracer_names
    }
    volume_field_names = ['divergence', 'vorticity', 'temperature_variation']
    volume_fields = {
        k: cx.field(getattr(state, k), sigma_levels, ylm_grid)
        for k in volume_field_names
    }
    if is_tendency:
      volume_fields = self.linear_to_absolute_rename(volume_fields)
    else:
      volume_fields = self.delinearize_transform(volume_fields)
    lsp = cx.field(jnp.squeeze(state.log_surface_pressure, axis=0), ylm_grid)
    return volume_fields | tracers | {'log_surface_pressure': lsp}

  def explicit_terms(self, state: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return self._from_primitive_equations_state(
        self.primitive_equation.explicit_terms(
            self._to_primitive_equations_state(state)
        )
    )

  def implicit_terms(self, state: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return self._from_primitive_equations_state(
        self.primitive_equation.implicit_terms(
            self._to_primitive_equations_state(state)
        )
    )

  def implicit_inverse(
      self, state: dict[str, cx.Field], step_size: float
  ) -> dict[str, cx.Field]:
    return self._from_primitive_equations_state(
        self.primitive_equation.implicit_inverse(
            self._to_primitive_equations_state(state), step_size
        ),
        is_tendency=False,
    )


class HeldSuarezForcing(time_integrators.ExplicitODE):
  """Equation module for Held-Suarez forcing.

  This module implements Held-Suarez forcing terms, which are often used for
  benchmarking atmospheric models. It includes Rayleigh friction to relax
  horizontal velocities to zero, and Newtonian cooling to relax temperature
  to an equilibrium profile.

  Attributes:
    ylm_map: Spherical harmonics mapping for the horizontal grid.
    levels: Vertical levels coordinate.
    sim_units: Physical constants and units for nondimensionalization.
    reference_temperatures: Reference temperature used for linearization. When
      used with PrimitiveEquations class, this should be the same as the one
      used to initialize the primitive equations class.
    p0: Reference surface pressure used in Held-Suarez forcing.
    sigma_b: Sigma level below which Rayleigh friction is applied.
    kf: Time scale for Rayleigh friction.
    ka: Time scale for Newtonian cooling in the troposphere.
    ks: Time scale for Newtonian cooling in the stratosphere.
    min_t: Minimum equilibrium temperature for Newtonian cooling.
    max_t: Maximum equilibrium temperature for Newtonian cooling.
    d_ty: Temperature diff for equilibrium profile in meridional direction.
    d_thz: Temperature diff for equilibrium profile in vertical direction.
  """

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      levels: coordinates.SigmaLevels | coordinates.HybridLevels,
      sim_units: units.SimUnits,
      reference_temperatures: Sequence[float],
      p0: typing.Quantity = 1e5 * typing.units.pascal,
      sigma_b: float = 0.7,
      kf: typing.Quantity = 1 / (1 * typing.units.day),
      ka: typing.Quantity = 1 / (40 * typing.units.day),
      ks: typing.Quantity = 1 / (4 * typing.units.day),
      min_t: typing.Quantity = 200 * typing.units.kelvin,
      max_t: typing.Quantity = 315 * typing.units.kelvin,
      d_ty: typing.Quantity = 60 * typing.units.kelvin,
      d_thz: typing.Quantity = 10 * typing.units.kelvin,
  ):
    self.ylm_map = ylm_map
    self.levels = levels
    self.sim_units = sim_units
    self.reference_temperatures = reference_temperatures
    self.p0 = p0
    self.sigma_b = sigma_b
    self.kf = kf
    self.ka = ka
    self.ks = ks
    self.min_t = min_t
    self.max_t = max_t
    self.d_ty = d_ty
    self.d_thz = d_thz
    if isinstance(levels, coordinates.SigmaLevels):
      self.forcing_cls = held_suarez.HeldSuarezForcingSigma
      self.units_kwargs = {}
    elif isinstance(levels, coordinates.HybridLevels):
      self.forcing_cls = held_suarez.HeldSuarezForcingHybrid
      self.units_kwargs = {'hpa_quantity': typing.units.hPa}
    else:
      raise ValueError(f'Unsupported vertical coordinate system: {levels}')
    self.linearize_transform = get_temperature_linearization_transform(
        ref_temperatures=reference_temperatures, levels=levels
    )
    self.linear_to_absolute_rename = transforms.Rename(
        rename_dict={'temperature_variation': 'temperature'}
    )

  @property
  def forcing(self):
    if isinstance(self.levels, coordinates.SigmaLevels):
      vertical_coords = self.levels.sigma_levels
    else:
      vertical_coords = self.levels.hybrid_levels
    dinosaur_coords = coordinate_systems.CoordinateSystem(
        horizontal=self.ylm_map.dinosaur_grid,
        vertical=vertical_coords,
        spmd_mesh=self.ylm_map.dinosaur_spmd_mesh,
    )
    return self.forcing_cls(
        coords=dinosaur_coords,
        physics_specs=self.sim_units,
        reference_temperature=np.asarray(self.reference_temperatures),
        p0=self.p0,
        sigma_b=self.sigma_b,
        kf=self.kf,
        ka=self.ka,
        ks=self.ks,
        minT=self.min_t,
        maxT=self.max_t,
        dTy=self.d_ty,
        dThz=self.d_thz,
        **self.units_kwargs,
    )

  def _to_primitive_equations_state(
      self, inputs: dict[str, cx.Field]
  ) -> primitive_equations.State:
    """Converts a dict of fields to a primitive equations state."""
    inputs = self.linearize_transform(inputs)  # temperature -> variation.
    log_surface_pressure = inputs['log_surface_pressure'].data[np.newaxis]
    return primitive_equations.State(
        divergence=inputs['divergence'].data,
        vorticity=inputs['vorticity'].data,
        temperature_variation=inputs['temperature_variation'].data,
        log_surface_pressure=log_surface_pressure,
    )

  def _from_primitive_equations_state(
      self, state: primitive_equations.State
  ) -> dict[str, cx.Field]:
    levels, ylm_grid = self.levels, self.ylm_map.modal_grid
    volume_field_names = ['divergence', 'vorticity', 'temperature_variation']
    volume_fields = {
        k: cx.field(getattr(state, k), levels, ylm_grid)
        for k in volume_field_names
    }
    volume_fields = self.linear_to_absolute_rename(volume_fields)
    lsp = cx.field(jnp.squeeze(state.log_surface_pressure, axis=0), ylm_grid)
    return volume_fields | {'log_surface_pressure': lsp}

  def explicit_terms(
      self, state: primitive_equations.StateWithTime
  ) -> primitive_equations.StateWithTime:
    return self._from_primitive_equations_state(
        self.forcing.explicit_terms(self._to_primitive_equations_state(state))
    )
