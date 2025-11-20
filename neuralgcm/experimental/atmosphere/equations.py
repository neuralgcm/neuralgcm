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
    levels: coordinates.SigmaLevels | None,
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
    ref_temp_field = cx.wrap(np.array(ref_temperatures), levels)

  def linearize_fn(abs_temp: cx.Field) -> cx.Field:
    canonical = cx.canonicalize_coordinates(abs_temp.coordinate)
    ylm_set = set(
        c for c in canonical if isinstance(c, coordinates.SphericalHarmonicGrid)
    )
    if ylm_set:
      assert len(ylm_set) == 1  # cannot have multiple ylm grids.
      [ylm_grid] = list(ylm_set)
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
    levels: coordinates.SigmaLevels | None,
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
    ref_temp_field = cx.wrap(np.array(ref_temperatures), levels)

  def delinearize_fn(del_temp: cx.Field) -> cx.Field:
    """Applies delinearization to `del_temp` field."""
    canonical = cx.canonicalize_coordinates(del_temp.coordinate)
    ylm_set = set(
        c for c in canonical if isinstance(c, coordinates.SphericalHarmonicGrid)
    )
    if ylm_set:
      assert len(ylm_set) == 1  # impossible to have multiple ylm grids.
      [ylm_grid] = list(ylm_set)
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
  """Equation module for moist primitive equations ."""

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      sigma_levels: coordinates.SigmaLevels,
      sim_units: units.SimUnits,
      reference_temperatures: Sequence[float],
      tracer_names: Sequence[str],
      orography_module: orographies.ModalOrography,
      vertical_advection: Callable[..., typing.Array] = (
          sigma_coordinates.centered_vertical_advection
      ),
      equation_cls=primitive_equations.MoistPrimitiveEquationsWithCloudMoisture,
      include_vertical_advection: bool = True,
  ):
    self.ylm_map = ylm_map
    self.sigma_levels = sigma_levels
    self.orography_module = orography_module
    self.sim_units = sim_units
    self.orography = orography_module
    self.reference_temperatures = reference_temperatures
    self.tracer_names = tracer_names
    self.vertical_advection = vertical_advection
    self.include_vertical_advection = include_vertical_advection
    self.equation_cls = equation_cls
    self.linearize_transform = get_temperature_linearization_transform(
        ref_temperatures=reference_temperatures, levels=sigma_levels
    )
    self.delinearize_transform = get_temperature_delinearization_transform(
        ref_temperatures=reference_temperatures, levels=sigma_levels
    )
    self.linear_to_absolute_rename = transforms.Rename(
        rename_dict={'temperature_variation': 'temperature'}
    )

  @property
  def primitive_equation(self):
    dinosaur_coords = coordinate_systems.CoordinateSystem(
        horizontal=self.ylm_map.dinosaur_grid,
        vertical=self.sigma_levels.sigma_levels,
        spmd_mesh=self.ylm_map.dinosaur_spmd_mesh,
    )
    return self.equation_cls(
        coords=dinosaur_coords,
        physics_specs=self.sim_units,
        reference_temperature=np.asarray(self.reference_temperatures),
        orography=self.orography_module.modal_orography.data,
        vertical_advection=self.vertical_advection,
        include_vertical_advection=self.include_vertical_advection,
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
    sigma_levels, ylm_grid = self.sigma_levels, self.ylm_map.modal_grid
    tracers = {
        k: cx.wrap(state.tracers[k], sigma_levels, ylm_grid)
        for k in self.tracer_names
    }
    volume_field_names = ['divergence', 'vorticity', 'temperature_variation']
    volume_fields = {
        k: cx.wrap(getattr(state, k), sigma_levels, ylm_grid)
        for k in volume_field_names
    }
    if is_tendency:
      volume_fields = self.linear_to_absolute_rename(volume_fields)
    else:
      volume_fields = self.delinearize_transform(volume_fields)
    lsp = cx.wrap(jnp.squeeze(state.log_surface_pressure, axis=0), ylm_grid)
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
        ), is_tendency=False
    )
