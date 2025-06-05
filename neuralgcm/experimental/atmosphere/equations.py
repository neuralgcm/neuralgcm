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

from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import sigma_coordinates
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import time_integrators
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np


class PrimitiveEquations(time_integrators.ImplicitExplicitODE):
  """Equation module for moist primitive equations ."""

  def __init__(
      self,
      ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
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
    self.ylm_transform = ylm_transform
    self.sigma_levels = sigma_levels
    self.orography_module = orography_module
    self.sim_units = sim_units
    self.orography = orography_module
    self.reference_temperatures = reference_temperatures
    self.tracer_names = tracer_names
    self.vertical_advection = vertical_advection
    self.include_vertical_advection = include_vertical_advection
    self.equation_cls = equation_cls

  @property
  def primitive_equation(self):
    dinosaur_coords = coordinate_systems.CoordinateSystem(
        horizontal=self.ylm_transform.dinosaur_grid,
        vertical=self.sigma_levels.sigma_levels,
        spmd_mesh=self.ylm_transform.dinosaur_spmd_mesh,
    )
    return self.equation_cls(
        coords=dinosaur_coords,
        physics_specs=self.sim_units,
        reference_temperature=np.asarray(self.reference_temperatures),
        orography=self.orography_module.modal_orography,
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
      self, state: primitive_equations.State
  ) -> dict[str, cx.Field]:
    sigma_levels, ylm_grid = self.sigma_levels, self.ylm_transform.modal_grid
    tracers = {
        k: cx.wrap(state.tracers[k], sigma_levels, ylm_grid)
        for k in self.tracer_names
    }
    volume_field_names = ['divergence', 'vorticity', 'temperature_variation']
    volume_fields = {
        k: cx.wrap(getattr(state, k), sigma_levels, ylm_grid)
        for k in volume_field_names
    }
    lsp = cx.wrap(jnp.squeeze(state.log_surface_pressure, axis=0), ylm_grid)
    return volume_fields | tracers | {'log_surface_pressure': lsp}

  def explicit_terms(
      self, state: primitive_equations.StateWithTime
  ) -> primitive_equations.StateWithTime:
    return self._from_primitive_equations_state(
        self.primitive_equation.explicit_terms(
            self._to_primitive_equations_state(state)
        )
    )

  def implicit_terms(
      self, state: primitive_equations.StateWithTime
  ) -> primitive_equations.StateWithTime:
    return self._from_primitive_equations_state(
        self.primitive_equation.implicit_terms(
            self._to_primitive_equations_state(state)
        )
    )

  def implicit_inverse(
      self, state: primitive_equations.StateWithTime, step_size: float
  ) -> primitive_equations.StateWithTime:
    return self._from_primitive_equations_state(
        self.primitive_equation.implicit_inverse(
            self._to_primitive_equations_state(state), step_size
        )
    )
