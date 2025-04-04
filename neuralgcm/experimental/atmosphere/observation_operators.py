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

"""Atmosphere-specific observation operators."""

import copy
import dataclasses
from typing import Sequence

from dinosaur import primitive_equations as dinosaur_primitive_equations
import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import pytree_mappings
from neuralgcm.experimental.atmosphere import equations
from neuralgcm.experimental.atmosphere import state_conversion
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import observation_operators
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units


@dataclasses.dataclass
class PressureLevelObservationOperator(
    observation_operators.ObservationOperator
):
  """Operator that predicts pressure-level observations on lon-lat grids.

  This operator is compatible with models that use primitive equation
  representation of the atmospheric state in spherical harmonics basis. The
  pressure-level observations are computed by converting the state to
  velocity/temperature/geopotential representation and then linearly
  interpolating to pressure levels using combination of linear and constant
  extrapolation strategies. If `observation_correction` mapping is specified,
  an additional correction is added to the predicted observations in lon-lat
  space.
  """

  primitive_equation: equations.PrimitiveEquations
  orography: orographies.ModalOrography
  pressure_levels: coordinates.PressureLevels
  sim_units: units.SimUnits
  tracer_names: Sequence[str]
  observation_correction: pytree_mappings.CoordsStateMapping | None
  mesh: parallelism.Mesh

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Observes fields in `query` on pressure/lon/lat coords from YLM `inputs`.

    Args:
      inputs: A dictionary of fields in spherical harmonic representation.
      query: A dictionary specifying coordinates of the queried fields.

    Returns:
      Predicted fields matching the `query`.
    """
    inputs = copy.copy(inputs)  # avoid mutating inputs.
    time = inputs.pop('time')
    input_coords = cx.get_coordinate(inputs['divergence'])
    # TODO(dkochkov): try to remove use of DinosaurCoordinates where possible,
    # as reconstructing them from axes results in a CartesianProduct instance
    # due to inheritance from cx.CartesianProduct.
    input_coords = coordinates.DinosaurCoordinates(
        vertical=input_coords.coordinates[0],
        horizontal=cx.compose_coordinates(*input_coords.coordinates[1:])
    )
    grid = input_coords.horizontal.to_lon_lat_grid()
    nondim_pressure = dataclasses.replace(
        self.pressure_levels,
        centers=self.sim_units.nondimensionalize(
            self.pressure_levels.centers * typing.units.millibar
        ),
    )
    nondim_target_coords = coordinates.DinosaurCoordinates(
        horizontal=grid, vertical=nondim_pressure
    )
    target_coords = coordinates.DinosaurCoordinates(
        horizontal=grid, vertical=self.pressure_levels
    )
    inputs = jax.tree.map(lambda x: x.data, inputs, is_leaf=cx.is_field)
    # TODO(dkochkov): make primitive_equation_to_uvtz work with flat structure
    # to avoid the need to nesting tracers.
    tracers = {k: inputs.pop(k) for k in self.tracer_names}
    # TODO(dkochkov): standardize around not having dummy dimension for surface
    # variables by adding expand_dims transform to features.
    inputs['log_surface_pressure'] = (
        inputs['log_surface_pressure'][jnp.newaxis, ...]
    )
    source_state = dinosaur_primitive_equations.State(
        **(inputs | {'tracers': tracers})
    )
    # TODO(dkochkov): pass temperature instead of temperature_variation as
    # prognostic fields in observer to remove this dependency.
    pressure_interpolated_state = state_conversion.primitive_equations_to_uvtz(
        source_state=source_state,
        input_coords=input_coords,
        primitive_equations=self.primitive_equation,
        orography=self.orography,
        target_coords=nondim_target_coords,
        sim_units=self.sim_units,
        mesh=self.mesh,
    )
    pressure_interpolated_state = parallelism.with_physics_sharding(
        self.mesh, pressure_interpolated_state
    )
    if self.observation_correction is not None:
      # TODO(dkochkov): make this unnested as we move towards using
      # dict[str, cx.Field] for internal representation.
      source_dict = source_state.asdict() | {'time': time.data}
      source_dict = parallelism.with_dycore_sharding(self.mesh, source_dict)
      correction = self.observation_correction(source_dict)
      correction = pytree_utils.replace_with_matching_or_default(
          pressure_interpolated_state, correction
      )
      correction = parallelism.with_physics_sharding(self.mesh, correction)
      add_fn = lambda x, y: x + y if y is not None else x
      pressure_interpolated_state = jax.tree.map(
          add_fn, pressure_interpolated_state, correction
      )

    observations_fields = {
        k: cx.wrap(v, target_coords)
        for k, v in pressure_interpolated_state.items()
    }
    return observation_operators.DataObservationOperator(
        observations_fields | {'time': time}
    ).observe(inputs, query)


@dataclasses.dataclass
class SigmaLevelObservationOperator(
    observation_operators.ObservationOperator
):
  """Operator that predicts sigma-level observations on lon-lat grids.

  This operator is compatible with models that use primitive equation
  representation of the atmospheric state in spherical harmonics basis. The
  sigma-level observations are computed by converting the state to
  velocity/temperature/geopotential representation and then linearly
  interpolating to desired sigma levels using combination of linear and constant
  extrapolation strategies. If `observation_correction` mapping is specified,
  an additional correction is added to the predicted observations in lon-lat
  space.
  """

  primitive_equation: equations.PrimitiveEquations
  orography: orographies.ModalOrography
  sigma_levels: coordinates.SigmaLevels
  sim_units: units.SimUnits
  tracer_names: Sequence[str]
  observation_correction: pytree_mappings.CoordsStateMapping | None
  mesh: parallelism.Mesh

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Observes fields in `query` on sigma/lon/lat coords from YLM `inputs`.

    Args:
      inputs: A dictionary of fields in spherical harmonic representation.
      query: A dictionary specifying coordinates of the queried fields.

    Returns:
      Predicted fields matching the `query`.
    """
    inputs = copy.copy(inputs)  # avoid mutating inputs.
    time = inputs.pop('time')
    input_coords = cx.get_coordinate(inputs['divergence'])
    # TODO(dkochkov): try to remove use of DinosaurCoordinates where possible,
    # as reconstructing them from axes results in a CartesianProduct instance
    # due to inheritance from cx.CartesianProduct.
    input_coords = coordinates.DinosaurCoordinates(
        vertical=input_coords.coordinates[0],
        horizontal=cx.compose_coordinates(*input_coords.coordinates[1:])
    )
    grid = input_coords.horizontal.to_lon_lat_grid()
    target_coords = coordinates.DinosaurCoordinates(
        horizontal=grid, vertical=self.sigma_levels
    )
    inputs = jax.tree.map(lambda x: x.data, inputs, is_leaf=cx.is_field)
    # TODO(dkochkov): make primitive_equation_to_uvtz work with flat structure
    # to avoid the need to nesting tracers.
    tracers = {k: inputs.pop(k) for k in self.tracer_names}
    # TODO(dkochkov): standardize around not having dummy dimension for surface
    # variables by adding expand_dims transform to features.
    inputs['log_surface_pressure'] = (
        inputs['log_surface_pressure'][jnp.newaxis, ...]
    )
    source_state = dinosaur_primitive_equations.State(
        **(inputs | {'tracers': tracers})
    )
    # TODO(dkochkov): pass temperature instead of temperature_variation as
    # prognostic fields in observer to remove this dependency.
    interpolated_state = state_conversion.primitive_equations_to_sigma(
        source_state=source_state,
        input_coords=input_coords,
        primitive_equations=self.primitive_equation,
        orography=self.orography,
        target_coords=target_coords,
        sim_units=self.sim_units,
        mesh=self.mesh,
    )
    interpolated_state = parallelism.with_physics_sharding(
        self.mesh, interpolated_state
    )
    if self.observation_correction is not None:
      # TODO(dkochkov): make this unnested as we move towards using
      # dict[str, cx.Field] for internal representation.
      source_dict = source_state.asdict() | {'time': time.data}
      source_dict = parallelism.with_dycore_sharding(self.mesh, source_dict)
      correction = self.observation_correction(source_dict)
      correction = pytree_utils.replace_with_matching_or_default(
          interpolated_state, correction
      )
      correction = parallelism.with_physics_sharding(self.mesh, correction)
      add_fn = lambda x, y: x + y if y is not None else x
      interpolated_state = jax.tree.map(
          add_fn, interpolated_state, correction
      )

    observations_fields = {
        k: cx.wrap(v, target_coords)
        for k, v in interpolated_state.items() if k != 'surface_pressure'
    }
    observations_fields['surface_pressure'] = cx.wrap(
        jnp.squeeze(interpolated_state['surface_pressure'], axis=0), grid
    )
    return observation_operators.DataObservationOperator(
        observations_fields | {'time': time}
    ).observe(inputs, query)
