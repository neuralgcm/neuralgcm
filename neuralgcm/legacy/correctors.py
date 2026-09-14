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
"""Modules that predict refinement or updates of time-advanced states."""

import dataclasses
from typing import Any, Callable, Optional
from dinosaur import coordinate_systems
from dinosaur import time_integration
from dinosaur import typing
import gin
import haiku as hk
import jax
from neuralgcm.legacy import equations
from neuralgcm.legacy import features
from neuralgcm.legacy import filters
from neuralgcm.legacy import integrators
from neuralgcm.legacy import mappings

Pytree = typing.Pytree
PyTreeState = typing.PyTreeState
Forcing = typing.Forcing

CorrectorFn = typing.CorrectorFn
CorrectorModule = typing.CorrectorModule
EquationModule = equations.EquationModule
FeaturesModule = features.FeaturesModule
MappingModule = mappings.MappingModule
StepModule = typing.StepModule
StepFilterModule = Callable[..., typing.PyTreeStepFilterFn]
TimeIntegrator = integrators.TimeIntegrator
TransformModule = typing.TransformModule


@gin.register
class PredictorEulerCorrector(hk.Module):
  """Corrector that takes Euler step ontop of a predictor step."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      predictor_module: StepModule,
      filter_module: StepFilterModule = filters.NoFilter,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.dt = dt
    self.step_fn = predictor_module(coords, dt, physics_specs, aux_features)
    self.filter_fn = filter_module(coords, dt, physics_specs, aux_features)

  def __call__(
      self,
      state: typing.PyTreeState,
      tendencies: typing.PyTreeState,
      forcing: Optional[Forcing] = None,
  ) -> typing.PyTreeState:
    state = self.step_fn(state, forcing)
    euler_add_fn = lambda x, y: x + self.dt * y if y is not None else x
    result = jax.tree_util.tree_map(euler_add_fn, state, tendencies)
    return self.filter_fn(state, result)


@gin.register
class DycoreWithPhysicsCorrector(hk.Module):
  """Corrector that runs dycore with physics tendencies added to explicit terms.

  This corrector treats predicted physics tendencies constant at each time
  interval and includes them to all substeps of the dycore step. To achieve this
  the dycore in this module is specified by the governing equation, rather than
  an `EquationStep`.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      dycore_equation_module: EquationModule = gin.REQUIRED,
      dycore_substeps: int = gin.REQUIRED,
      time_integrator: TimeIntegrator = integrators.imex_rk_sil3,
      filter_module: StepFilterModule = filters.NoFilter,
      checkpoint_explicit_terms: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    dycore_equation = dycore_equation_module(
        coords, dt, physics_specs, aux_features)
    if checkpoint_explicit_terms:
      dycore_equation = time_integration.ImplicitExplicitODE.from_functions(
          hk.remat(dycore_equation.explicit_terms),
          dycore_equation.implicit_terms,
          dycore_equation.implicit_inverse)  # pyrefly: ignore[bad-argument-type]
    self.coords = coords
    self.dycore_equation = dycore_equation
    self.dycore_substeps = dycore_substeps
    self.inner_dt = dt / dycore_substeps
    self.dt = dt
    self.time_integrator = time_integrator
    self.filter_fn = filter_module(coords, dt, physics_specs, aux_features)

  def __call__(
      self,
      state: typing.PyTreeState,
      tendencies: typing.PyTreeState,
      forcing: Optional[Forcing] = None,
  ) -> typing.PyTreeState:
    state, tendencies = self.coords.with_dycore_sharding((state, tendencies))
    physics_parametrization_eq = time_integration.ExplicitODE.from_functions(
        lambda state: tendencies)
    all_equations = (self.dycore_equation, physics_parametrization_eq)
    equation = time_integration.compose_equations(all_equations)
    step_fn = self.time_integrator(equation, self.inner_dt)
    # TODO(dkochkov) make step_with_filters work with single filter.
    step_fn = time_integration.step_with_filters(step_fn, [self.filter_fn])
    step_fn = time_integration.repeated(step_fn, self.dycore_substeps, hk.scan)
    state = time_integration.maybe_fix_sim_time_roundoff(
        step_fn(state), self.dt
    )
    state = self.coords.with_dycore_sharding(state)
    return state


@gin.register
class CustomCoordsCorrector(hk.Module):
  """Corrector module that uses gin-configured coordinates instead of coords.

  This class currently supports model states in spectral representation. It
  could be easily extended to nodal-state models by converting to modal space
  prior to spectral interpolation and back after the timestep if performed.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      corrector_module: CorrectorModule = gin.REQUIRED,
      custom_coords: coordinate_systems.CoordinateSystem = gin.REQUIRED,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    custom_coords = dataclasses.replace(
        custom_coords, spmd_mesh=coords.spmd_mesh
    )
    self.corrector_fn = corrector_module(
        custom_coords, dt, physics_specs, aux_features)
    self.to_custom_coords_fn = coordinate_systems.get_spectral_interpolate_fn(
        coords, custom_coords)
    self.from_custom_coords_fn = coordinate_systems.get_spectral_interpolate_fn(
        custom_coords, coords)

  def __call__(
      self,
      state: typing.PyTreeState,
      tendencies: typing.PyTreeState,
      forcing: Optional[Forcing] = None,
  ) -> typing.PyTreeState:
    state = self.to_custom_coords_fn(state)
    tendencies = self.to_custom_coords_fn(tendencies)
    # TODO(dkochkov) Consider adding forcing interpolated to custom coords.
    custom_out = self.corrector_fn(state, tendencies, None)
    return self.from_custom_coords_fn(custom_out)
