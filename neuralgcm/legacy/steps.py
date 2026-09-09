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
"""Modules that parameterize composed time-steppers."""

import abc
import functools
from typing import Any, Callable, Optional, Sequence
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import time_integration
from dinosaur import typing
import gin
import haiku as hk
from neuralgcm.legacy import diagnostics
from neuralgcm.legacy import integrators
from neuralgcm.legacy import perturbations
from neuralgcm.legacy import stochastic

DiagnosticModule = diagnostics.DiagnosticModule
Forcing = typing.Forcing
Pytree = typing.Pytree
ModelState = typing.ModelState
EquationModule = Callable[..., time_integration.ImplicitExplicitODE]
CorrectorModule = typing.CorrectorModule
PerturbationModule = perturbations.PerturbationModule
RandomnessModule = stochastic.RandomnessModule
PyTreeStepFilterModule = typing.PyTreeStepFilterModule
TimeIntegrator = integrators.TimeIntegrator
TransformModule = typing.TransformModule


class BaseStep(abc.ABC):
  """Base class for Step modules."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      diagnostics_module: DiagnosticModule = diagnostics.NoDiagnostics,
      randomness_module: RandomnessModule = stochastic.NoRandomField,
  ):
    self.diagnostics_fn = diagnostics_module(
        coords, dt, physics_specs, aux_features)
    self.randomness_fn = randomness_module(
        coords, dt, physics_specs, aux_features)

  @abc.abstractmethod
  def __call__(
      self,
      state: ModelState,
      forcing: typing.Forcing,
  ) -> ModelState:
    """Computes the state of the system evolved in time by `self.dt`."""

  def finalize_state(
      self,
      x: ModelState,
      forcing: typing.Forcing,
  ) -> ModelState:
    """Finalizes initialization of a model state `x`, encoded from data.

    This method ensures that state has all of the `ModelState` fields
    initialized in a way compatible with this step function. This includes
    populating initial `diagnostics`, `memory` and `randomness` fields.

    Args:
      x: Initial values for the model state typically provided by the encoder.
        forcing: Data covariates from the same time slice as `x`.

    Returns:
      Initialized model state.
    """
    x.randomness = self.randomness_fn.unconditional_sample(
        hk.maybe_next_rng_key()
    )
    x.diagnostics = self.diagnostics_fn(
        x, physics_tendencies=None, forcing=forcing)
    return x


@gin.register
class EquationStep(BaseStep, hk.Module):
  """Step module that advances the state by integrating an equation in time."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      equation_module: EquationModule,
      time_integrator: TimeIntegrator = integrators.imex_rk_sil3,
      filter_modules: Sequence[PyTreeStepFilterModule] = tuple(),
      checkpoint_explicit_terms: bool = True,
      name: Optional[str] = None,
  ):
    hk.Module.__init__(self, name=name)
    BaseStep.__init__(self, coords, dt, physics_specs, aux_features)
    equation = equation_module(coords, dt, physics_specs, aux_features)
    if checkpoint_explicit_terms:
      equation = time_integration.ImplicitExplicitODE.from_functions(
          hk.remat(equation.explicit_terms),
          equation.implicit_terms,
          equation.implicit_inverse)  # pyrefly: ignore[bad-argument-type]
    step_fn = time_integrator(equation, dt)
    filter_fns = [
        module(coords, dt, physics_specs, aux_features)
        for module in filter_modules]
    self.dt = dt
    self.step_fn = time_integration.step_with_filters(step_fn, filter_fns)

  def __call__(
      self,
      x: ModelState,
      forcing: Optional[typing.Forcing] = None,
  ) -> ModelState:
    """Computes the state of the system evolved in time by `dt`."""
    del forcing
    next_state = time_integration.maybe_fix_sim_time_roundoff(
        self.step_fn(x.state), self.dt)
    return ModelState(next_state)  # pyrefly: ignore[bad-argument-count]


@gin.register
class RepeatedStep(BaseStep, hk.Module):
  """Step module that consists of repeated substeps of the same form."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      inner_step_module: typing.StepModule,
      num_inner_steps: int = 1,
      name: Optional[str] = None,
  ):
    hk.Module.__init__(self, name=name)
    BaseStep.__init__(self, coords, dt, physics_specs, aux_features)
    inner_dt = dt / num_inner_steps
    self.step_fn = inner_step_module(
        coords, inner_dt, physics_specs, aux_features)
    self.num_inner_steps = num_inner_steps

  def __call__(
      self,
      state: ModelState,
      forcing: typing.Forcing,
  ) -> ModelState:
    """Computes the state of the system evolved in time by `dt`."""
    step_fn = functools.partial(self.step_fn, forcing=forcing)  # pyrefly: ignore[unexpected-keyword]
    step_fn = time_integration.repeated(step_fn, self.num_inner_steps, hk.scan)
    return step_fn(state)


@gin.register
class CustomCoordsStep(BaseStep, hk.Module):
  """Step module that uses gin-configured coordinates instead of coords.

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
      step_module: typing.StepModule,
      custom_coords: coordinate_systems.CoordinateSystem = gin.REQUIRED,  # pyrefly: ignore[bad-function-definition]
      name: Optional[str] = None,
  ):
    hk.Module.__init__(self, name=name)
    BaseStep.__init__(self, coords, dt, physics_specs, aux_features)
    self.step_fn = step_module(
        custom_coords, dt, physics_specs, aux_features)
    self.to_custom_coords_fn = coordinate_systems.get_spectral_interpolate_fn(
        coords, custom_coords)
    self.from_custom_coords_fn = coordinate_systems.get_spectral_interpolate_fn(
        custom_coords, coords)

  def __call__(
      self,
      x: typing.PyTreeState,
      forcing: typing.Forcing,
  ) -> typing.PyTreeState:
    del forcing  # currently not supported.
    x = self.to_custom_coords_fn(x)
    custom_out = self.step_fn(x, None)
    return self.from_custom_coords_fn(custom_out)


@gin.register
class StochasticPhysicsParameterizationStep(BaseStep, hk.Module):
  """Step module that uses stochastic physics tendencies with dycore."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      corrector_module: CorrectorModule,
      physics_parameterization_module: typing.ParameterizationModule,
      num_substeps: int = 1,
      diagnostics_module: DiagnosticModule = diagnostics.NoDiagnostics,
      randomness_module: RandomnessModule = stochastic.ZerosRandomField,
      perturbation_module: PerturbationModule = perturbations.NoPerturbation,
      checkpoint_substep: bool = False,
      name: Optional[str] = None,
  ):
    hk.Module.__init__(self, name=name)
    BaseStep.__init__(
        self, coords, dt, physics_specs, aux_features,
        diagnostics_module=diagnostics_module,
        randomness_module=randomness_module)
    inner_dt = dt / num_substeps
    self.num_substeps = num_substeps
    self.corrector_fn = corrector_module(
        coords, inner_dt, physics_specs, aux_features)
    self.physics_parameterization_fn = physics_parameterization_module(
        coords, inner_dt, physics_specs, aux_features)
    self.perturbation_fn = perturbation_module(
        coords, inner_dt, physics_specs, aux_features)
    self.checkpoint_substep = checkpoint_substep
    self.coords = coords

  def finalize_state(
      self,
      x: ModelState,
      forcing: typing.Forcing,
  ) -> ModelState:
    """Finalizes initialization of a model state `x`, encoded from data.

    This method ensures that state has all of the `ModelState` fields
    initialized in a way compatible with this step function. This includes
    populating initial `diagnostics`, `memory` and `randomness` fields.

    This is called by StochasticModularStepModel.encode, after encoding the data

    Args:
      x: Initial values for the model state typically provided by the encoder.
      forcing: Data covariates from the same time slice as `x`.

    Returns:
      Initialized model state.
    """
    # TODO(dkochkov) Consider adding an option of not overriding randomness.
    x.randomness = self.randomness_fn.unconditional_sample(
        hk.maybe_next_rng_key()
    )
    pp_tendency = self.physics_parameterization_fn(
        x.state, x.memory, x.diagnostics, x.randomness.nodal_value, forcing
    )
    x.diagnostics = self.diagnostics_fn(x, pp_tendency, forcing)
    return x

  def __call__(
      self,
      state: ModelState,
      forcing: typing.Forcing,
  ) -> ModelState:
    """Computes the state of the system evolved in time by `dt`."""

    def step_fn(x):
      x = self.coords.with_dycore_sharding(x)
      # TODO(dkochkov) Consider passing `x` to physics_parameterization.
      pp_tendency = self.physics_parameterization_fn(
          x.state, x.memory, x.diagnostics, x.randomness.nodal_value, forcing
      )

      pp_tendency = self.perturbation_fn(
          pp_tendency,
          state=x.state,
          randomness=x.randomness.nodal_value,
      )

      next_state = self.corrector_fn(x.state, pp_tendency, forcing)
      # TODO(dkochkov) update stochastic modules to take optional state.
      next_randomness = self.randomness_fn.advance(x.randomness)
      next_memory = x.state if x.memory is not None else None
      next_diagnostics = self.diagnostics_fn(x, pp_tendency, forcing)
      x_next = ModelState(
          state=next_state, memory=next_memory, diagnostics=next_diagnostics,  # pyrefly: ignore[unexpected-keyword]
          randomness=next_randomness)  # pyrefly: ignore[unexpected-keyword]
      x_next = self.coords.with_dycore_sharding(x_next)
      return x_next

    if self.checkpoint_substep:
      step_fn = hk.remat(step_fn)
    step_fn = time_integration.repeated(step_fn, self.num_substeps, hk.scan)
    return step_fn(state)


# TODO(dkochkov) Move vertical advection step to transforms.py.


@gin.register
class SemiLagrangianVerticalAdvectionStep(hk.Module):
  """Step module that applies vertical advection for the primitive equations."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.dt = dt

  def __call__(self, state):
    return primitive_equations.semi_lagrangian_vertical_advection_step(
        state, self.coords, self.dt
    )
