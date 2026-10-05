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
"""ML modules for equation-based models."""

from typing import Any, Callable, Optional, Sequence, Union
from dinosaur import coordinate_systems
from dinosaur import held_suarez
from dinosaur import primitive_equations
from dinosaur import pytree_utils
from dinosaur import scales
from dinosaur import shallow_water
from dinosaur import sigma_coordinates
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import xarray_utils
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm.legacy import features
from neuralgcm.legacy import mappings
from neuralgcm.legacy import orographies
from neuralgcm.legacy import parameterizations

units = scales.units
SCALE = scales.DEFAULT_SCALE
QuantityOrStr = Union[str, scales.Quantity]
EquationModule = Callable[..., time_integration.ImplicitExplicitODE]
TransformModule = typing.TransformModule
FeaturesModule = features.FeaturesModule
OrographyModule = orographies.OrographyModule
MappingModule = mappings.MappingModule
StepFilterModule = Callable[..., typing.PyTreeStepFilterFn]

REF_TEMP_KEY = xarray_utils.REF_TEMP_KEY
REF_POTENTIAL_KEY = xarray_utils.REF_POTENTIAL_KEY
OROGRAPHY = xarray_utils.OROGRAPHY


@gin.register
class ShallowWaterEquations(shallow_water.ShallowWaterEquations):
  """Equation module for shallow water system."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: shallow_water.ShallowWaterSpecs,
      aux_features: typing.AuxFeatures,
      orography_module: OrographyModule = orographies.ClippedOrography,
      name: Optional[str] = None,
  ):
    reference_potential = aux_features.get(REF_POTENTIAL_KEY, None)
    if reference_potential is None:
      raise ValueError(f'must supply {REF_POTENTIAL_KEY} in `aux_features`.')
    modal_orography_init_fn = orography_module(
        coords, dt, physics_specs, aux_features)
    modal_orography = modal_orography_init_fn()  # pyrefly: ignore[not-callable]
    super().__init__(
        coords=coords,
        physics_specs=physics_specs,  # pyrefly: ignore[bad-argument-type]
        orography=modal_orography,
        reference_potential=reference_potential,
    )


@gin.register
class PrimitiveEquations(primitive_equations.PrimitiveEquations):
  """Equation module for primitive equations."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: primitive_equations.PrimitiveEquationsSpecs,
      aux_features: typing.AuxFeatures,
      orography_module: OrographyModule = orographies.ClippedOrography,
      vertical_advection: Callable[..., jax.Array] = (
          sigma_coordinates.centered_vertical_advection
      ),
      include_vertical_advection: bool = True,
      name: Optional[str] = None,
  ):
    ref_temperatures = aux_features.get(REF_TEMP_KEY, None)
    if ref_temperatures is None:
      raise ValueError(f'must supply {REF_TEMP_KEY} in `aux_features`.')
    modal_orography_init_fn = orography_module(
        coords, dt, physics_specs, aux_features)
    modal_orography = modal_orography_init_fn()  # pyrefly: ignore[not-callable]
    super().__init__(
        coords=coords,
        physics_specs=physics_specs,  # pyrefly: ignore[bad-argument-type]
        reference_temperature=ref_temperatures,
        orography=modal_orography,
        vertical_advection=vertical_advection,
        include_vertical_advection=include_vertical_advection,
    )


@gin.register
class PrimitiveEquationsWithTime(
    primitive_equations.PrimitiveEquationsWithTime
):
  """Equation module for primitive equations."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: primitive_equations.PrimitiveEquationsSpecs,
      aux_features: typing.AuxFeatures,
      orography_module: OrographyModule = orographies.ClippedOrography,
      vertical_advection: Callable[..., jax.Array] = (
          sigma_coordinates.centered_vertical_advection
      ),
      include_vertical_advection: bool = True,
      name: Optional[str] = None,
  ):
    ref_temperatures = aux_features.get(REF_TEMP_KEY, None)
    if ref_temperatures is None:
      raise ValueError(f'must supply {REF_TEMP_KEY} in `aux_features`.')
    modal_orography_init_fn = orography_module(
        coords, dt, physics_specs, aux_features)
    modal_orography = modal_orography_init_fn()  # pyrefly: ignore[not-callable]
    super().__init__(
        coords=coords,
        physics_specs=physics_specs,  # pyrefly: ignore[bad-argument-type]
        reference_temperature=ref_temperatures,
        orography=modal_orography,
        vertical_advection=vertical_advection,
        include_vertical_advection=include_vertical_advection,
    )


@gin.register
class MoistPrimitiveEquations(
    primitive_equations.MoistPrimitiveEquations
):
  """Equation module for moist primitive equations."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: primitive_equations.PrimitiveEquationsSpecs,
      aux_features: typing.AuxFeatures,
      orography_module: OrographyModule = orographies.ClippedOrography,
      vertical_advection: Callable[..., jax.Array] = (
          sigma_coordinates.centered_vertical_advection
      ),
      include_vertical_advection: bool = True,
      name: Optional[str] = None,
  ):
    ref_temperatures = aux_features.get(REF_TEMP_KEY, None)
    if ref_temperatures is None:
      raise ValueError(f'must supply {REF_TEMP_KEY} in `aux_features`.')
    modal_orography_init_fn = orography_module(
        coords, dt, physics_specs, aux_features)
    modal_orography = modal_orography_init_fn()  # pyrefly: ignore[not-callable]
    super().__init__(
        coords=coords,
        physics_specs=physics_specs,  # pyrefly: ignore[bad-argument-type]
        reference_temperature=ref_temperatures,
        orography=modal_orography,
        vertical_advection=vertical_advection,
        include_vertical_advection=include_vertical_advection,
    )


@gin.register
class MoistPrimitiveEquationsWithCloudMoisture(
    primitive_equations.MoistPrimitiveEquationsWithCloudMoisture
):
  """Equation module for moist primitive equations with clouds."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: primitive_equations.PrimitiveEquationsSpecs,
      aux_features: typing.AuxFeatures,
      orography_module: OrographyModule = orographies.ClippedOrography,
      vertical_advection: Callable[..., jax.Array] = (
          sigma_coordinates.centered_vertical_advection
      ),
      include_vertical_advection: bool = True,
      name: Optional[str] = None,
  ):
    ref_temperatures = aux_features.get(REF_TEMP_KEY, None)
    if ref_temperatures is None:
      raise ValueError(f'must supply {REF_TEMP_KEY} in `aux_features`.')
    modal_orography_init_fn = orography_module(
        coords, dt, physics_specs, aux_features)
    modal_orography = modal_orography_init_fn()  # pyrefly: ignore[not-callable]
    super().__init__(
        coords=coords,
        physics_specs=physics_specs,  # pyrefly: ignore[bad-argument-type]
        reference_temperature=ref_temperatures,
        orography=modal_orography,
        vertical_advection=vertical_advection,
        include_vertical_advection=include_vertical_advection,
    )


@gin.register
class MoistPrimitiveEquationsWithCloudMoisutre(
    MoistPrimitiveEquationsWithCloudMoisture
):
  """Temporary alias with mis-spelled name."""


@gin.register
class HeldSuarezEquations(held_suarez.HeldSuarezForcing):
  """Equation module for Held-Suarez forcing equations."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: primitive_equations.PrimitiveEquationsSpecs,
      aux_features: typing.AuxFeatures,
      name: Optional[str] = None,
  ):
    ref_temperatures = aux_features.get(REF_TEMP_KEY, None)
    if ref_temperatures is None:
      raise ValueError(f'must supply {REF_TEMP_KEY} in `aux_features`.')
    super().__init__(
        coords=coords,
        physics_specs=physics_specs,  # pyrefly: ignore[bad-argument-type]
        reference_temperature=ref_temperatures)


# TODO(dkochkov) Test if vertical diffusion works well with euler integrator.


@gin.register
class VerticalDiffusion(time_integration.ExplicitODE):
  """Equation module that adds explicit diffusion along vertical direction."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      timescale: QuantityOrStr = gin.REQUIRED,
  ):
    self.coords = coords
    timescale = dt / physics_specs.nondimensionalize(scales.Quantity(timescale))
    timescales = coords.vertical.boundaries * timescale  # pyrefly: ignore[missing-attribute]
    self.level_weighted_timescales = timescales[:, jnp.newaxis, jnp.newaxis]

  def explicit_terms(self, state: typing.PyTreeState) -> typing.PyTreeState:
    def vertical_diffusion_fn(x: typing.Array) -> typing.Array:
      # TODO(dkochkov) Consider using sigma_coordinates.centered_difference.
      x_grad = x[1:, ...] - x[:-1, ...]
      # padding with zero values for vertical fluxes.
      pad_width = ((1, 1), (0, 0), (0, 0))
      x_grad = jnp.pad(x_grad, pad_width)
      fluxes = self.level_weighted_timescales * x_grad
      # TODO(dkochkov) Consider using sigma_coordinates.centered_difference.
      return fluxes[1:, ...] - fluxes[:-1, ...]

    nodal_state = self.coords.horizontal.to_nodal(state)
    nodal_tendency = pytree_utils.tree_map_where(
        condition_fn=lambda x: jnp.asarray(x).shape == self.coords.nodal_shape,  # pyrefly: ignore[bad-argument-type]
        f=vertical_diffusion_fn,
        g=jnp.zeros_like,
        x=nodal_state)
    modal_tendency = self.coords.horizontal.to_modal(nodal_tendency)
    return self.coords.horizontal.clip_wavenumbers(modal_tendency)


@gin.register
class NoDynamics(time_integration.ImplicitExplicitODE):
  """The constant ODE, ∂u/∂t = 0."""

  def __init__(self, *args, **kwargs):
    del args, kwargs

  def explicit_terms(self, x: typing.PyTreeState) -> typing.PyTreeState:
    return 0 * x  # pyrefly: ignore[bad-return, unsupported-operation]

  def implicit_terms(self, x: typing.PyTreeState) -> typing.PyTreeState:
    return 0 * x  # pyrefly: ignore[bad-return, unsupported-operation]

  def implicit_inverse(
      self, x: typing.PyTreeState, time_step: float
  ) -> typing.PyTreeState:
    return x


@gin.register
def composed_equations_module(
    coords: coordinate_systems.CoordinateSystem,
    dt: float,
    physics_specs: Any,
    aux_features: typing.AuxFeatures,
    equation_modules: Sequence[EquationModule],
) -> time_integration.ImplicitExplicitODE:
  """Returns an equation module that represents a composition of equations."""
  equations = tuple(eq(coords, dt, physics_specs, aux_features)
                    for eq in equation_modules)
  return time_integration.compose_equations(equations)


@gin.register
class DirectNeuralEquations(hk.Module, time_integration.ExplicitODE):
  """Computes explicit tendencies for the input state.

  This equation module predicts tendencies directly in the nodal representation
  and returns values transformed back to the modal space. The nodal tendencies
  are computed by the `nodal_mapping_module` from preprocessed nodal features
  computed by `modal_to_nodal_features_module` followed by the
  `tendency_transform_module`.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      modal_to_nodal_features_module: FeaturesModule,
      nodal_mapping_module: mappings.MappingModule,
      tendency_transform_module: TransformModule,
      prediction_mask: Optional[typing.Pytree] = None,
      filter_module: Optional[StepFilterModule] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.parameterization_fn = parameterizations.DirectNeuralParameterization(
        coords=coords,
        dt=dt,
        physics_specs=physics_specs,
        aux_features=aux_features,
        modal_to_nodal_features_module=modal_to_nodal_features_module,
        nodal_mapping_module=nodal_mapping_module,
        tendency_transform_module=tendency_transform_module,
        prediction_mask=prediction_mask,
        filter_module=filter_module,
        name=name,
    )

  def explicit_terms(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    modal_tendencies = self.parameterization_fn(inputs, forcing=None)
    modal_tendencies = pytree_utils.none_to_zeros(modal_tendencies, inputs)
    return modal_tendencies


@gin.register
class DivCurlNeuralEquations(hk.Module, time_integration.ExplicitODE):
  """Computes explicit tendencies using div and curl operators for `u, v` terms.

  This equation module predicts tendencies of the inputs with velocity-based
  parameterization of the `divergence` and `vorticity` components. Specifically,
  we replace predictions of `divergence` and `vorticity` by nodal predictions
  of `u`, and `v`, which are then differentiated using modal representation.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      modal_to_nodal_features_module: FeaturesModule,
      nodal_mapping_module: mappings.MappingModule,
      tendency_transform_module: TransformModule,
      prediction_mask: Optional[typing.Pytree] = None,
      filter_module: Optional[StepFilterModule] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.parameterization_fn = parameterizations.DivCurlNeuralParameterization(
        coords=coords,
        dt=dt,
        physics_specs=physics_specs,
        aux_features=aux_features,
        modal_to_nodal_features_module=modal_to_nodal_features_module,
        nodal_mapping_module=nodal_mapping_module,
        tendency_transform_module=tendency_transform_module,
        prediction_mask=prediction_mask,
        filter_module=filter_module,
        name=name,
    )

  def explicit_terms(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    modal_tendencies = self.parameterization_fn(inputs, forcing=None)
    modal_tendencies = pytree_utils.none_to_zeros(modal_tendencies, inputs)
    return modal_tendencies
