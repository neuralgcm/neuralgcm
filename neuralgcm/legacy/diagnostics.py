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
"""Defines `diagnostic` modules that compute diagnostic predictions."""

from collections import abc
from typing import Any, Callable, Optional, Protocol

from dinosaur import coordinate_systems
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import typing

import gin
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


TransformModule = typing.TransformModule

PRECIPITATION = 'precipitation'
EVAPORATION = 'evaporation'


class DiagnosticFn(Protocol):
  """Implements initialization and computation of model diagnostic fields."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
  ):
    del coords, dt, physics_specs, aux_features

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> dict[str, jax.Array]:
    """Computes diagnostic field from `model_state` and `physics_tendencies`."""
    ...


DiagnosticModule = Callable[..., DiagnosticFn]


@gin.register
class NoDiagnostics:
  """Diagnostic module that computes no diagnostics."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
  ):
    del coords, dt, physics_specs, aux_features

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> dict[str, jax.Array]:
    return {}


@gin.register
class CombinedDiagnostics:
  """Computes a combination of multiple diagnostics."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
      diagnostic_modules: abc.Sequence[DiagnosticModule] = gin.REQUIRED,  # pyrefly: ignore[bad-function-definition]
  ):
    self.diagnostic_fns = [
        module(coords, dt, physics_specs, aux_features)
        for module in diagnostic_modules
    ]

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> dict[str, jax.Array]:
    diagnostics = {}
    for fn in self.diagnostic_fns:
      new_diagnostics = fn(model_state, physics_tendencies, forcing)
      if any(k in diagnostics for k in new_diagnostics):
        raise ValueError(
            f'{new_diagnostics.keys()} overlaps with {diagnostics.keys()}'
        )
      diagnostics.update(new_diagnostics)
    return diagnostics


@gin.register
class PrecipitationMinusEvaporationDiagnostics:
  """Computes `P-E` by integrating physics_tendencies.

  Depending on the `method` computes either precipitation minus evaporation
  rate, which in ERA5 has units `kg m**-2 s**-1` or time-accumulated value
  in `kg m**-2` if `method == cumulative`.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
      moisture_species: tuple[str, ...] = (
          'specific_humidity',
          'specific_cloud_ice_water_content',
          'specific_cloud_liquid_water_content',
      ),
      method: str = 'rate',
  ):
    del aux_features
    self.coords = coords
    self.dt = dt
    self.physics_specs = physics_specs
    self.moisture_species = moisture_species
    self.method = method
    self.to_nodal_fn = coords.horizontal.to_nodal

  def _compute_evaporation_minus_precipitation(
      self, model_state: typing.ModelState, physics_tendencies: typing.Pytree
  ) -> typing.Array:
    """Computes evaporation minus precipitation."""
    lsp = model_state.state.log_surface_pressure
    p_surface = jnp.squeeze(jnp.exp(self.to_nodal_fn(lsp)), axis=0)
    moisture_tendencies = [
        v
        for tracer, v in physics_tendencies.tracers.items()
        if tracer in self.moisture_species
    ]
    moisture_tendencies = sum(self.to_nodal_fn(moisture_tendencies))
    scale = p_surface / self.physics_specs.g
    e_minus_p = scale * sigma_coordinates.sigma_integral(
        moisture_tendencies, self.coords.vertical, keepdims=False  # pyrefly: ignore[bad-argument-type]
    )
    return e_minus_p

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> typing.Pytree:
    """Computes precipitation minus evaporation."""
    del forcing  # unused
    e_minus_p = self._compute_evaporation_minus_precipitation(
        model_state, physics_tendencies
    )
    if self.method == 'rate':
      return {'P_minus_E_rate': -e_minus_p}
    elif self.method == 'cumulative':
      # TODO(dkochkov) Address possible precision loss due to small deltas.
      surface_nodal_shape = self.coords.horizontal.nodal_shape
      previous = model_state.diagnostics.get(
          'P_minus_E_cumulative',
          jnp.zeros(surface_nodal_shape))
      return {'P_minus_E_cumulative': previous - (e_minus_p * self.dt)}
    else:
      raise ValueError(f'Unknown {self.method=}, must be `rate`/`cumulative`')


@gin.register
class PrecipitableWaterDiagnostics:
  """Computes cumulative preciptable water in the state."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
      moisture_species: tuple[str, ...] = (
          'specific_humidity',
          'specific_cloud_ice_water_content',
          'specific_cloud_liquid_water_content',
      ),
  ):
    del dt, aux_features
    self.coords = coords
    self.physics_specs = physics_specs
    self.moisture_species = moisture_species
    self.to_nodal_fn = coords.horizontal.to_nodal

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> typing.Pytree:
    """Computes preciptable water."""
    del physics_tendencies, forcing  # unused
    lsp = model_state.state.log_surface_pressure
    p_surface = jnp.squeeze(jnp.exp(self.to_nodal_fn(lsp)), axis=0)
    moisture_tracers = [
        v
        for tracer, v in model_state.tracers.items()  # pyrefly: ignore[missing-attribute]
        if tracer in self.moisture_species
    ]
    moisture = sum(self.to_nodal_fn(moisture_tracers))
    water_density = self.physics_specs.nondimensionalize(scales.WATER_DENSITY)
    scale = p_surface / (self.physics_specs.g * water_density)
    water = scale * sigma_coordinates.sigma_integral(
        moisture, self.coords.vertical, keepdims=False  # pyrefly: ignore[bad-argument-type]
    )
    return {'precipitable_water': water}


@gin.register
class NodalModelDiagnosticsDecoder:
  """Diagnostics decoder that returns elements from model_state.diagnostics."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
  ):
    del dt, aux_features
    self.coords = coords
    self.physics_specs = physics_specs

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> typing.Pytree:
    """Computes precipitation minus evaporation."""
    del physics_tendencies, forcing  # unused.
    nodal_diagnostics = coordinate_systems.maybe_to_nodal(
        model_state.diagnostics, self.coords
    )
    return nodal_diagnostics


# TODO(janniyuval) add a decoder that can add some Gaussian noise to evap/precip
@gin.register
class PrecipitationDiagnosticsConstrained(
    hk.Module, PrecipitationMinusEvaporationDiagnostics
):
  """Predict evaporation and computes cumulative precipitation.

  Calculation is based on calculating `P-E` by integrating physics_tendencies.
  Depending on the `method` computes either precipitation
  rate, (which in ERA5 has units `kg m**-2 s**-1`) or time-accumulated value
  in `Length` units (GPCP uses mm/day) if `method == cumulative`.
  Evaporation has the units of `kg m**-2 s**-1` in ERA5.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
      embedding_module: typing.EmbeddingModule,
      moisture_species: tuple[str, ...] = (
          'specific_humidity',
          'specific_cloud_ice_water_content',
          'specific_cloud_liquid_water_content',
      ),
      is_precipitation: bool = True,
      method_precipitation: str = 'cumulative',
      method_evaporation: str = 'rate',
      name: Optional[str] = None,
      field_name: str = 'total_precipitation',
  ):
    # del aux_features
    super().__init__(name=name)
    self.coords = coords
    self.dt = dt
    self.physics_specs = physics_specs
    self.moisture_species = moisture_species
    self.method_precipitation = method_precipitation
    self.method_evaporation = method_evaporation
    self.to_nodal_fn = coords.horizontal.to_nodal
    self.is_precipitation = is_precipitation
    if self.is_precipitation:
      predicted_name = PRECIPITATION
      diagnosed_name = EVAPORATION
    else:
      predicted_name = EVAPORATION
      diagnosed_name = PRECIPITATION

    self.predicted_name = predicted_name
    self.diagnosed_name = diagnosed_name

    output_shapes = {
        f'{predicted_name}': np.asarray(coords.surface_nodal_shape)
    }

    self.embedding_fn = embedding_module(
        coords, dt, physics_specs, aux_features, output_shapes=output_shapes
    )
    self.water_density = self.physics_specs.nondimensionalize(
        scales.WATER_DENSITY
    )
    self.field_name = field_name

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> typing.Pytree:
    """Computes precipitation minus evaporation."""
    e_minus_p = self._compute_evaporation_minus_precipitation(
        model_state, physics_tendencies
    )
    water_budget = self.embedding_fn(
        model_state.state,
        model_state.memory,
        model_state.diagnostics,
        model_state.randomness,
        forcing,
    )
    water_budget[self.diagnosed_name] = (
        -e_minus_p - water_budget[self.predicted_name]
    )

    # Note: In ERA5 mean_evaporation_rate (kg m**-2 s**-1)
    # is negative for evaporation.
    # In GPCP precipitation is positive (mm/day).
    # Here e_minus_p is positive for evaporation.
    output_dict = {}
    surface_nodal_shape = self.coords.horizontal.nodal_shape
    if self.method_precipitation == 'rate':  # units: length/time
      output_dict[PRECIPITATION + '_rate'] = (
          water_budget[PRECIPITATION]
      ) / self.water_density
    elif self.method_precipitation == 'cumulative':  # units: length
      previous = model_state.diagnostics.get(
          self.field_name, jnp.zeros(surface_nodal_shape)
      )
      # TODO(janniyuval) remove precipitation_cumulative_mean once no models
      # use it.
      assert self.field_name in [
          'total_precipitation',
          'precipitation_cumulative_mean',
      ], self.field_name
      output_dict[self.field_name] = previous + (
          (water_budget[PRECIPITATION] / self.water_density) * self.dt
      )
    else:
      raise ValueError(
          f'Precipitation method is {self.method_precipitation=}, but it must'
          ' be `rate`/`cumulative`'
      )
    if self.method_evaporation == 'rate':  # units: mass length**-2 time**-1
      output_dict[EVAPORATION] = water_budget[EVAPORATION]
    elif self.method_evaporation == 'cumulative':  # units: length
      previous_evap = model_state.diagnostics.get(
          EVAPORATION + '_cumulative', jnp.zeros(surface_nodal_shape)
      )
      output_dict[EVAPORATION + '_cumulative'] = (
          previous_evap
          + (water_budget[EVAPORATION] / self.water_density) * self.dt
      )
    else:
      raise ValueError(
          f'Evaporation method is {self.method_evaporation=},  but it must be'
          ' `rate`/`cumulative`'
      )
    return output_dict


@gin.register
class SurfacePressureDiagnostics:
  """Getting the surface pressure of the state."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
  ):
    del dt, aux_features, physics_specs
    self.to_nodal_fn = coords.horizontal.to_nodal

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> typing.Pytree:
    """Computes surface pressure."""
    del physics_tendencies, forcing  # unused
    lsp = model_state.state.log_surface_pressure
    surface_pressure = jnp.squeeze(jnp.exp(self.to_nodal_fn(lsp)), axis=0)
    return {'surface_pressure': surface_pressure}
