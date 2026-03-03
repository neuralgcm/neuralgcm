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

"""Modules that implement transformations specific to atmospheric models.

Currently this includes both feature-generating transforms and state transforms.
"""

from __future__ import annotations

import coordax as cx
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import diagnostics
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_harmonics
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units


@nnx_compat.dataclass
class ToModalWithDivCurl(transforms.TransformABC):
  """Module that converts inputs to modal replacing velocity with div/curl."""

  ylm_map: spherical_harmonics.FixedYlmMapping
  u_key: str = 'u_component_of_wind'
  v_key: str = 'v_component_of_wind'

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    grid = self.ylm_map.nodal_grid
    mesh = self.ylm_map.mesh
    if self.u_key not in inputs or self.v_key not in inputs:
      raise ValueError(
          f'{(self.u_key, self.v_key)=} not found in {inputs.keys()=}'
      )
    sec_lat = 1 / grid.cos_lat
    u, v = inputs.pop(self.u_key), inputs.pop(self.v_key)
    u, v = parallelism.with_physics_to_dycore_sharding(mesh, (u, v))
    # here u,v stand for velocity / cos(lat), but the cos(lat) is cancelled in
    # divergence and curl operators below.
    inputs[self.u_key] = u * sec_lat
    inputs[self.v_key] = v * sec_lat
    inputs = parallelism.with_dycore_sharding(mesh, inputs)
    modal_outputs = self.ylm_map.to_modal(inputs)
    modal_outputs = parallelism.with_dycore_sharding(mesh, modal_outputs)
    u, v = modal_outputs.pop(self.u_key), modal_outputs.pop(self.v_key)
    modal_outputs['divergence'] = parallelism.with_dycore_sharding(
        mesh, self.ylm_map.div_cos_lat(u, v)
    )
    modal_outputs['vorticity'] = parallelism.with_dycore_sharding(
        mesh, self.ylm_map.curl_cos_lat(u, v)
    )
    return modal_outputs


@nnx_compat.dataclass
class PressureFeatures(transforms.TransformABC):
  """Feature module that computes pressure."""

  ylm_map: spherical_harmonics.FixedYlmMapping
  levels: (
      coordinates.SigmaLevels
      | coordinates.HybridLevels
      | coordinates.PressureLevels
  )
  feature_name: str = 'pressure'
  sim_units: units.SimUnits | None = None

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    if isinstance(self.levels, coordinates.PressureLevels):
      return {self.feature_name: self.levels.fields['pressure']}
    log_surface_p = self.ylm_map.to_nodal(inputs['log_surface_pressure'])
    surface_p = cx.cmap(jnp.exp)(log_surface_p)
    if isinstance(self.levels, coordinates.SigmaLevels):
      sigma = self.levels.fields['sigma']
      pressure = sigma * surface_p  # order matters here to put sigma upfront.
    elif isinstance(self.levels, coordinates.HybridLevels):
      pressure = self.levels.pressure_centers(surface_p, self.sim_units)
    else:
      raise ValueError(f'Unsupported level type: {type(self.levels)}')
    return {self.feature_name: pressure}


class VelocityAndPrognosticsWithModalGradients(transforms.TransformABC):
  """Features module that returns prognostics + u,v and optionally gradients."""

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      surface_field_names: tuple[str, ...] = tuple(),
      volume_field_names: tuple[str, ...] = tuple(),
      compute_gradients_transform: (
          transforms.ToModalWithDerivatives | None
      ) = None,
      inputs_are_modal: bool = True,
      u_key: str = 'u_component_of_wind',
      v_key: str = 'v_component_of_wind',
  ):
    if compute_gradients_transform is None:
      compute_gradients_transform = lambda x: {}
    self.ylm_map = ylm_map
    self.surface_field_names = surface_field_names
    self.volume_field_names = volume_field_names
    self.fields_to_include = surface_field_names + volume_field_names
    self.compute_gradients_transform = compute_gradients_transform
    self.u_key = u_key
    self.v_key = v_key
    if inputs_are_modal:
      self.pre_process = lambda x: x
    else:
      self.pre_process = transforms.ToModal(ylm_map)

  def _extract_features(
      self,
      inputs: dict[str, cx.Field],
      prefix: str = '',
  ) -> dict[str, cx.Field]:
    """Returns a nodal velocity and prognostic features."""
    # Note: all intermediate features have an explicit cos-lat factors in key.
    # These factors are removed in the `__call__` method before returning.
    # compute `u, v` if div/curl is available and `u, v` not in prognosics.
    if set(['vorticity', 'divergence']).issubset(inputs.keys()) and (
        not set([self.u_key, self.v_key]).intersection(inputs.keys())
    ):
      cos_lat_u, cos_lat_v = spherical_harmonics.get_cos_lat_vector(
          inputs['vorticity'], inputs['divergence'], self.ylm_map
      )
      modal_features = {}
      if self.u_key in self.fields_to_include:
        modal_features[typing.KeyWithCosLatFactor(prefix + self.u_key, 1)] = (
            cos_lat_u
        )
      if self.v_key in self.fields_to_include:
        modal_features[typing.KeyWithCosLatFactor(prefix + self.v_key, 1)] = (
            cos_lat_v
        )
    elif self.u_key in inputs and self.v_key in inputs:
      modal_features = {
          typing.KeyWithCosLatFactor(prefix + self.u_key, 0): inputs[
              self.u_key
          ],
          typing.KeyWithCosLatFactor(prefix + self.v_key, 0): inputs[
              self.v_key
          ],
      }
    else:
      modal_features = {}
    prognostics_keys = list(inputs.keys())
    for k in set(self.fields_to_include) - set([self.u_key, self.v_key]):
      if k not in prognostics_keys:
        raise ValueError(f'Requested field {k} not in {prognostics_keys=}.')
      modal_features[typing.KeyWithCosLatFactor(prefix + k, 0)] = inputs[k]

    # Computing gradient features and adjusting cos_lat factors.
    modal_features = parallelism.with_dycore_sharding(
        self.ylm_map.mesh, modal_features
    )
    diff_operator_features = self.compute_gradients_transform(modal_features)
    # Computing all features in nodal space.
    features = {}
    for k, v in (diff_operator_features | modal_features).items():
      value = self.ylm_map.to_nodal(v)
      lat_power = k.factor_order
      sec_lat_scale = (1 / self.ylm_map.cos_lat(value) ** lat_power)
      features[k.name] = value * sec_lat_scale
    features = parallelism.with_dycore_sharding(
        self.ylm_map.mesh, features
    )
    return features

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    inputs = self.pre_process(inputs)
    nodal_features = self._extract_features(inputs)
    return nodal_features


@nnx_compat.dataclass
class ConstrainWaterBudget(transforms.TransformABC):
  """Constrains precipitation or evaporation based on precipitation+evaporation.

  If `observation_key` is precipitation, it constrains it to be positive and
  smaller than precipitation+evaporation if precipitation+evaporation is
  positive. If `observation_key` is evaporation, it constrains it to be negative
  and larger than precipitation+evaporation if precipitation+evaporation is
  negative. Evaporation is assumed to be negative. Precipitation is assumed to
  be positive.

  Attributes:
    p_plus_e_diagnostic: Diagnostics that computes precipitation + evaporation.
    var_to_constrain: Key in inputs to constrain, precipitation or evaporation.
    precipitation_key: Key for precipitation.
    evaporation_key: Key for evaporation.
    p_plus_e_key: Key for precipitation + evaporation in p_plus_e_diagnostic.
  """

  p_plus_e_diagnostic: diagnostics.DiagnosticModule
  var_to_constrain: str
  precipitation_key: str
  evaporation_key: str
  p_plus_e_key: str = 'precipitation_plus_evaporation_rate'

  def __post_init__(self):
    if self.var_to_constrain not in [
        self.precipitation_key,
        self.evaporation_key,
    ]:
      raise ValueError(
          f'{self.var_to_constrain=} should be either'
          f' {self.precipitation_key=} or {self.evaporation_key=}.'
      )

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    """Applies constraint."""
    diag_values = self.p_plus_e_diagnostic.diagnostic_values()
    if self.p_plus_e_key not in diag_values:
      raise ValueError(f'{self.p_plus_e_key} not in {diag_values.keys()=}')
    p_plus_e = diag_values[self.p_plus_e_key]
    if self.var_to_constrain not in inputs:
      raise ValueError(f'{self.var_to_constrain} not in {inputs.keys()=}')
    observation = inputs[self.var_to_constrain]
    if self.var_to_constrain == self.precipitation_key:
      diagnosed_key = self.evaporation_key
      constrained_observation = cx.cmap(
          lambda x, a, b: jnp.maximum(x, jnp.maximum(a, b))
      )(observation, p_plus_e, 0)
      precipitation_and_evaporation = {
          self.var_to_constrain: constrained_observation,
          diagnosed_key: p_plus_e - constrained_observation,
      }
    elif self.var_to_constrain == self.evaporation_key:
      diagnosed_key = self.precipitation_key
      constrained_observation = cx.cmap(
          lambda x, a, b: jnp.minimum(x, jnp.minimum(a, b))
      )(observation, p_plus_e, 0)
      precipitation_and_evaporation = {
          self.var_to_constrain: constrained_observation,
          diagnosed_key: p_plus_e - constrained_observation,
      }
    else:
      raise ValueError(
          f'{self.var_to_constrain=} should be either'
          f' {self.precipitation_key=} or {self.evaporation_key=}.'
      )
    outputs = inputs.copy()
    outputs.update(precipitation_and_evaporation)
    return outputs
