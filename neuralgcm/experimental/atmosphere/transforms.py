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

import functools

import coordax as cx
from dinosaur import spherical_harmonic
import jax.numpy as jnp
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing


@nnx_compat.dataclass
class ToModalWithDivCurl(transforms.TransformABC):
  """Module that converts inputs to modal replacing velocity with div/curl."""

  ylm_transform: spherical_transforms.SphericalHarmonicsTransform
  u_key: str = 'u_component_of_wind'
  v_key: str = 'v_component_of_wind'

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    grid = self.ylm_transform.nodal_grid
    ylm_grid = self.ylm_transform.modal_grid
    # TODO(dkochkov): Consider doing a strict check rather than filtering.
    inputs = transforms.filter_fields_by_coordinate(inputs, grid)
    dinosaur_grid = self.ylm_transform.dinosaur_grid
    mesh = self.ylm_transform.mesh
    if self.u_key not in inputs or self.v_key not in inputs:
      raise ValueError(
          f'{(self.u_key, self.v_key)=} not found in {inputs.keys()=}'
      )
    sec_lat = cx.wrap(1 / dinosaur_grid.cos_lat, grid.axes[1])
    u, v = inputs.pop(self.u_key), inputs.pop(self.v_key)
    u, v = parallelism.with_physics_to_dycore_sharding(mesh, (u, v))
    # here u,v stand for velocity / cos(lat), but the cos(lat) is cancelled in
    # divergence and curl operators below.
    inputs[self.u_key] = u * sec_lat
    inputs[self.v_key] = v * sec_lat
    inputs = parallelism.with_dycore_sharding(mesh, inputs)
    modal_outputs = self.ylm_transform.to_modal(inputs)
    modal_outputs = parallelism.with_dycore_sharding(mesh, modal_outputs)
    u, v = modal_outputs.pop(self.u_key), modal_outputs.pop(self.v_key)
    u, v = cx.untag((u, v), ylm_grid)
    modal_outputs['divergence'] = parallelism.with_dycore_sharding(
        mesh,
        cx.cmap(dinosaur_grid.div_cos_lat, out_axes=u.named_axes)((u, v)).tag(
            ylm_grid
        ),
    )
    modal_outputs['vorticity'] = parallelism.with_dycore_sharding(
        mesh,
        cx.cmap(dinosaur_grid.curl_cos_lat, out_axes=u.named_axes)((u, v)).tag(
            ylm_grid
        ),
    )
    return modal_outputs


@nnx_compat.dataclass
class PressureOnSigmaFeatures(transforms.TransformABC):
  """Feature module that computes pressure."""

  ylm_transform: spherical_transforms.SphericalHarmonicsTransform
  sigma: coordinates.SigmaLevels
  feature_name: str = 'pressure'

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    log_surface_p = self.ylm_transform.to_nodal(inputs['log_surface_pressure'])
    sigma = self.sigma.fields['sigma']
    surface_p = cx.cmap(jnp.exp)(log_surface_p)
    pressure = sigma * surface_p  # order matters here to put sigma upfront.
    return {self.feature_name: pressure}


class VelocityAndPrognosticsWithModalGradients(transforms.TransformABC):
  """Features module that returns prognostics + u,v and optionally gradients."""

  def __init__(
      self,
      ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
      surface_field_names: tuple[str, ...] = tuple(),
      volume_field_names: tuple[str, ...] = tuple(),
      compute_gradients_transform: (
          transforms.ToModalWithFilteredGradients | None
      ) = None,
      inputs_are_modal: bool = True,
      u_key: str = 'u_component_of_wind',
      v_key: str = 'v_component_of_wind',
  ):
    if compute_gradients_transform is None:
      compute_gradients_transform = lambda x: {}
    self.ylm_transform = ylm_transform
    self.surface_field_names = surface_field_names
    self.volume_field_names = volume_field_names
    self.fields_to_include = surface_field_names + volume_field_names
    self.compute_gradients_transform = compute_gradients_transform
    self.u_key = u_key
    self.v_key = v_key
    if inputs_are_modal:
      self.pre_process = lambda x: x
    else:
      self.pre_process = transforms.ToModal(ylm_transform)

  def _extract_features(
      self,
      inputs: dict[str, cx.Field],
      prefix: str = '',
  ) -> dict[str, cx.Field]:
    """Returns a nodal velocity and prognostic features."""
    # Note: all intermediate features have an explicit cos-lat factors in key.
    # These factors are removed in the `__call__` method before returning.

    ylm_grid = self.ylm_transform.modal_grid
    dinosaur_grid = self.ylm_transform.dinosaur_grid
    # compute `u, v` if div/curl is available and `u, v` not in prognosics.
    if set(['vorticity', 'divergence']).issubset(inputs.keys()) and (
        not set([self.u_key, self.v_key]).intersection(inputs.keys())
    ):
      vorticity, divergence = inputs['vorticity'], inputs['divergence']
      vorticity, divergence = cx.untag((vorticity, divergence), ylm_grid)
      cos_lat_fn = functools.partial(
          spherical_harmonic.get_cos_lat_vector,
          grid=dinosaur_grid,
      )
      cos_lat_fn = cx.cmap(cos_lat_fn, out_axes=vorticity.named_axes)
      cos_lat_u, cos_lat_v = cx.tag(cos_lat_fn(vorticity, divergence), ylm_grid)
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
        self.ylm_transform.mesh, modal_features
    )
    lat_axis = self.ylm_transform.nodal_grid.axes[1]
    diff_operator_features = self.compute_gradients_transform(modal_features)
    sec_lat = cx.wrap(1 / dinosaur_grid.cos_lat, lat_axis)
    sec2_lat = cx.wrap(dinosaur_grid.sec2_lat, lat_axis)
    sec_lat_scales = {0: 1, 1: sec_lat, 2: sec2_lat}
    # Computing all features in nodal space.
    features = {}
    for k, v in (diff_operator_features | modal_features).items():
      sec_lat_scale = sec_lat_scales[k.factor_order]
      features[k.name] = self.ylm_transform.to_nodal(v) * sec_lat_scale
    features = parallelism.with_dycore_sharding(
        self.ylm_transform.mesh, features
    )
    return features

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    inputs = self.pre_process(inputs)
    ylm_grid = self.ylm_transform.modal_grid
    # TODO(dkochkov): Consider doing a strict check rather than filtering.
    filtered_inputs = transforms.filter_fields_by_coordinate(inputs, ylm_grid)
    nodal_features = self._extract_features(filtered_inputs)
    return nodal_features
