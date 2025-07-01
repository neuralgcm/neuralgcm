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

"""Modules that parameterize unsimulated atmospheric processes."""

import copy

import coordax as cx
from flax import nnx
from neuralgcm.experimental.atmosphere import transforms as atm_transforms
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import pytree_utils
from neuralgcm.experimental.core import spatial_filters
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing


ShapeFloatStruct = typing.ShapeFloatStruct


class ModalNeuralDivCurlParameterization(nnx.Module):
  """Computes modal tendencies with `u, v` → `δ, ζ` transform."""

  def __init__(
      self,
      *,
      ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
      sigma: coordinates.SigmaLevels,
      surface_field_names: tuple[str, ...],
      volume_field_names: tuple[str, ...],
      features_module: transforms.Transform,
      mapping_factory: transforms.TransformFactory,
      tendency_transform: transforms.Transform,
      modal_filter: spatial_filters.ModalSpatialFilter,
      input_state_shapes: typing.Pytree,
      u_key: str = 'u_component_of_wind',
      v_key: str = 'v_component_of_wind',
      mesh: parallelism.Mesh,
      rngs: nnx.Rngs,
  ):
    output_coords = {}
    uv_fields = set([u_key, v_key])
    div_curl_fields = set(['divergence', 'vorticity'])
    if len(div_curl_fields.intersection(volume_field_names)) != 2:
      raise ValueError('Volume fields must contain `divergence & vorticity`.')

    grid = ylm_transform.nodal_grid
    for name in (set(volume_field_names) | uv_fields) - div_curl_fields:
      output_coords[name] = cx.compose_coordinates(sigma, grid)
    for name in set(surface_field_names):
      output_coords[name] = grid

    input_shapes = features_module.output_shapes(input_state_shapes)
    self.parameterization_mapping = mapping_factory(
        input_shapes=input_shapes,
        targets=output_coords,
        rngs=rngs,
    )
    self.mesh = mesh
    self.features_module = features_module
    self.tendency_transform = tendency_transform
    self.to_div_curl = atm_transforms.ToModalWithDivCurl(ylm_transform)
    self.filter = modal_filter

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    inputs = copy.copy(inputs)  # avoid mutating inputs.
    inputs = parallelism.with_dycore_sharding(self.mesh, inputs)
    features = self.features_module(inputs)
    inputs.pop('time')  # we use inputs to model outputs, but do not need time.
    features = parallelism.with_dycore_to_physics_sharding(self.mesh, features)
    tendencies = self.parameterization_mapping(features)
    tendencies = parallelism.with_physics_to_dycore_sharding(
        self.mesh, tendencies
    )
    tendencies = self.tendency_transform(tendencies)
    modal_tendencies = self.to_div_curl(tendencies)
    modal_tendencies = self.filter.filter_modal(modal_tendencies)
    modal_tendencies = pytree_utils.replace_with_matching_or_default(
        inputs, modal_tendencies, default=0.0
    )
    modal_tendencies = parallelism.with_dycore_sharding(
        self.mesh, modal_tendencies
    )
    return modal_tendencies
