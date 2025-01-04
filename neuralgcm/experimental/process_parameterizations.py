# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules that parameterize unsimulated processes."""

from typing import Callable

from flax import nnx
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import pytree_mappings
from neuralgcm.experimental import pytree_transforms
from neuralgcm.experimental import pytree_utils
from neuralgcm.experimental import spatial_filters
from neuralgcm.experimental import typing


ShapeFloatStruct = typing.ShapeFloatStruct


class ModalNeuralDivCurlParameterization(nnx.Module):
  """Computes modal tendencies with `u, v` → `δ, ζ` transform."""

  def __init__(
      self,
      *,
      coords: coordinates.DinosaurCoordinates,
      surface_field_names: tuple[str, ...],
      volume_field_names: tuple[str, ...],
      features_module: pytree_transforms.Transform,
      mapping_factory: Callable[
          ..., pytree_mappings.ChannelMapping | pytree_mappings.VariableMapping
      ],
      tendency_transform: pytree_transforms.Transform,
      modal_filter: spatial_filters.ModalSpatialFilter,
      input_state_shapes: typing.Pytree | None = None,
      u_key: str = 'u_component_of_wind',
      v_key: str = 'v_component_of_wind',
      rngs: nnx.Rngs,
  ):
    output_shapes = {}
    # TODO(dkochkov): Add checks for the field names that are required.
    uv_fields = set([u_key, v_key])
    div_curl_fields = set(['divergence', 'vorticity'])
    if len(div_curl_fields.intersection(volume_field_names)) != 2:
      raise ValueError('Volume fields must contain `divergence & vorticity`.')
    layers = coords.vertical.shape[0]
    dinosaur_grid = coords.dinosaur_grid
    # TODO(dkochkov): Compute these using coords modifications.
    for name in (set(volume_field_names) | uv_fields) - div_curl_fields:
      output_shapes[name] = ShapeFloatStruct(
          (layers,) + dinosaur_grid.nodal_shape
      )
    for name in set(surface_field_names):
      output_shapes[name] = ShapeFloatStruct((1,) + dinosaur_grid.nodal_shape)
    if input_state_shapes is None:
      input_state_shapes = pytree_mappings.MINIMAL_STATE_STRUCT  # default.
    input_shapes = features_module.output_shapes(input_state_shapes)
    self.parameterization_mapping = mapping_factory(
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        rngs=rngs,
    )
    self.features_module = features_module
    self.tendency_transform = tendency_transform
    self.to_div_curl = pytree_transforms.ToModalWithDivCurl(coords.horizontal)
    self.filter = modal_filter

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    # inputs = self.coords.with_dycore_sharding(inputs)
    inputs_dict, from_dict_fn = pytree_utils.as_dict(inputs)
    features = self.features_module(inputs_dict)
    # features = self.coords.dycore_to_physics_sharding(features)
    tendencies = self.parameterization_mapping(features)
    # tendencies = self.coords.physics_to_dycore_sharding(tendencies)
    tendencies = self.tendency_transform(tendencies)
    modal_tendencies = self.to_div_curl(tendencies)
    modal_tendencies = self.filter.filter_modal(modal_tendencies)
    modal_tendencies = pytree_utils.replace_with_matching_or_default(
        inputs_dict, modal_tendencies, default=0.0
    )
    outputs = from_dict_fn(modal_tendencies)
    # outputs = self.coords.with_dycore_sharding(outputs)
    return outputs
