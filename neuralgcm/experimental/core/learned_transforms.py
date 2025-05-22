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

"""Transforms that are parameterized by learnable parameters like NN."""

import dataclasses

from flax import nnx
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import field_utils
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import towers
from neuralgcm.experimental.core import transforms


@nnx_compat.dataclass
class UnaryFieldTowerTransform(transforms.TransformABC, nnx.Module):
  """Transforms fields with UnaryFieldTower and splits the output to fields."""

  coords: dict[str, cx.Coordinate]
  tower: towers.UnaryFieldTower
  field_combiner: field_utils.FieldCombiner
  in_transform: transforms.Transform = transforms.Identity()
  out_transform: transforms.Transform = transforms.Identity()
  feature_sharding_schema: str | None = None
  result_sharding_schema: str | None = None
  mesh: parallelism.Mesh = dataclasses.field(kw_only=True)

  def __call__(self, inputs):
    apply_sharding = self.mesh.with_sharding_constraint
    in_features = self.in_transform(inputs)
    in_features = apply_sharding(in_features, self.feature_sharding_schema)
    in_field = self.field_combiner(in_features)
    in_field = in_field.tag(self.tower.net_in_dims[0])
    out_field = self.tower(in_field)
    out_fields = field_utils.split_to_fields(out_field, self.coords)
    out_fields = apply_sharding(out_fields, self.result_sharding_schema)
    return self.out_transform(out_fields)

  @classmethod
  def build_using_factories(
      cls,
      input_shapes,
      coords,
      tower_factory,
      field_combiner,
      in_transform=transforms.Identity(),
      out_transform=transforms.Identity(),
      *,
      mesh: parallelism.Mesh,
      rngs,
  ):
    in_shapes = in_transform.output_shapes(input_shapes)
    in_field_shape = nnx.eval_shape(field_combiner, in_shapes)
    out_shapes = field_utils.shape_struct_fields_from_coords(coords)
    out_field_shape = nnx.eval_shape(field_combiner, out_shapes)
    input_size = in_field_shape.positional_shape[0]
    output_size = out_field_shape.positional_shape[0]
    tower = tower_factory(input_size, output_size, rngs=rngs)
    return cls(
        coords, tower, field_combiner, in_transform, out_transform, mesh=mesh
    )


@nnx_compat.dataclass
class WeightedLandSeaIceTowersTransform(transforms.TransformABC, nnx.Module):
  """Combines FieldTowerTransformsTransforms for landd, sea and sea ice.

  Outputs are computed by evaluating UnaryFieldTowerTransforms for each
  component, followed by a weighted sum based on the fraction of each land, sea
  and sea icea at each grid level.

  Attributes:
  """

  coords: dict[str, cx.Coordinate] = dataclasses.field(init=False)
  land_transform: UnaryFieldTowerTransform
  sea_transform: UnaryFieldTowerTransform
  sea_ice_transform: UnaryFieldTowerTransform
  land_sea_mask_transform: transforms.Transform
  sea_ice_value_transform: transforms.Transform
  mesh: parallelism.Mesh = dataclasses.field(kw_only=True)

  def __post_init__(self):
    # ensure that coords are the same for all transforms.
    coords = set([
        tuple(sorted(self.land_transform.coords.items())),
        tuple(sorted(self.sea_transform.coords.items())),
        tuple(sorted(self.sea_ice_transform.coords.items())),
    ])
    if len(coords) != 1:
      raise ValueError(
          'Land, sea and sea ice transforms must have the same output shapes.'
      )
    self.coords = dict(list(coords)[0])

  def __call__(self, inputs):

    # Here we assume NaNs in sea_ice_cover are the superset those in SST.
    sea_ice_fraction = self.sea_ice_value_transform(inputs)['sea_ice_cover']
    land_mask = cx.cmap(jnp.isnan)(sea_ice_fraction)
    sea_ice_fraction = cx.cmap(jnp.nan_to_num)(sea_ice_fraction)
    land_fraction = cx.cmap(jnp.maximum)(
        self.land_sea_mask_transform(inputs)['land_sea_mask'],
        land_mask,
    )
    sea_fraction = 1 - land_fraction

    land_outputs = self.land_transform(inputs)
    sea_outputs = self.sea_transform(inputs)
    sea_ice_outputs = self.sea_ice_transform(inputs)

    # weight and combine outputs
    land_weight = land_fraction
    sea_ice_weight = sea_ice_fraction * sea_fraction  # ice covered sea.
    sea_weight = (1 - sea_ice_fraction) * sea_fraction  # sea without ice.

    result = {}
    for k in self.coords:
      result[k] = (
          land_outputs[k] * land_weight
          + sea_outputs[k] * sea_weight
          + sea_ice_outputs[k] * sea_ice_weight
      )

    return result
