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
class ForwardTowerTransform(transforms.TransformABC, nnx.Module):
  """Transforms fields with ForwardTower and splits the output to fields.

  Attributes:
    targets: A dictionary mapping output names to their coordinates.
    tower: The ForwardTower module to apply to the combined inputs.
    dims_to_align: A tuple of dimsension names or coordinates used to align
      fields when combining inputs and splitting outputs.
    in_transform: Optional transform to be applied to inputs.
    out_transform: Optional transform to be applied to module outputs.
    feature_sharding_schema: Optional features sharding schema.
    result_sharding_schema: Optional result sharding schema.
    mesh: The `parallelism.Mesh` used for sharding.
  """

  targets: dict[str, cx.Coordinate]
  tower: towers.ForwardTower
  dims_to_align: tuple[str | cx.Coordinate, ...]
  in_transform: transforms.Transform = transforms.Identity()
  out_transform: transforms.Transform = transforms.Identity()
  feature_sharding_schema: str | None = None
  result_sharding_schema: str | None = None
  mesh: parallelism.Mesh = dataclasses.field(kw_only=True)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    apply_sharding = self.mesh.with_sharding_constraint
    in_features = self.in_transform(inputs)
    in_features = apply_sharding(in_features, self.feature_sharding_schema)
    tag = self.tower.inputs_in_dims[0]
    in_field = field_utils.combine_fields(in_features, self.dims_to_align, tag)
    out_field = self.tower(in_field)
    out_fields = field_utils.split_to_fields(out_field, self.targets)
    out_fields = apply_sharding(out_fields, self.result_sharding_schema)
    return self.out_transform(out_fields)

  @classmethod
  def build_using_factories(
      cls,
      input_shapes: dict[str, cx.Field],
      targets: dict[str, cx.Coordinate],
      tower_factory: towers.ForwardTowerFactory,
      dims_to_align: tuple[str | cx.Coordinate, ...],
      in_transform=transforms.Identity(),
      out_transform=transforms.Identity(),
      feature_sharding_schema: str | None = None,
      result_sharding_schema: str | None = None,
      *,
      mesh: parallelism.Mesh,
      rngs,
  ):
    """Builds a ForwardTowerTransform using factories for submodules.

    Args:
      input_shapes: A dictionary of fields with the same shape structure as
        expected inputs. Used to determine the input size for the tower.
      targets: A dictionary mapping output field names to their coordinates.
        Used to determine the output size for the tower and for the `targets`
        attribute of the transform.
      tower_factory: A factory function that creates the ForwardTower. It
        should accept input_size, output_size, and rngs as arguments.
      dims_to_align: A tuple of dimension names or coordinates used to align
        fields when combining inputs and splitting outputs.
      in_transform: Optional transform to be applied to inputs.
      out_transform: Optional transform to be applied to module outputs.
      feature_sharding_schema: Optional features sharding schema.
      result_sharding_schema: Optional result sharding schema.
      mesh: The `parallelism.Mesh` used for sharding.
      rngs: The random number generators for initializing the tower.

    Returns:
      An instance of ForwardTowerTransform.
    """
    in_shapes = in_transform.output_shapes(input_shapes)
    in_field_shape = nnx.eval_shape(
        field_utils.combine_fields, in_shapes, dims_to_align
    )
    out_shapes = field_utils.shape_struct_fields_from_coords(targets)
    out_field_shape = nnx.eval_shape(
        field_utils.combine_fields, out_shapes, dims_to_align
    )
    input_size = in_field_shape.positional_shape[0]
    output_size = out_field_shape.positional_shape[0]
    tower = tower_factory(input_size, output_size, rngs=rngs)
    return cls(
        targets=targets,
        tower=tower,
        dims_to_align=dims_to_align,
        in_transform=in_transform,
        out_transform=out_transform,
        feature_sharding_schema=feature_sharding_schema,
        result_sharding_schema=result_sharding_schema,
        mesh=mesh,
    )


@nnx_compat.dataclass
class WeightedLandSeaIceTowersTransform(transforms.TransformABC, nnx.Module):
  """Combines FieldTowerTransformsTransforms for landd, sea and sea ice.

  Outputs are computed by evaluating ForwardTowerTransforms for each
  component, followed by a weighted sum based on the fraction of each land, sea
  and sea icea at each grid level.

  targets: A dictionary mapping output names to their coordinates. This is
      derived from the `land_transform`, `sea_transform`, and
      `sea_ice_transform` and is set in `__post_init__`. All three transforms
      must have the same target shapes.
    land_transform: A tower transform applied to inputs over land.
    sea_transform: A tower transform applied to inputs over sea.
    sea_ice_transform: A tower transform applied to inputs over sea-ice.
    land_sea_mask_transform: A transform that provides the 'land_sea_mask'
      field, indicating the fraction of land.
    sea_ice_value_transform: A transform that provides the 'sea_ice_cover'
      field, indicating the fraction of sea ice.
    mesh: The `parallelism.Mesh` used for sharding.
  """

  targets: dict[str, cx.Coordinate] = dataclasses.field(init=False)
  land_transform: ForwardTowerTransform
  sea_transform: ForwardTowerTransform
  sea_ice_transform: ForwardTowerTransform
  land_sea_mask_transform: transforms.Transform
  sea_ice_value_transform: transforms.Transform
  mesh: parallelism.Mesh = dataclasses.field(kw_only=True)

  def __post_init__(self):
    # ensure that coords are the same for all transforms.
    targets = set([
        tuple(sorted(self.land_transform.targets.items())),
        tuple(sorted(self.sea_transform.targets.items())),
        tuple(sorted(self.sea_ice_transform.targets.items())),
    ])
    if len(targets) != 1:
      raise ValueError(
          'Land, sea and sea ice transforms must have the same output shapes.'
      )
    self.targets = dict(list(targets)[0])

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    # Here we assume NaNs in sea_ice_cover are the superset those in SST.
    sea_ice_fraction = self.sea_ice_value_transform(inputs)['sea_ice_cover']
    # land mask is set to True wherever sea ice cover is not defined.
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
    for k in self.targets:
      result[k] = (
          land_outputs[k] * land_weight
          + sea_outputs[k] * sea_weight
          + sea_ice_outputs[k] * sea_ice_weight
      )
    return result
