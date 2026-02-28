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
import math

import coordax as cx
from flax import nnx
import jax.numpy as jnp
from neuralgcm.experimental.core import field_utils
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import towers
from neuralgcm.experimental.core import transforms
from neuralgcm.experimental.core import typing


def _infer_positional_idx(
    fields: dict[str, cx.Field],
) -> int:
  """Returns the unique positional index in fields."""
  indices = {
      k: f.dims.index(None) - f.ndim
      for k, f in fields.items()
      if None in f.dims
  }
  if not indices:
    raise ValueError(
        'No positional index found in fields. This can happen if concat_dims '
        'are missing or if the input fields are not properly expanded. '
        'Consider adding `InsertAxis` at the end of feature transforms to '
        'indicate the desired location of the concatenated axis.'
    )
  elif len(set(indices.values())) == 1:
    [idx] = list(set(indices.values()))
    return idx
  raise ValueError(f'Fields have mismatching positional axes: {indices}')


def _concat_fields(
    fields: dict[str, cx.Field],
    concat_dims: tuple[str | cx.Coordinate, ...],
) -> cx.Field:
  """Concatenates aligned `fields` along `concat_dims`..

  This function assumes that `fields` are aligned and that the concatenation
  axis in each field is either one of `concat_dims` or absent (in which case a
  dummy dimension is added).

  Args:
    fields: Fields to concatenate.
    concat_dims: Dimension names or coordinates to concatenate along. There
      should be at most one axis from `concat_dims` in each field.

  Returns:
    A concatenated `features` with all tower_in_dims tagged.
  """
  fields = cx.untag(fields, *concat_dims, allow_missing=True)
  idx = _infer_positional_idx(fields)
  fields = field_utils.ensure_positional_axis_idx(fields, idx=idx)
  fields = dict(sorted(fields.items()))  # ensure deterministic concat order.
  return cx.cpmap(lambda *vs: jnp.concatenate(vs))(*fields.values())


@nnx_compat.dataclass
class ForwardTowerTransform(transforms.TransformABC, nnx.Module):
  """Transforms fields with ForwardTower and splits the output to fields.

  Attributes:
    target_split_axes: Mapping of output names to their split axes.
    tower: The ForwardTower module to apply to the combined inputs.
    concat_dims: Dimensions used to concatenate fields when combining inputs.
    inputs_transform: Optional transform to be applied to inputs.
    out_transform: Optional transform to be applied to module outputs.
    feature_sharding_schema: Optional features sharding schema.
    result_sharding_schema: Optional result sharding schema.
    mesh: The `parallelism.Mesh` used for sharding.
  """

  target_split_axes: dict[str, cx.Coordinate]
  tower: towers.ForwardTower
  concat_dims: tuple[str | cx.Coordinate, ...]
  inputs_transform: typing.Transform = transforms.Identity()
  out_transform: typing.Transform = transforms.Identity()
  feature_sharding_schema: str | None = None
  result_sharding_schema: str | None = None
  mesh: parallelism.Mesh = dataclasses.field(kw_only=True)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    tower_in_dims = self.tower.inputs_in_dims
    in_c = tower_in_dims[0]  # convention: first dim is the input channel.
    out_c = self.tower.out_dims[0]  # convention: first dim is output channel.
    apply_sharding = self.mesh.with_sharding_constraint
    features = self.inputs_transform(inputs)
    features = apply_sharding(features, self.feature_sharding_schema)
    # Concatenating features and passing them through the tower.
    in_field = _concat_fields(features, self.concat_dims)
    in_field = in_field.tag(in_c)
    out_field = self.tower(in_field)
    # Splitting the output into targets.
    out_fields = field_utils.split_field_axis(
        out_field, out_c, self.target_split_axes
    )
    out_fields = apply_sharding(out_fields, self.result_sharding_schema)
    return self.out_transform(out_fields)

  @classmethod
  def build_using_factories(
      cls,
      input_shapes: dict[str, cx.Field],
      target_split_axes: dict[str, cx.Coordinate],
      tower_factory: towers.ForwardTowerFactory,
      concat_dims: tuple[str | cx.Coordinate, ...] = (),
      inputs_transform: typing.Transform = transforms.Identity(),
      out_transform: typing.Transform = transforms.Identity(),
      feature_sharding_schema: str | None = None,
      result_sharding_schema: str | None = None,
      *,
      mesh: parallelism.Mesh,
      rngs,
  ):
    """Builds a ForwardTowerTransform using factories for submodules.

    Args:
      input_shapes: Fields with expected input shape structure.
      target_split_axes: Mapping of output field names to split coordinates.
      tower_factory: Factory creating the ForwardTower.
      concat_dims: Dimensions used to concatenate input fields.
      inputs_transform: Optional transform applied to inputs.
      out_transform: Optional transform applied to module outputs.
      feature_sharding_schema: Optional features sharding schema.
      result_sharding_schema: Optional result sharding schema.
      mesh: The `parallelism.Mesh` used for sharding.
      rngs: Random number generators for tower initialization.

    Returns:
      An instance of ForwardTowerTransform.
    """
    in_shapes = inputs_transform.output_shapes(input_shapes)
    in_field_shape = nnx.eval_shape(
        lambda s: _concat_fields(s, concat_dims),
        in_shapes,
    )
    input_size = in_field_shape.positional_shape[0]
    output_size = sum(math.prod(c.shape) for c in target_split_axes.values())
    tower = tower_factory(input_size, output_size, rngs=rngs)
    return cls(
        target_split_axes=target_split_axes,
        tower=tower,
        concat_dims=concat_dims,
        inputs_transform=inputs_transform,
        out_transform=out_transform,
        feature_sharding_schema=feature_sharding_schema,
        result_sharding_schema=result_sharding_schema,
        mesh=mesh,
    )


@nnx_compat.dataclass
class RecurrentTowerTransform(transforms.TransformABC, nnx.Module):
  """Transforms fields with RecurrentTower and splits the output to fields.

  Attributes:
    target_split_axes: Mapping of output names to their split axes.
    tower: The RecurrentTower module to apply to the combined inputs.
    concat_dims: Dimensions used to concatenate fields when combining inputs.
    inputs_transform: Optional transform to be applied to inputs.
    out_transform: Optional transform to be applied to module outputs.
    state_keys: Keys for recurrence state in input dict.
    feature_sharding_schema: Optional features sharding schema.
    result_sharding_schema: Optional result sharding schema.
    mesh: The `parallelism.Mesh` used for sharding.
  """

  target_split_axes: dict[str, cx.Coordinate]
  tower: towers.RecurrentTower
  concat_dims: tuple[str | cx.Coordinate, ...]
  inputs_transform: typing.Transform = transforms.Identity()
  out_transform: typing.Transform = transforms.Identity()
  state_keys: tuple[str, ...] = ('lstm_c', 'lstm_h')
  feature_sharding_schema: str | None = None
  result_sharding_schema: str | None = None
  mesh: parallelism.Mesh = dataclasses.field(kw_only=True)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    apply_sharding = self.mesh.with_sharding_constraint
    in_features_with_state = self.inputs_transform(inputs)
    in_features_with_state = apply_sharding(
        in_features_with_state, self.feature_sharding_schema
    )
    in_features = in_features_with_state.copy()
    in_c = self.tower.inputs_in_dims[0]
    out_c = self.tower.out_dims[0]
    carry_parts = [in_features.pop(key) for key in self.state_keys]
    carry = tuple(carry_parts) if len(carry_parts) > 1 else carry_parts[0]

    in_field = _concat_fields(in_features, self.concat_dims)
    in_field = in_field.tag(in_c)
    out_carry, out_field = self.tower(carry, in_field)
    out_fields = field_utils.split_field_axis(
        out_field, out_c, self.target_split_axes
    )
    out_fields = self.out_transform(out_fields)

    out_carry = (out_carry,) if cx.is_field(out_carry) else out_carry
    for key, val in zip(self.state_keys, out_carry):
      out_fields[key] = val

    out_fields = apply_sharding(out_fields, self.result_sharding_schema)
    return out_fields

  @classmethod
  def build_using_factories(
      cls,
      input_shapes: dict[str, cx.Field],
      target_split_axes: dict[str, cx.Coordinate],
      tower_factory: towers.RecurrentTowerFactory,
      concat_dims: tuple[str | cx.Coordinate, ...],
      inputs_transform=transforms.Identity(),
      out_transform=transforms.Identity(),
      state_keys: tuple[str, ...] = ('lstm_c', 'lstm_h'),
      feature_sharding_schema: str | None = None,
      result_sharding_schema: str | None = None,
      *,
      mesh: parallelism.Mesh,
      rngs,
  ):
    """Builds a RecurrentTowerTransform using factories for submodules.

    Args:
      input_shapes: Fields with expected input shape structure.
      target_split_axes: Mapping of output field names to split coordinates.
      tower_factory: Factory creating the RecurrentTower.
      concat_dims: Dimensions used to concatenate input fields.
      inputs_transform: Optional transform applied to inputs.
      out_transform: Optional transform applied to module outputs.
      state_keys: Keys for recurrence state in input dict.
      feature_sharding_schema: Optional features sharding schema.
      result_sharding_schema: Optional result sharding schema.
      mesh: The `parallelism.Mesh` used for sharding.
      rngs: Random number generators for tower initialization.

    Returns:
      An instance of RecurrentTowerTransform.
    """

    in_shapes = inputs_transform.output_shapes(input_shapes)
    in_shapes_for_combine = in_shapes.copy()
    for key in state_keys:
      in_shapes_for_combine.pop(key)
    in_field_shape = nnx.eval_shape(
        lambda s: _concat_fields(s, concat_dims),
        in_shapes_for_combine,
    )
    input_size = in_field_shape.positional_shape[0]
    output_size = sum(math.prod(c.shape) for c in target_split_axes.values())
    tower = tower_factory(input_size, output_size, rngs=rngs)
    return cls(
        target_split_axes=target_split_axes,
        tower=tower,
        concat_dims=concat_dims,
        inputs_transform=inputs_transform,
        out_transform=out_transform,
        state_keys=state_keys,
        feature_sharding_schema=feature_sharding_schema,
        result_sharding_schema=result_sharding_schema,
        mesh=mesh,
    )


@nnx_compat.dataclass
class TransformerTowerTransform(transforms.TransformABC, nnx.Module):
  """Transforms fields with TransformerTower and splits the output to fields.

  Attributes:
    target_split_axes: Mapping of output names to their split axes.
    tower: The TransformerTower to apply to generated inputs, latents and mask.
    concat_dims: Dimensions used to align fields when combining main inputs.
    inputs_transform: Transform to extract fields for main inputs.
    latents_transform: Transform to extract fields for latents input.
    mask_values_transform: Transform to extract fields for attention mask.
    out_transform: Transform to be applied to module outputs.
    latents_concat_dims: Optional dimensions used to align fields when
      combining latents.
    feature_sharding_schema: Optional features sharding schema.
    result_sharding_schema: Optional result sharding schema.
    mesh: The `parallelism.Mesh` used for sharding.
  """

  target_split_axes: dict[str, cx.Coordinate]
  tower: towers.TransformerTower
  concat_dims: tuple[str | cx.Coordinate, ...]
  inputs_transform: typing.Transform = transforms.Identity()
  latents_transform: typing.Transform = transforms.Empty()
  mask_values_transform: typing.Transform = transforms.Empty()
  out_transform: typing.Transform = transforms.Identity()
  latents_concat_dims: tuple[str | cx.Coordinate, ...] | None = None
  feature_sharding_schema: str | None = None
  result_sharding_schema: str | None = None
  mesh: parallelism.Mesh = dataclasses.field(kw_only=True)

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    feature_schema = self.feature_sharding_schema
    result_schema = self.result_sharding_schema
    with_features_sharding = lambda x: self.mesh.with_sharding_constraint(
        x, feature_schema
    )
    with_results_sharding = lambda x: self.mesh.with_sharding_constraint(
        x, result_schema
    )
    z_mask_concat_dims = self.latents_concat_dims or self.concat_dims
    in_c = self.tower.inputs_in_dims[0]  # tag for the channel dimension.
    out_c = self.tower.out_dims[0]  # tag for the channel dimension.

    def f_combine(f: dict[str, cx.Field] | None, dims):
      if not f:
        return None
      f = with_features_sharding(f)
      return _concat_fields(f, dims).tag(in_c)

    x = f_combine(self.inputs_transform(inputs), self.concat_dims)
    z = f_combine(self.latents_transform(inputs), z_mask_concat_dims)
    mask_concat_dims = self.latents_concat_dims or self.concat_dims
    mask = f_combine(self.mask_values_transform(inputs), mask_concat_dims)
    out_field = self.tower(inputs=x, latents=z, mask=mask)

    out_fields = field_utils.split_field_axis(
        out_field, out_c, self.target_split_axes
    )
    out_fields = with_results_sharding(out_fields)
    return self.out_transform(out_fields)

  @classmethod
  def build_using_factories(
      cls,
      input_shapes: dict[str, cx.Field],
      target_split_axes: dict[str, cx.Coordinate],
      tower_factory: towers.TransformerTowerFactory,
      concat_dims: tuple[str | cx.Coordinate, ...],
      inputs_transform: typing.Transform = transforms.Identity(),
      latents_transform: typing.Transform = transforms.Empty(),
      mask_values_transform: typing.Transform = transforms.Empty(),
      out_transform: typing.Transform = transforms.Identity(),
      latents_concat_dims: tuple[str | cx.Coordinate, ...] | None = None,
      feature_sharding_schema: str | None = None,
      result_sharding_schema: str | None = None,
      *,
      mesh: parallelism.Mesh,
      rngs,
  ):
    """Builds a TransformerTowerTransform using factories for submodules.

    Args:
      input_shapes: Fields with expected input shape structure.
      target_split_axes: Mapping of output field names to split coordinates.
      tower_factory: Factory creating the TransformerTower.
      concat_dims: Dimensions used to align fields when combining main inputs.
      inputs_transform: Transform to extract fields for main inputs.
      latents_transform: Transform to extract fields for latents input.
      mask_values_transform: Transform to extract fields for attention mask.
      out_transform: Transform applied to module outputs.
      latents_concat_dims: Dimensions used to align fields when combining
        latents.
      feature_sharding_schema: Optional features sharding schema.
      result_sharding_schema: Optional result sharding schema.
      mesh: The `parallelism.Mesh` used for sharding.
      rngs: Random number generators for tower initialization.

    Returns:
      An instance of TransformerTowerTransform.
    """
    inputs_in_shapes = inputs_transform.output_shapes(input_shapes)
    inputs_field_shape = nnx.eval_shape(
        lambda s: _concat_fields(s, concat_dims),
        inputs_in_shapes,
    )
    input_size = inputs_field_shape.positional_shape[0]
    output_size = sum(math.prod(c.shape) for c in target_split_axes.values())
    tower = tower_factory(input_size, output_size, rngs=rngs)
    return cls(
        target_split_axes=target_split_axes,
        tower=tower,
        concat_dims=concat_dims,
        inputs_transform=inputs_transform,
        latents_transform=latents_transform,
        mask_values_transform=mask_values_transform,
        latents_concat_dims=latents_concat_dims,
        out_transform=out_transform,
        feature_sharding_schema=feature_sharding_schema,
        result_sharding_schema=result_sharding_schema,
        mesh=mesh,
    )


@nnx_compat.dataclass
class LandSeaIceTowersTransform(transforms.TransformABC, nnx.Module):
  """Combines FieldTowerTransforms for land, sea and sea ice.

  Outputs are computed by evaluating ForwardTower for each
  component, followed by a weighted sum based on the fraction of each land, sea
  and sea icea at each grid level.

  Attributes:
    target_split_axes: Mapping of output names to their split axes. This is
      derived from the `land_transform`, `sea_transform`, and
      `sea_ice_transform` and is set in `__post_init__`. All three transforms
      must have the same `target_split_axes`.
    land_transform: A tower transform applied to inputs over land.
    sea_transform: A tower transform applied to inputs over sea.
    sea_ice_transform: A tower transform applied to inputs over sea-ice.
    land_sea_mask_transform: A transform that provides the 'land_sea_mask'
      field, indicating the fraction of land.
    sea_ice_value_transform: A transform that provides the 'sea_ice_cover'
      field, indicating the fraction of sea ice.
    mesh: The `parallelism.Mesh` used for sharding.
  """

  target_split_axes: dict[str, cx.Coordinate] = dataclasses.field(init=False)
  land_transform: ForwardTowerTransform
  sea_transform: ForwardTowerTransform
  sea_ice_transform: ForwardTowerTransform
  land_sea_mask_transform: typing.Transform
  sea_ice_value_transform: typing.Transform
  mesh: parallelism.Mesh = dataclasses.field(kw_only=True)

  def __post_init__(self):
    # ensure that coords are the same for all transforms.
    target_split_axes = set([
        tuple(sorted(self.land_transform.target_split_axes.items())),
        tuple(sorted(self.sea_transform.target_split_axes.items())),
        tuple(sorted(self.sea_ice_transform.target_split_axes.items())),
    ])
    if len(target_split_axes) != 1:
      raise ValueError(
          'Land, sea and sea ice transforms must have the same output shapes.'
      )
    self.target_split_axes = dict(list(target_split_axes)[0])

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
    for k in self.target_split_axes:
      result[k] = (
          land_outputs[k] * land_weight
          + sea_outputs[k] * sea_weight
          + sea_ice_outputs[k] * sea_ice_weight
      )
    return result
