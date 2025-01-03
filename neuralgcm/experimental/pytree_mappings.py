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

"""Modules that define learnable mappings between input/output pytrees."""

from typing import Callable, Protocol, Sequence

from flax import nnx
import jax
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import pytree_transforms
from neuralgcm.experimental import pytree_utils
from neuralgcm.experimental import towers
from neuralgcm.experimental import typing


class PytreeMapping(Protocol):
  """Protocol for pytree_mapping modules."""

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    ...

  @property
  def output_shapes(self) -> typing.Pytree:
    ...


PytreeMappingFactory = Callable[..., PytreeMapping]


MINIMAL_STATE_STRUCT = {'sim_time': 0.0}


class ChannelMapping(nnx.Module):
  """Maps the pytree of nodal features to a pytree of specified structure.

  This module packs the pytree into a single array of shape (n, lon, lat),
  passes it to a NN tower, and unpacks the result into a pytree with the
  structure of output_shapes, typically (m, lon, lat).
  """

  def __init__(
      self,
      input_shapes: typing.Pytree,
      output_shapes: typing.Pytree,
      *,
      tower_factory: towers.UnaryTowerFactory,
      rngs: nnx.Rngs,
  ):
    # TODO(dkochkov) consider passing `coords` to infer how input features are
    # structured and which axis in output_shapes is a channel axis.
    f_axis = -3  # default column axis.
    input_size = sum([x.shape[f_axis] for x in jax.tree.leaves(input_shapes)])
    output_size = sum([x.shape[f_axis] for x in jax.tree.leaves(output_shapes)])
    # tower preserves the last two spatial dimensions.
    self.tower = tower_factory(input_size, output_size, rngs=rngs)
    self._output_shapes = output_shapes
    self.feature_axis = f_axis

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    # TODO(dkochkov) Consider adding check that inputs is not just an array.
    array = pytree_utils.pack_pytree(inputs, self.feature_axis)
    if array.ndim != 3:
      raise ValueError(f'Expected input array with ndim=3, got {array.shape=}')
    outputs = self.tower(array)
    if outputs.ndim != 3:
      raise ValueError(f'Expected outputs with ndim=3, got {outputs.shape=}')
    return pytree_utils.unpack_to_pytree(
        outputs, self._output_shapes, self.feature_axis
    )

  @property
  def output_shapes(self):
    return self._output_shapes


class VariableMapping(nnx.Module):
  """Maps the pytree of nodal volume features to a pytree of given structure.

  This module stacks the input pytree into an array of shape
  (channel, level, lon, lat), passes it to a NN tower.  The output from the NN
  is expected to have shape (n, level, lon, lat), and gets unpacked to a pytree
  with the structure of output_shapes, e.g.
    output_shapes = {
        'var_1': jnp.asarray((level, lon, lat)),
        'var_2': jnp.asarray((level, lon, lat)),
        ...,
        'var_n': jnp.asarray((level, lon, lat)),
    }

  The leaves of the input pytree must have the same shape, e.g. (1, lon, lat) or
  (level, lon, lat). To mix shapes, broadcast before passing to the mapping.
  """

  def __init__(
      self,
      input_shapes: typing.Pytree,
      output_shapes: typing.Pytree,
      *,
      tower_factory: towers.UnaryTowerFactory,
      rngs: nnx.Rngs,
  ):
    feature_axis = 0
    input_size = len(jax.tree.leaves(input_shapes))
    output_size = len(jax.tree.leaves(output_shapes))

    # tower preserves the last two spatial dimensions.
    self.tower = tower_factory(input_size, output_size, rngs=rngs)
    self._output_shapes = output_shapes
    self.feature_axis = feature_axis

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    array = pytree_utils.stack_pytree(inputs, axis=self.feature_axis)
    if array.ndim != 4:
      raise ValueError(f'Expected input array with ndim=4, got {array.shape=}')
    outputs = self.tower(array)
    if outputs.ndim != 4:
      raise ValueError(f'Expected outputs with ndim=4, got {outputs.shape=}')
    return pytree_utils.unstack_to_pytree(
        outputs, self._output_shapes, axis=self.feature_axis
    )

  @property
  def output_shapes(self):
    return self._output_shapes


class MappingWithNormalizedInputs(nnx.Module):
  """Applies pytree mapping with input normalization."""

  def __init__(
      self,
      input_shapes: typing.Pytree,
      output_shapes: typing.Pytree,
      *,
      mapping_factory: PytreeMappingFactory,
      normalization_factory: pytree_transforms.TransformFactory,
      rngs: nnx.Rngs,
  ):
    self._output_shapes = output_shapes
    self.normalization = normalization_factory(input_shapes, rngs=rngs)
    self.mapping = mapping_factory(
        input_shapes=self.normalization.output_shapes(input_shapes),
        output_shapes=output_shapes,
        rngs=rngs,
    )

  @property
  def output_shapes(self):
    return self._output_shapes

  def normalize(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.normalization(inputs)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.mapping(self.normalize(inputs))


# TODO(dkochkov) Consider parameterizing this with mapping factories.
class ParallelMapping(nnx.Module):
  """Maps a pytree to a pytree by additively compbining multiple mappings."""

  def __init__(
      self,
      mappings: Sequence[PytreeMapping],
  ):
    self.mappings = mappings

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    results = [mapping(inputs) for mapping in self.mappings]
    return jax.tree.map(lambda *args: sum(args), *results)

  @property
  def output_shapes(self):
    return self.mappings[0].output_shapes


class Embedding(nnx.Module):
  """Generates floating-value embeddings from inputs using a pytree mapping."""

  def __init__(
      self,
      output_shapes: dict[str, typing.ShapeDtypeStruct],
      feature_module: nnx.Module,
      mapping_factory: nnx.Module,
      transform: pytree_transforms.Transform = pytree_transforms.Identity(),
      input_state_shapes: typing.Pytree | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    if input_state_shapes is None:
      input_state_shapes = MINIMAL_STATE_STRUCT  # default.
    self.feature_module = feature_module
    self.mapping = mapping_factory(
        input_shapes=feature_module.output_shapes(input_state_shapes),
        output_shapes=output_shapes,
        rngs=rngs,
    )
    self.transform = transform

  def __call__(self, inputs: typing.Pytree):
    # TODO(dkochkov): Figure out if we need to support sharding here, it would
    # require passing coords object as they are holding sharding information.
    # inputs = self.coords.with_dycore_sharding(inputs)
    features = self.feature_module(inputs)
    # features = self.coords.dycore_to_physics_sharding(features)
    outputs = self.mapping(features)
    # outputs = self.coords.physics_to_dycore_sharding(outputs)
    outputs = self.transform(outputs)
    # outputs = self.coords.with_dycore_sharding(outputs)
    return outputs

  @property
  def output_shapes(self) -> typing.Pytree:
    return self.transform.output_shapes(self.mapping.output_shapes)


class LandSeaIceEmbedding(nnx.Module):
  """Embedding module that combines embeddings over land, sea and sea ice."""

  def __init__(
      self,
      output_shapes: dict[str, typing.ShapeDtypeStruct],
      sea_embedding_factory,
      land_embedding_factory,
      sea_ice_embedding_factory,
      land_sea_mask_features,
      sea_ice_features,
      transform: pytree_transforms.Transform = pytree_transforms.Identity(),
      *,
      rngs: nnx.Rngs,
  ):
    self.land_embedding = land_embedding_factory(
        output_shapes=output_shapes, rngs=rngs
    )
    self.sea_embedding = sea_embedding_factory(
        output_shapes=output_shapes, rngs=rngs
    )
    self.sea_ice_embedding = sea_ice_embedding_factory(
        output_shapes=output_shapes, rngs=rngs
    )
    self.land_sea_mask_features = land_sea_mask_features
    self.sea_ice_features = sea_ice_features
    self.transform = transform
    self._output_shapes = output_shapes

  def __call__(
      self,
      inputs: typing.Pytree,
  ) -> typing.Pytree:
    """Returns the embedding output on nodal locations."""
    # get outputs from each model
    land_outputs = self.land_embedding(inputs)
    sea_outputs = self.sea_embedding(inputs)
    sea_ice_outputs = self.sea_ice_embedding(inputs)

    # prepare masks with fractional values in [0, 1]
    land_fraction = self.land_sea_mask_features(inputs)['land_sea_mask']
    sea_fraction = 1 - land_fraction
    sea_ice_fraction = self.sea_ice_features(inputs)['sea_ice_cover']
    # weight and combine outputs
    land_weight = land_fraction
    sea_ice_weight = sea_ice_fraction * sea_fraction  # ice covered sea
    sea_weight = (1 - sea_ice_fraction) * sea_fraction  # sea without ice

    def tree_scale(a, x):
      # Multiply leaves of `x` by `a`.
      return jax.tree.map(lambda y: a * y, x)

    outputs = jax.tree.map(
        lambda a, b, c: a + b + c,
        tree_scale(land_weight, land_outputs),
        tree_scale(sea_weight, sea_outputs),
        tree_scale(sea_ice_weight, sea_ice_outputs),
    )
    return self.transform(outputs)

  @property
  def output_shapes(self) -> typing.Pytree:
    parts = [self.land_embedding, self.sea_embedding, self.sea_ice_embedding]
    shapes = [x.output_shapes for x in parts]
    sample_shape = shapes[0]
    for shape in shapes:
      if shape != sample_shape:
        raise ValueError(f'Inconsistent embedding output shapes: {shapes=}')
    return self.transform.output_shapes(sample_shape)


class CoordsStateMapping(nnx.Module):
  """Predicts a pytree of a state specified by coordinates.."""

  def __init__(
      self,
      *,
      coords: coordinates.DinosaurCoordinates,
      surface_field_names: tuple[str, ...],
      volume_field_names: tuple[str, ...],
      embedding_factory: PytreeMappingFactory,
      transform: pytree_transforms.Transform = pytree_transforms.Identity(),
      rngs: nnx.Rngs,
  ):
    output_shapes = {}
    for name in volume_field_names:
      output_shapes[name] = typing.ShapeFloatStruct(coords.shape)
    for name in surface_field_names:
      output_shapes[name] = typing.ShapeFloatStruct(
          (1,) + coords.horizontal.shape
      )
    self.coords = coords
    self.embedding = embedding_factory(output_shapes, rngs=rngs)
    self.transform = transform

  def __call__(self, inputs: typing.PyTreeState) -> typing.PyTreeState:
    output = self.embedding(inputs)
    output = self.transform(output)
    # outputs = self.coords.with_dycore_sharding(outputs)
    return output

  @property
  def output_shapes(self) -> typing.Pytree:
    return self.transform.output_shapes(self.embedding.output_shapes)


class MaskedSurfaceFeatureEmbedding(nnx.Module):
  """Embedding to add learned features to masked portions of data."""

  def __init__(
      self,
      variables_to_embed: Sequence[str],
      mask_values: dict[str, typing.Array],
      grid: coordinates.LonLatGrid,
      feature_names: dict[str, str],
      *,
      feature_module: nnx.Module,
      param_type: pytree_transforms.TransformParams | nnx.Param = nnx.Param,
      initializer: (
          nnx.Initializer | typing.Array
      ) = nnx.initializers.truncated_normal(stddev=1),
      rngs: nnx.Rngs,
  ):
    spatial_shape = grid.shape
    self.spatial_shape = spatial_shape

    self.mask_values = {
        k: nnx.Variable(1.0 - mask_value)
        for k, mask_value in mask_values.items()
    }
    self.feature_names = feature_names
    self.feature_module = feature_module
    self.features = {}
    for k in variables_to_embed:
      if isinstance(initializer[k], typing.Array):
        self.features[k] = param_type(initializer[k])
      else:
        self.features[k] = param_type(
            initializer[k](key=rngs.params(), shape=(spatial_shape))
        )

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    inputs = self.feature_module(inputs)
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    outputs = {}
    for k, v in inputs.items():
      if k in self.features:
        outputs[self.feature_names[k]] = (
            v + self.features[k] * self.mask_values[k]
        )
    return from_dict_fn(outputs)

  @property
  def output_shapes(self) -> typing.Pytree:
    return {
        self.feature_names[k]: typing.ShapeFloatStruct(
            (1,) + self.spatial_shape
        )
        for k, _ in self.features.items()
    }


class MaskedVolumeFeatureEmbedding(nnx.Module):
  """Embedding to add learned features to masked portions of data."""

  def __init__(
      self,
      variables_to_embed: Sequence[str],
      mask_values: dict[str, typing.Array],
      coords: cx.Coordinate,
      feature_names: dict[str, str],
      *,
      feature_module: nnx.Module,
      param_type: pytree_transforms.TransformParams | nnx.Param = nnx.Param,
      initializer: (
          dict[str, typing.Array] | nnx.initializers.Initializer | None
      ) = None,
      rngs: nnx.Rngs,
  ):
    default_w_init = nnx.initializers.lecun_normal()
    self.coords = coords
    spatial_shape = coords.shape
    self.spatial_shape = spatial_shape
    initializer = initializer if initializer is not None else default_w_init
    self.mask_values = {
        k: nnx.Variable(1.0 - mask_value)
        for k, mask_value in mask_values.items()
    }
    # self.mask_values = {k: 0.0 for k, mask_value in mask_values.items()}
    self.feature_names = feature_names
    self.feature_module = feature_module
    self.features = {}
    for k in variables_to_embed:
      if isinstance(initializer, dict):
        self.features[k] = param_type(initializer[k])
      else:
        self.features[k] = param_type(
            initializer(key=rngs.params(), shape=(spatial_shape))
        )

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    inputs = self.feature_module(inputs)
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    outputs = {}
    for k, v in inputs.items():
      if k in self.features:
        outputs[self.feature_names[k]] = (
            v + self.features[k] * self.mask_values[k]
        )
    return from_dict_fn(outputs)

  @property
  def output_shapes(self) -> typing.Pytree:
    return {
        self.feature_names[k]: typing.ShapeFloatStruct(self.spatial_shape)
        for k, _ in self.features.items()
    }
