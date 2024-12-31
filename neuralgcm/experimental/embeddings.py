"""Modules that generate embedding vectors from data."""

import dataclasses
from typing import Protocol, Sequence

from flax import nnx
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import pytree_transforms
from neuralgcm.experimental import pytree_utils
from neuralgcm.experimental import typing


default_w_init = nnx.initializers.lecun_normal()


class EmbeddingModule(Protocol):
  """Protocol for input_feature modules."""

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    ...

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    ...


class MaskedDataSurfaceFeatureEmbedding(nnx.Module):
  """Embedding to add learned features to masked portions of data."""

  def __init__(
      self,
      variables_to_embed: Sequence[str],
      mask_values: dict[str, typing.Array],
      grid: coordinates.LonLatGrid,
      feature_names: dict[str, str] = dataclasses.field(
          default_factory=lambda: {'sea_surface_temperature': 'embedded_SST'}
      ),
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

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    del input_shapes  # unused.
    return {
        self.feature_names[k]: typing.ShapeFloatStruct(
            (1,) + self.spatial_shape
        )
        for k, _ in self.features.items()
    }


class MaskedDataVolumeFeatureEmbedding(nnx.Module):
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

  def output_shapes(
      self, input_shapes: typing.Pytree | None = None
  ) -> typing.Pytree:
    del input_shapes  # unused.
    return {
        self.feature_names[k]: typing.ShapeFloatStruct(self.spatial_shape)
        for k, _ in self.features.items()
    }
