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
"""Modules that transform data between pytrees."""

from typing import Callable, Optional, Sequence
from dinosaur import pytree_utils
from dinosaur import typing
import gin
import haiku as hk
import jax
from neuralgcm.legacy import transforms


Array = typing.Array
Tower = Callable[[int], Callable[..., Array]]
MappingModule = Callable[[typing.Pytree], typing.Pytree]


@gin.register(denylist=['output_shapes'])
class NodalMapping(hk.Module):
  """Maps the pytree of nodal features to a pytree of specified structure.

  This module packs the pytree into a single array of shape (n, lon, lat),
  passes it to a NN tower, and unpacks the result into a pytree with the
  structure of output_shapes, typically (m, lon, lat).
  """

  def __init__(
      self,
      output_shapes: typing.Pytree,
      tower_factory: Tower = gin.REQUIRED,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    feature_axis = -3  # default column axis.
    output_size = sum([x[feature_axis]
                       for x in jax.tree_util.tree_leaves(output_shapes)])
    # tower preserves the last two spatial dimensions.
    self.tower = tower_factory(output_size)
    self.output_shapes = output_shapes
    self.feature_axis = feature_axis

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    array = pytree_utils.pack_pytree(inputs, self.feature_axis)
    if array.ndim != 3:
      raise ValueError(f'Expected input array with ndim=3, got {array.shape=}')
    outputs = self.tower(array)
    if outputs.ndim != 3:
      raise ValueError(f'Expected outputs with ndim=3, got {outputs.shape=}')
    return pytree_utils.unpack_to_pytree(
        outputs, self.output_shapes, self.feature_axis)


@gin.register(denylist=['output_shapes'])
class NodalVolumeMapping(hk.Module):
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
      output_shapes: typing.Pytree,
      tower_factory: Tower = gin.REQUIRED,
      name: Optional[str] = None
  ):
    super().__init__(name=name)
    feature_axis = 0
    output_size = len(jax.tree_util.tree_leaves(output_shapes))

    # tower preserves the last two spatial dimensions.
    self.tower = tower_factory(output_size)
    self.output_shapes = output_shapes
    self.feature_axis = feature_axis

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    array = pytree_utils.stack_pytree(inputs, axis=self.feature_axis)
    if array.ndim != 4:
      raise ValueError(f'Expected input array with ndim=4, got {array.shape=}')
    outputs = self.tower(array)
    if outputs.ndim != 4:
      raise ValueError(f'Expected outputs with ndim=4, got {outputs.shape=}')
    return pytree_utils.unstack_to_pytree(
        outputs, self.output_shapes, axis=self.feature_axis
    )


@gin.register
class NodalVolumeTransformerMapping(hk.Module):
  """Maps the pytree of nodal volume features to a pytree of given structure.

  Similar to NodalVolumeMapping, but uses certain features as positional
  encoding arguments to the underlying transformer networks. Inputs are
  expected to be of shape (channel, level, lon, lat), which are split into
  encoder inputs, decoder inputs and positional encodings, which are then passed
  to transformer towers. The output of the NN is expected to have shape
  (n, level*, lon, lat), and gets unpacked to a pytree with the structure of
  output_shapes, e.g.
    output_shapes = {
        'out_1': jnp.asarray((level*, lon, lat)),
        'out_2': jnp.asarray((level*, lon, lat)),
        ...,
        'out_n': jnp.asarray((level*, lon, lat)),
    }
  Note: the output number of levels `level*` is equal to those defined by the
  `decoder_inputs_selection_module`. In case it is empty, level* == level.

  The leaves of the encoder/decoder pytrees must have the same shape, e.g.
  (1, lon, lat) or (level, lon, lat) or (level*, lon, lat). To mix shapes,
  broadcast before passing to the mapping.
  """

  def __init__(
      self,
      output_shapes: typing.Pytree,
      encoder_transformer_tower_factory: Tower = gin.REQUIRED,
      decoder_transformer_tower_factory: Tower = gin.REQUIRED,
      latent_size: int = gin.REQUIRED,
      encoder_inputs_selection_module=gin.REQUIRED,
      decoder_inputs_selection_module=transforms.EmptyTransform,
      encoder_pos_encoding_module=transforms.EmptyTransform,
      decoder_pos_encoding_module=transforms.EmptyTransform,
      name: Optional[str] = None
  ):
    super().__init__(name=name)
    feature_axis = 0
    output_size = len(jax.tree_util.tree_leaves(output_shapes))
    self.encoder_tower = encoder_transformer_tower_factory(latent_size)
    self.decoder_tower = decoder_transformer_tower_factory(output_size)
    self.output_shapes = output_shapes
    self.feature_axis = feature_axis
    self.get_encoder_inputs_fn = encoder_inputs_selection_module()
    self.get_decode_inputs_fn = decoder_inputs_selection_module()
    self.encoder_positional_encodings_fn = encoder_pos_encoding_module()
    self.decoder_positional_encodings_fn = decoder_pos_encoding_module()

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    enc_inputs = self.get_encoder_inputs_fn(inputs)
    dec_inputs = self.get_decode_inputs_fn(inputs)
    enc_array = pytree_utils.stack_pytree(enc_inputs, axis=self.feature_axis)
    dec_array = pytree_utils.stack_pytree(dec_inputs, axis=self.feature_axis)
    enc_pos_encoding = pytree_utils.stack_pytree(
        self.encoder_positional_encodings_fn(inputs), axis=self.feature_axis)
    dec_pos_encoding = pytree_utils.stack_pytree(
        self.decoder_positional_encodings_fn(inputs), axis=self.feature_axis)
    input_ndims = set(
        x.ndim
        for x in [enc_array, dec_array, enc_pos_encoding, dec_pos_encoding]
        if x is not None)
    if input_ndims != {4}:
      raise ValueError(f'Expected all inputs have ndim=4, got {input_ndims=}')
    latents = self.encoder_tower(enc_array, None, enc_pos_encoding)
    # if dec_array is None, use latents as `inputs` and provide no `latents`.
    decoder_latents = None if dec_array is None else latents
    # if dec_array is None, use `latents`, otherwise use dec_array as `inputs`.
    dec_array = dec_array if dec_array is not None else latents
    outputs = self.decoder_tower(dec_array, decoder_latents, dec_pos_encoding)
    if outputs.ndim != 4:
      raise ValueError(f'Expected outputs with ndim=4, got {outputs.shape=}')
    return pytree_utils.unstack_to_pytree(
        outputs, self.output_shapes, axis=self.feature_axis
    )


@gin.register(denylist=['output_shapes'])
class ParallelMapping(hk.Module):
  """Maps a pytree to a pytree by additively compbining multiple mappings.

  Outputs of `mappings` must be compatible with each other.
  """

  def __init__(
      self,
      output_shapes: typing.Pytree,
      mappings: Sequence[MappingModule] = gin.REQUIRED,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.mapping_fns = [m(output_shapes) for m in mappings]

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    results = [mapping_fn(inputs) for mapping_fn in self.mapping_fns]
    return jax.tree_util.tree_map(lambda *args: sum(args), *results)
