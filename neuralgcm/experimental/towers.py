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
"""Modules that define stacks of neural-net layers acting on spatial arrays."""

import functools
import itertools
from typing import Callable, Protocol, Sequence

from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental import standard_layers
from neuralgcm.experimental import typing

Array = typing.Array


class UnaryTower(Protocol):
  """Protocol for towers that transform array -> array preserving spatial shape.

  This is a protocol, so we can use any module that implements the same
  signature without complicating the inheretance hierarchy.
  """

  def __init__(self, input_size: int, output_size: int, **kwargs):
    ...

  def __call__(self, inputs: jax.Array) -> jax.Array:
    ...


UnaryTowerFactory = Callable[..., UnaryTower]


# TODO(dkochkov) consider passing spatial axes to parameterize # of vmap needed.


class ColumnTower(nnx.Module):
  """Tower that operates on latent/level axis with shared NN across all lat/lon.

  input shape: [nn_input_shape, lon, lat],
  output shape: [nn_output_shape, lon, lat].
  """

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      column_net_factory: nnx.Module,
      apply_remat: bool = False,
      rngs: nnx.Rngs,
  ):
    self.column_network = column_net_factory(input_size, output_size, rngs=rngs)
    self.apply_remat = apply_remat

  def __call__(self, inputs: Array) -> Array:
    """Applies Column tower to inputs."""
    vmap = functools.partial(nnx.vmap, in_axes=(None, -1), out_axes=-1)

    def vmap_fn(net, inputs):
      return net(inputs)

    mapped_column_net = vmap(vmap(vmap_fn))
    if self.apply_remat:
      mapped_column_net = nnx.remat(mapped_column_net)
    return mapped_column_net(self.column_network, inputs)


class EpdTower(nnx.Module):
  """EPD tower module parameterized by encode/process/decode towers."""

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      latent_size: int,
      num_process_blocks: int,
      encode_tower_factory: UnaryTowerFactory,
      process_tower_factory: UnaryTowerFactory,
      decode_tower_factory: UnaryTowerFactory,
      post_encode_activation: Callable[[Array], Array] | None = None,
      pre_decode_activation: Callable[[Array], Array] | None = None,
      final_activation: Callable[[Array], Array] | None = None,
      rngs: nnx.Rngs,
  ):
    self.input_size = input_size
    self.output_size = output_size
    self.latent_size = latent_size
    self.post_encode_activation = post_encode_activation
    self.pre_decode_activation = pre_decode_activation
    self.final_activation = final_activation
    self.encoder = encode_tower_factory(input_size, latent_size, rngs=rngs)
    self.decoder = decode_tower_factory(latent_size, output_size, rngs=rngs)
    self.process_towers = tuple([
        process_tower_factory(latent_size, latent_size, rngs=rngs)
        for _ in range(num_process_blocks)
    ])

  def __call__(self, inputs: Array) -> Array:
    """Applies EPD tower to inputs."""
    encoded = self.encoder(inputs)
    if self.post_encode_activation is not None:
      encoded = self.post_encode_activation(encoded)
    current = encoded
    for process_block in self.process_towers:
      current = current + process_block(current)
    if self.pre_decode_activation is not None:
      current = self.pre_decode_activation(current)
    out = self.decoder(current)
    if self.final_activation is not None:
      return self.final_activation(out)
    return out


class ChainedTower(nnx.Module):
  """Chains multiple towers together."""

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      intermediate_sizes: tuple[int | str, ...],
      tower_modules: Sequence[UnaryTowerFactory],
      rngs: nnx.Rngs,
  ):
    self.input_size = input_size
    self.output_size = output_size
    replace_values = {'input_size': input_size, 'output_size': output_size}
    replace_str = lambda s: replace_values[s]
    intermediate_sizes = tuple(
        replace_str(s) if isinstance(s, str) else s for s in intermediate_sizes
    )
    output_sizes = (*intermediate_sizes, output_size)
    input_sizes = (input_size, *intermediate_sizes)
    zip_vals = zip(input_sizes, output_sizes, tower_modules, strict=True)
    self.towers = tuple(
        tower_factory(d_in, d_out, rngs=rngs)
        for d_in, d_out, tower_factory in zip_vals
    )

  def __call__(self, inputs: Array) -> Array:
    current = inputs
    for tower in self.towers:
      current = tower(current)
    return current


class ConvLonLatTower(nnx.Module):
  """Two dimensional ConvNet tower module."""

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      num_hidden_units: int,
      num_hidden_layers: int,
      kernel_size: tuple[int, int] = (3, 3),
      dilation: int = 1,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
      use_bias: bool = True,
      activate_final: bool = False,
      apply_remat: bool = False,
      rngs: nnx.Rngs,
  ):
    self.apply_remat = apply_remat
    self.activation = activation
    self.activate_final = activate_final
    self.num_hidden_layers = num_hidden_layers
    self.conv_layers = []
    self.conv_layers.append(
        standard_layers.ConvLonLat(
            input_size=input_size,
            output_size=num_hidden_units,
            use_bias=use_bias,
            kernel_size=kernel_size,
            dilation=dilation,
            rngs=rngs,
        )
    )
    for _ in range(num_hidden_layers):
      self.conv_layers.append(
          standard_layers.ConvLonLat(
              input_size=num_hidden_units,
              output_size=num_hidden_units,
              use_bias=use_bias,
              kernel_size=kernel_size,
              dilation=dilation,
              rngs=rngs,
          )
      )
    self.conv_layers.append(
        standard_layers.ConvLonLat(
            input_size=num_hidden_units,
            output_size=output_size,
            use_bias=use_bias,
            kernel_size=kernel_size,
            dilation=dilation,
            rngs=rngs,
        )
    )

  def apply(self, inputs: Array) -> Array:
    for i, layer in enumerate(self.conv_layers):
      inputs = layer(inputs)
      if i != self.num_hidden_layers + 1 or self.activate_final:
        inputs = self.activation(inputs)
    return inputs

  def __call__(self, inputs: Array) -> Array:
    apply_fn = self.apply
    if self.apply_remat:
      apply_fn = nnx.remat(apply_fn)
    return apply_fn(inputs)


class ResNet(nnx.Module):
  """A ResNet tower module that uses ConvLonLatTower for each block.

  input size: number of input channels.
  output size: number of output channels.

  conv_block_num_channels: number of channels for output of the intermediate and
    final layer of conv block components.

  conv_block_num_hidden_layers: number of hidden layers in each conv block.
  kernel size: tuple of convolutional kernel used in all layers.
  dilations: tuple of dilations used if an int is passed all layers will use the
    same dilation.

  activation: activation function throughout the tower and subcomponents.
  use_bias: whether to use bias.
  apply_remat: whether to apply rematerialization during training.
  num_hidden_layers_skip_residual: from the final layer, how many additional
    layers to skip passing the residual connection.
  """

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      conv_block_num_channels: tuple[int, ...],
      conv_block_num_hidden_layers: int = 0,
      kernel_size: tuple[int, int] = (3, 3),
      dilations: int | tuple[int, ...] = 1,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
      use_bias: bool = True,
      apply_remat: bool = False,
      num_hidden_layers_skip_residual: int = 0,
      rngs: nnx.Rngs,
  ):

    if isinstance(dilations, int):
      dilations = tuple([dilations] * (len(conv_block_num_channels) + 1))

    in_through_hidden_channels = (input_size,) + conv_block_num_channels
    self.num_hidden_layers_skip_residual = num_hidden_layers_skip_residual
    self.activation = activation
    self.apply_remat = apply_remat

    self.convolutional_blocks = []
    self.projection_blocks = []
    for dilation, (din, dout) in zip(
        dilations[:-1],
        itertools.pairwise(in_through_hidden_channels),
        strict=True,
    ):
      self.convolutional_blocks.append(
          ConvLonLatTower(
              input_size=din,
              output_size=dout,
              num_hidden_units=dout,
              num_hidden_layers=conv_block_num_hidden_layers,
              use_bias=use_bias,
              kernel_size=kernel_size,
              dilation=dilation,
              activation=activation,
              activate_final=False,
              rngs=rngs,
          )
      )
      if din != dout:
        self.projection_blocks.append(
            standard_layers.ConvLonLat(
                input_size=din,
                output_size=dout,
                use_bias=False,
                kernel_size=(1, 1),
                dilation=1,
                rngs=rngs,
            )
        )
      else:
        self.projection_blocks.append(lambda x: x)

    self.final_convolutional_block = ConvLonLatTower(
        input_size=conv_block_num_channels[-1],
        output_size=output_size,
        num_hidden_units=output_size,
        num_hidden_layers=conv_block_num_hidden_layers,
        use_bias=use_bias,
        kernel_size=kernel_size,
        dilation=dilations[-1],
        activation=activation,
        activate_final=False,
        rngs=rngs,
    )

  def apply(self, inputs: Array) -> Array:
    carry = inputs
    for i, (convolution, projection) in enumerate(
        zip(self.convolutional_blocks, self.projection_blocks)
    ):
      residual_connection = projection(carry)
      carry = convolution(carry)
      if i < len(self.projection_blocks) - self.num_hidden_layers_skip_residual:
        carry += residual_connection
      carry = self.activation(carry)

    carry = self.final_convolutional_block(carry)
    return carry

  def __call__(self, inputs: Array) -> Array:
    apply_fn = self.apply
    if self.apply_remat:
      apply_fn = nnx.remat(apply_fn)
    return apply_fn(inputs)


class Unet(nnx.Module):
  """A U-Net tower module that uses ConvLonLatTower for each block."""

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      conv_block_num_channels: tuple[int, ...],
      downsampler=functools.partial(
          standard_layers.ResizeLonLat, mode='downsample'
      ),
      upsampler=functools.partial(
          standard_layers.ResizeLonLat, mode='upsample'
      ),
      reshape_ratio: tuple[int, int] = (2, 2),
      conv_block_hidden_layers: int = 0,
      kernel_size: tuple[int, int] = (3, 3),
      dilations: tuple[int, ...] | int = 1,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
      use_bias: bool = True,
      apply_remat: bool = False,
      rngs: nnx.Rngs,
  ):
    self.apply_remat = apply_remat
    if isinstance(dilations, int):
      dilations = tuple([dilations] * (len(conv_block_num_channels)))

    if len(conv_block_num_channels) != len(dilations):
      raise ValueError(
          'hidden_sizes and dilations must have the same length. Or dilations '
          'must be a single value.'
      )

    self.reshape_ratio = reshape_ratio

    self.downsampler = downsampler(reshape_ratio)
    self.upsampler = upsampler(reshape_ratio)

    in_to_center_channels = (input_size,) + conv_block_num_channels
    self.downward_blocks = []
    for i, din_dout in enumerate(itertools.pairwise(in_to_center_channels)):
      self.downward_blocks.append(
          ConvLonLatTower(
              input_size=din_dout[0],
              output_size=din_dout[1],
              num_hidden_units=din_dout[1],
              num_hidden_layers=conv_block_hidden_layers,
              use_bias=use_bias,
              kernel_size=kernel_size,
              dilation=dilations[i],
              activation=activation,
              activate_final=True,
              rngs=rngs,
          )
      )

    self.bottom_block = ConvLonLatTower(
        input_size=conv_block_num_channels[-1],
        output_size=conv_block_num_channels[-1],
        num_hidden_units=conv_block_num_channels[-1],
        num_hidden_layers=conv_block_hidden_layers,
        use_bias=use_bias,
        kernel_size=kernel_size,
        dilation=dilations[-1],
        activation=activation,
        activate_final=True,
        rngs=rngs,
    )

    self.upward_blocks = []
    reversed_dilation = tuple(reversed(dilations))
    for i, din_dout in enumerate(
        itertools.pairwise(in_to_center_channels[-1:0:-1]),
    ):
      self.upward_blocks.append(
          ConvLonLatTower(
              input_size=din_dout[0],
              output_size=din_dout[1],
              num_hidden_units=din_dout[1],
              num_hidden_layers=conv_block_hidden_layers,
              use_bias=use_bias,
              kernel_size=kernel_size,
              dilation=reversed_dilation[i],
              activation=activation,
              activate_final=True,
              rngs=rngs,
          )
      )

    self.upward_blocks.append(
        ConvLonLatTower(
            input_size=conv_block_num_channels[0],
            output_size=output_size,
            num_hidden_units=output_size,
            num_hidden_layers=conv_block_hidden_layers,
            use_bias=use_bias,
            kernel_size=kernel_size,
            dilation=dilations[0],
            activation=activation,
            activate_final=False,
            rngs=rngs,
        )
    )

  def apply(self, inputs):
    carry = inputs
    skip_connections = []
    for block in self.downward_blocks:
      carry = block(carry)
      skip_connections.append(carry)
      carry = self.downsampler(carry)
    carry = self.bottom_block(carry)
    for level, block in enumerate(self.upward_blocks, start=1):
      carry = self.upsampler(carry)
      carry += skip_connections[-level]
      carry = block(carry)
    return carry

  def __call__(self, inputs):
    apply_fn = self.apply
    if self.apply_remat:
      apply_fn = nnx.remat(apply_fn)
    return apply_fn(inputs)
