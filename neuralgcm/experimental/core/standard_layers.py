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
"""Modules for standard NN layers: MLP, Convolutions, Pooling, etc."""

from __future__ import annotations

import itertools
from typing import Callable, Literal, Protocol, Sequence

from flax import nnx
from flax import typing as flax_typing
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import typing

Array = typing.Array

default_w_init = nnx.initializers.lecun_normal()
default_b_init = nnx.initializers.zeros_init()

ResizeMode = Literal['upsample', 'downsample']
ResizeMethod = Literal['bilinear', 'nearest', 'cubic', 'lanczos3', 'lanczos5']


class UnaryLayer(Protocol):
  """Protocol for neural network layers that transform `Array` --> `Array`.

  This is a protocol, so we can use any module that implements the same
  signature without complicating the inheretance hierarchy. Currently input and
  output sizes are assumed to be leading dimensions in layer input/output
  shapes, with the rest of the shapes being preserved.
  """

  def __init__(self, input_size: int, output_size: int, **kwargs):
    ...

  def __call__(self, inputs: jax.Array) -> jax.Array:
    ...

  @property
  def input_size(self) -> int:
    ...

  @property
  def output_size(self) -> int:
    ...


UnaryLayerFactory = Callable[..., UnaryLayer]


class Mlp(nnx.Module):
  """Multi-layer-perceptron."""

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      intermediate_sizes: Sequence[int],
      activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      w_init: nnx.initializers.Initializer = default_w_init,
      b_init: nnx.initializers.Initializer = default_b_init,
      use_bias: bool = True,
      w_init_final: nnx.initializers.Initializer | None = None,
      b_init_final: nnx.initializers.Initializer | None = None,
      activate_final: bool = False,
      dtype: typing.Dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    """Constructs a MLP layer.

    Args:
      input_size: size of the input array.
      output_size: size of the array returned by the last layer.
      intermediate_sizes: sequence of hidden layer sizes.
      activation: activation function to apply between linear layers.
      w_init: initializer for linear layer kernels.
      b_init: initializer for linear layer biases.
      use_bias: whether to use bias in linear layers.
      w_init_final: initializer for the final linear layer kernel.
      b_init_final: initializer for the final linear layer bias.
      activate_final: whether to activate outputs of the final layer.
      dtype: linear layer param dtype.
      rngs: rngs for linear layers.
    """
    self.input_size = input_size
    self.output_size = output_size
    output_sizes = tuple(intermediate_sizes) + (output_size,)
    if w_init_final is None:
      w_init_final = w_init
    if b_init_final is None:
      b_init_final = b_init
    self.activation = activation
    self.activate_final = activate_final
    output_sizes = tuple(output_sizes)
    input_sizes = (input_size,) + output_sizes[:-1]
    layers = []
    for i, (d_in, d_out) in enumerate(zip(input_sizes, output_sizes)):
      layers.append(
          nnx.Linear(
              in_features=d_in,
              out_features=d_out,
              kernel_init=w_init if i < len(input_sizes) - 1 else w_init_final,
              bias_init=b_init if i < len(input_sizes) - 1 else b_init_final,
              use_bias=use_bias,
              param_dtype=dtype,
              rngs=rngs,
          )
      )
    self.layers = tuple(layers)

  def __call__(
      self,
      inputs: jax.Array,
  ) -> jax.Array:
    """Applies Mlp to `inputs`."""
    num_layers = len(self.layers)
    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        out = self.activation(out)
    return out

  # TODO(dkochkov): Deprecate MlpUniform in favor of this classmethod.
  @classmethod
  def uniform(
      cls,
      input_size: int,
      output_size: int,
      *,
      hidden_size: int,
      hidden_layers: int,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
      w_init: nnx.initializers.Initializer = default_w_init,
      b_init: nnx.initializers.Initializer = default_b_init,
      use_bias: bool = True,
      w_init_final: nnx.initializers.Initializer | None = None,
      b_init_final: nnx.initializers.Initializer | None = None,
      activate_final: bool = False,
      dtype: typing.Dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    intermediate_sizes = (hidden_size,) * hidden_layers
    return cls(
        input_size=input_size,
        output_size=output_size,
        intermediate_sizes=intermediate_sizes,
        activation=activation,
        w_init=w_init,
        b_init=b_init,
        use_bias=use_bias,
        w_init_final=w_init_final,
        b_init_final=b_init_final,
        activate_final=activate_final,
        dtype=dtype,
        rngs=rngs,
    )


class MlpUniform(Mlp):
  """Multi-layer perceptron module with uniform layer sizes."""

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      hidden_size: int,
      n_hidden_layers: int,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
      w_init: nnx.initializers.Initializer = default_w_init,
      b_init: nnx.initializers.Initializer = default_b_init,
      use_bias: bool = True,
      w_init_final: nnx.initializers.Initializer | None = None,
      b_init_final: nnx.initializers.Initializer | None = None,
      activate_final: bool = False,
      dtype: typing.Dtype = jnp.float32,
      rngs: nnx.Rngs,
  ):
    intermediate_sizes = (hidden_size,) * n_hidden_layers
    super().__init__(
        input_size=input_size,
        output_size=output_size,
        intermediate_sizes=intermediate_sizes,
        activation=activation,
        w_init=w_init,
        b_init=b_init,
        use_bias=use_bias,
        w_init_final=w_init_final,
        b_init_final=b_init_final,
        activate_final=activate_final,
        dtype=dtype,
        rngs=rngs,
    )


def conv_dilated_ncw(
    inputs,
    kernel,
    strides,
    padding_lax,
    lhs_dilation,
    rhs_dilation,
    dimension_numbers,
    feature_group_count,
    precision,
):
  """A modified version of conv_general_dilated that uses NCW convention."""
  del dimension_numbers  # unused.
  return jax.lax.conv_general_dilated(
      inputs,
      kernel,
      strides,
      padding_lax,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      dimension_numbers=('NCW', 'WIO', 'NCW'),
      feature_group_count=feature_group_count,
      precision=precision,
  )


# TODO(dkochkov) Investigate performance of this layer.


class ConvLevel(nnx.Conv):
  """1D convolution in the NCW data format that preserves tail axis shape."""

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      kernel_size: int,
      strides: int = 1,
      padding: str = 'SAME',
      input_dilation: int = 1,
      kernel_dilation: int = 1,
      use_bias: bool = True,
      precision: flax_typing.PrecisionLike = None,
      w_init: nnx.initializers.Initializer = default_w_init,
      b_init: nnx.initializers.Initializer = default_b_init,
      rngs: nnx.Rngs,
  ):
    self.input_size = input_size
    self.output_size = output_size
    # ensures that `bias` broadcasts correctly.
    ncw_b_init = lambda key, shape, dtype: b_init(key, shape + (1,), dtype)
    super().__init__(
        in_features=input_size,
        out_features=output_size,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        input_dilation=input_dilation,
        kernel_dilation=kernel_dilation,
        use_bias=use_bias,
        precision=precision,
        kernel_init=w_init,
        bias_init=ncw_b_init,
        conv_general_dilated=conv_dilated_ncw,
        rngs=rngs,
    )


class CnnLevel(nnx.Module):
  """1D CNN in the NCW data format that preserves tail axis shape."""

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      channels: Sequence[int],
      kernel_sizes: int | Sequence[int],
      strides: int | Sequence[int] = 1,
      padding: str = 'SAME',
      input_dilation: int = 1,
      kernel_dilations: int | Sequence[int] = 1,
      use_bias: bool = True,
      precision: flax_typing.PrecisionLike = None,
      w_init: nnx.initializers.Initializer = default_w_init,
      b_init: nnx.initializers.Initializer = default_b_init,
      w_init_final: nnx.initializers.Initializer = default_w_init,
      b_init_final: nnx.initializers.Initializer = default_b_init,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
      activate_final: bool = False,
      rngs: nnx.Rngs,
  ):
    self.input_size = input_size
    self.output_size = output_size
    n_hidden = len(channels)
    n_total = n_hidden + 1
    if isinstance(kernel_sizes, int):
      kernel_sizes = [kernel_sizes] * n_total
    if isinstance(kernel_dilations, int):
      kernel_dilations = [kernel_dilations] * n_total
    channels = list(channels) + [output_size]
    if len(set([len(channels), len(kernel_sizes), len(kernel_dilations)])) != 1:
      raise ValueError(
          f'Missing kernel | dilation specs for {n_total=} '
          f'layers, got {kernel_sizes=}, {kernel_dilations=}.'
      )
    w_inits = [w_init] * n_hidden + [w_init_final]
    b_inits = [b_init] * n_hidden + [b_init_final]
    params = zip(channels, kernel_sizes, kernel_dilations, w_inits, b_inits)
    self.layers = []
    din = input_size
    for dout, kernel, dilation, w_init_i, b_init_i in params:
      self.layers.append(
          ConvLevel(
              input_size=din,
              output_size=dout,
              kernel_size=kernel,
              strides=strides,
              input_dilation=input_dilation,
              kernel_dilation=dilation,
              padding=padding,
              use_bias=use_bias,
              w_init=w_init_i,
              b_init=b_init_i,
              precision=precision,
              rngs=rngs,
          )
      )
      din = dout
    self.activation = activation
    self.activate_final = activate_final

  def __call__(self, inputs: jax.Array) -> jax.Array:
    out = inputs
    num_layers = len(self.layers)
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        out = self.activation(out)
    return out


class ConvLonLat(nnx.Module):
  """Two dimensional convolution with lon-lat padding and leading channel axis.

  This module maps inputs of shape [din, lon, lat] -> [dout, lon, lat],
  preserving the tail axes shape.
  """

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      use_bias: bool = True,
      kernel_size: tuple[int, int] = (3, 3),
      dilation: int = 1,
      rngs: nnx.Rngs,
  ):
    self.conv_layer = nnx.Conv(
        in_features=input_size,
        out_features=output_size,
        kernel_size=kernel_size,
        rngs=rngs,
        padding='valid',
        kernel_dilation=dilation,
        use_bias=use_bias,
    )
    self.pad_size = int((kernel_size[0] - 1) / 2 * dilation)

  def __call__(self, inputs: Array) -> Array:
    inputs = jnp.moveaxis(jnp.expand_dims(inputs, axis=0), 1, -1)
    inputs = jnp.pad(
        inputs,
        ((0, 0), (self.pad_size, self.pad_size), (0, 0), (0, 0)),
        mode='wrap',
    )
    inputs = jnp.pad(
        inputs,
        ((0, 0), (0, 0), (self.pad_size, self.pad_size), (0, 0)),
        mode='reflect',
    )
    outputs = self.conv_layer(inputs).squeeze(axis=0)
    return jnp.moveaxis(outputs, -1, 0)


@nnx_compat.dataclass
class Epd(nnx.Module):
  """EPD layer parameterized by encode/process/decode layers."""

  encode: UnaryLayer
  decode: UnaryLayer
  process_blocks: Sequence[UnaryLayer]
  post_encode_activation: Callable[[Array], Array] | None = None
  pre_decode_activation: Callable[[Array], Array] | None = None
  final_activation: Callable[[Array], Array] | None = None

  @property
  def input_size(self) -> int:
    return self.encode.input_size

  @property
  def output_size(self) -> int:
    return self.decode.output_size

  @classmethod
  def build_using_factories(
      cls,
      input_size: int,
      output_size: int,
      *,
      latent_size: int,
      num_process_blocks: int,
      encode_factory: UnaryLayerFactory,
      process_factory: UnaryLayerFactory,
      decode_factory: UnaryLayerFactory,
      post_encode_activation: Callable[[Array], Array] | None = None,
      pre_decode_activation: Callable[[Array], Array] | None = None,
      final_activation: Callable[[Array], Array] | None = None,
      rngs: nnx.Rngs,
  ) -> Epd:
    encode = encode_factory(input_size, latent_size, rngs=rngs)
    decode = decode_factory(latent_size, output_size, rngs=rngs)
    process_blocks = tuple([
        process_factory(latent_size, latent_size, rngs=rngs)
        for _ in range(num_process_blocks)
    ])
    return cls(
        encode=encode,
        decode=decode,
        process_blocks=process_blocks,
        post_encode_activation=post_encode_activation,
        pre_decode_activation=pre_decode_activation,
        final_activation=final_activation,
    )

  def __call__(self, inputs: Array) -> Array:
    """Applies all EPD layers to inputs."""
    encoded = self.encode(inputs)
    if self.post_encode_activation is not None:
      encoded = self.post_encode_activation(encoded)
    current = encoded
    for process_block in self.process_blocks:
      current = current + process_block(current)
    if self.pre_decode_activation is not None:
      current = self.pre_decode_activation(current)
    out = self.decode(current)
    if self.final_activation is not None:
      return self.final_activation(out)
    return out


@nnx_compat.dataclass
class Sequential(nnx.Module):
  """Layer that applies a collection of layers in sequence."""

  layers: Sequence[UnaryLayer]

  @property
  def input_size(self) -> int:
    return self.layers[0].input_size

  @property
  def output_size(self) -> int:
    return self.layers[-1].output_size

  @classmethod
  def build_using_factories(
      cls,
      input_size: int,
      output_size: int,
      *,
      intermediate_sizes: tuple[int | str, ...],
      layer_factories: Sequence[UnaryLayerFactory],
      rngs: nnx.Rngs,
  ) -> Sequential:
    replace_values = {'input_size': input_size, 'output_size': output_size}
    # Resolve string references in intermediate_sizes to actual values.
    resolved_intermediate_sizes = tuple(
        replace_values[s] if isinstance(s, str) else s
        for s in intermediate_sizes
    )
    in_out_sizes = resolved_intermediate_sizes + (output_size,)
    built_layers = []
    for (d_in, d_out), layer_factory in zip(
        itertools.pairwise(in_out_sizes), layer_factories, strict=True
    ):
      built_layers.append(layer_factory(d_in, d_out, rngs=rngs))
    return cls(layers=tuple(built_layers))

  def __call__(self, inputs: Array) -> Array:
    current = inputs
    for layer in self.layers:
      current = layer(current)
    return current


@nnx_compat.dataclass
class CnnLonLat(nnx.Module):
  """CNN that uses ConvLonLat layers and preserves tail axis shape."""

  conv_layers: Sequence[ConvLonLat]
  activation: Callable[[Array], Array]
  activation_final: Callable[[Array], Array] = lambda x: x

  @classmethod
  def build_using_factories(
      cls,
      input_size: int,
      output_size: int,
      *,
      hidden_size: int,
      hidden_layers: int,
      kernel_size: tuple[int, int] = (3, 3),
      dilation: int = 1,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
      activation_final: Callable[[Array], Array] = lambda x: x,
      use_bias: bool = True,
      rngs: nnx.Rngs,
  ) -> CnnLonLat:
    """Builds a ConvLonLatTower with the given parameters."""
    sizes = [input_size] + [hidden_size] * (hidden_layers + 1) + [output_size]
    layers = []
    for d_in, d_out in itertools.pairwise(sizes):
      layers.append(
          ConvLonLat(
              input_size=d_in,
              output_size=d_out,
              use_bias=use_bias,
              kernel_size=kernel_size,
              dilation=dilation,
              rngs=rngs,
          )
      )
    return cls(
        conv_layers=tuple(layers),
        activation=activation,
        activation_final=activation_final,
    )

  def __call__(self, inputs: Array) -> Array:
    for i, layer in enumerate(self.conv_layers):
      inputs = layer(inputs)
      if i != len(self.conv_layers) - 1:
        inputs = self.activation(inputs)
    return self.activation_final(inputs)


class ResizeLonLat(nnx.Module):
  """Layer to resize array inputs."""

  def __init__(
      self,
      resize_ratio: tuple[int, int] = (2, 2),
      *,
      mode: ResizeMode,
      method: ResizeMethod = 'bilinear',
  ):
    self.resize_ratio = resize_ratio
    self.method = method
    self.mode = mode

  def __call__(self, inputs: typing.Array):

    fixed_shape, lon_lat_shape = inputs.shape[:-2], inputs.shape[-2:]
    if self.mode == 'upsample':
      lon_lat_shape = jax.tree.map(
          lambda x, s: x * s, lon_lat_shape, self.resize_ratio
      )

    elif self.mode == 'downsample':
      div_fn = lambda x, y: divmod(x, y)[0]
      mod_fn = lambda x, y: divmod(x, y)[1]
      lon_lat_shape = jax.tree.map(div_fn, lon_lat_shape, self.resize_ratio)
      remainders = jax.tree.map(mod_fn, lon_lat_shape, self.resize_ratio)
      if remainders != (0, 0):
        raise ValueError(
            f'{lon_lat_shape=} is not divisible by {self.resize_ratio=}'
        )
    else:
      raise ValueError(f'Invalid mode: {self.mode}')
    return jax.image.resize(
        inputs, fixed_shape + lon_lat_shape, method=self.method
    )
