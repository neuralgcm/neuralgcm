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
"""Modules for transformer layers and related utilities."""

from __future__ import annotations

import abc
import dataclasses
import functools
from typing import Callable, Protocol, Sequence, Self

import einops
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental.core import boundaries
from neuralgcm.experimental.core import nnx_compat
from neuralgcm.experimental.core import spherical_transforms
from neuralgcm.experimental.core import standard_layers
from neuralgcm.experimental.core import typing
import numpy as np


Gating = Callable[[typing.Array, typing.Array], typing.Array]
default_kernel_init = nnx.initializers.lecun_normal()


class TransformerLayer(Protocol):
  """Protocol for transformer layers."""

  def __call__(
      self,
      inputs: typing.Array,
      latents: typing.Array | None = None,
      inputs_pos_encoding: typing.Array | None = None,
      latents_pos_encoding: typing.Array | None = None,
      mask: typing.Array | None = None,
  ) -> typing.Array:
    ...


class MultiHeadAttention(nnx.Module):
  """Adaptation of nnx.MultiHeadAttention with attention_bias input.

  This module is a fork of the default MultiHeadAttention implementation in
  `nnx`, where we removed components that are unlikely to be used in modeling of
  a dynamical system and added an explicit optional `attention_bias` argument to
  the `__call__` method to enable supplying positional biases.

  Attrs:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    in_features: int or tuple with number of input features.
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection.
    in_kv_features: number of input features for computing key and value.
    dtype: the dtype of the computation (default: infer from inputs and params)
    param_dtype: the dtype passed to parameter initializers (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    out_kernel_init: optional initializer for the kernel of the output Dense
      layer, if None, the kernel_init is used.
    bias_init: initializer for the bias of the Dense layers.
    out_bias_init: optional initializer for the bias of the output Dense layer,
      if None, the bias_init is used.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    normalize_qk: should QK normalization be applied (arxiv.org/abs/2302.05442).
    rngs: rng key.
  """

  def __init__(
      self,
      num_heads: int,
      in_features: int,
      qkv_features: int | None = None,
      out_features: int | None = None,
      in_kv_features: int | None = None,
      *,
      dtype: nnx.Dtype | None = None,
      param_dtype: nnx.Dtype = jnp.float32,
      precision: nnx.PrecisionLike = None,
      kernel_init: nnx.initializers.Initializer = default_kernel_init,
      out_kernel_init: nnx.initializers.Initializer | None = None,
      bias_init: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      out_bias_init: nnx.initializers.Initializer | None = None,
      use_bias: bool = True,
      attention_fn: Callable[..., jax.Array] = nnx.dot_product_attention,
      normalize_qk: bool = False,
      rngs: nnx.Rngs,
  ):
    self.num_heads = num_heads
    self.in_features = in_features
    self.qkv_features = (
        qkv_features if qkv_features is not None else in_features
    )
    self.out_features = (
        out_features if out_features is not None else in_features
    )
    self.in_kv_features = (
        in_kv_features if in_kv_features is not None else in_features
    )
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.out_kernel_init = out_kernel_init
    self.bias_init = bias_init
    self.out_bias_init = out_bias_init
    self.use_bias = use_bias
    self.attention_fn = attention_fn
    self.normalize_qk = normalize_qk

    if self.qkv_features % self.num_heads != 0:
      raise ValueError(
          f'Memory dimension ({self.qkv_features}) must be divisible by '
          f"'num_heads' heads ({self.num_heads})."
      )

    self.head_dim = self.qkv_features // self.num_heads

    linear_general = functools.partial(
        nnx.LinearGeneral,
        out_features=(self.num_heads, self.head_dim),
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    self.query = linear_general(self.in_features, rngs=rngs)
    self.key = linear_general(self.in_kv_features, rngs=rngs)
    self.value = linear_general(self.in_kv_features, rngs=rngs)

    self.query_ln: nnx.LayerNorm | None
    self.key_ln: nnx.LayerNorm | None
    if self.normalize_qk:
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      self.query_ln = nnx.LayerNorm(
          self.head_dim,
          use_bias=False,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          rngs=rngs,
      )
      self.key_ln = nnx.LayerNorm(
          self.head_dim,
          use_bias=False,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          rngs=rngs,
      )
    else:
      self.query_ln = None
      self.key_ln = None

    self.out = nnx.LinearGeneral(
        in_features=(self.num_heads, self.head_dim),
        out_features=self.out_features,
        axis=(-2, -1),
        kernel_init=self.out_kernel_init or self.kernel_init,
        bias_init=self.out_bias_init or self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs_q: jax.Array,
      inputs_k: jax.Array | None = None,
      inputs_v: jax.Array | None = None,
      attention_bias: jax.Array | None = None,
      mask: jax.Array | None = None,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention with optional attention bias term and
    projects the results to an output vector.

    If both inputs_k and inputs_v are None, they will both copy the value of
    inputs_q (self attention).
    If only inputs_v is None, it will copy the value of inputs_k.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., length, features]`.
      inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
        inputs_k will copy the value of inputs_q.
      inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
        inputs_v will copy the value of inputs_k.
      attention_bias: values the bias to be added to the attention calculation.
        Must broadcast to `[batch_sizes..., n_heads, query_length,
        key/value_length]`.
      mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`. Attention weights are masked out if their
        corresponding mask value is `False`.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if inputs_k is None:
      if inputs_v is not None:
        raise ValueError(
            '`inputs_k` cannot be None if `inputs_v` is not None. To have both'
            ' `inputs_k` and `inputs_v` be the same value, pass in the value to'
            ' `inputs_k` and leave `inputs_v` as None.'
        )
      inputs_k = inputs_q
    if inputs_v is None:
      inputs_v = inputs_k

    if inputs_q.shape[-1] != self.in_features:
      raise ValueError(
          f'Incompatible input dimension, got {inputs_q.shape[-1]} '
          f'but module expects {self.in_features}.'
      )

    query = self.query(inputs_q)
    key = self.key(inputs_k)
    value = self.value(inputs_v)

    if self.normalize_qk:
      assert self.query_ln is not None and self.key_ln is not None
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = self.query_ln(query)
      key = self.key_ln(key)

    x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        bias=attention_bias,
        dtype=self.dtype,
        precision=self.precision,
    )
    out = self.out(x)  # post-attention projection.
    return out


@nnx_compat.dataclass
class TransformerBase(nnx.Module, abc.ABC):
  """Base class defining core transformer attributes and helper methods.

  This base class provides forward pass implementation and checks on
  default architectural parameters. This module repesents a sequence of
  transformer blocks, with details of how the model attends over input
  dimensions defined by the subclasses. In addition to standard components this
  module includes optional layer normalization and gating mechanism, resembling
  GTrXL transformer from https://arxiv.org/pdf/1910.06764.pdf.

  Similar to other networks defined in standard_layers.py the __call__ method
  expect a leading channel dimension followed by spatial shape (while the order)
  is generally reversed when applying MultiHeadAttention. The computation
  pattern is defined via several abstract methods that must be implemented by
  the subclasses:

    1. `rearrange_inputs`, `rearrange_latents` `rearrange_outputs` implement how
       arrays are transformed between [C, spatial_axes] to [B, L, C] format
       with C representing channels, L - sequence length, B batches of sequences
       and spatial_axes representing non-latent dimensions of the input array.
    2. `attention_key_source`, `attention_value_source` define how key and value
       source arrays are computed from all available inputs.
    3. `apply_attention` that defines how a given MultiHeadAttention module
       should be applied to provided `q, k, v` input, supplemented by positional
       encodings, attention mask and index of the attention block.

  Attributes:
    input_size: The number of input channels.
    output_size: The number of output channels.
    n_layers: The number of transformer blocks (attention + dense layers).
    attentions: Sequence of `MultiHeadAttention` modules.
    dense_layers: Sequence of `UnaryLayer`s applied after attention layers.
    layer_norms: Optional sequence of `nnx.LayerNorm` modules, applied before
      attention and dense layers.
    gating: A single gating function or a sequence of gating functions applied
      to residual connections. Defaults to a no skip connections.
  """

  input_size: int = dataclasses.field(init=False)
  output_size: int = dataclasses.field(init=False)
  n_layers: int = dataclasses.field(init=False)
  attentions: Sequence[MultiHeadAttention]
  dense_layers: Sequence[standard_layers.UnaryLayer]
  layer_norms: Sequence[nnx.LayerNorm] | None
  gating: Sequence[Gating] | Gating = dataclasses.field(
      kw_only=True, default=lambda skip, x: x
  )

  def __post_init__(self):
    n_layers = len(self.attentions)
    if n_layers < 1:
      raise ValueError(f'{type(self)} got empty sequence of attention layer.')
    self.input_size = self.attentions[0].in_features
    self.output_size = self.attentions[-1].out_features
    self.n_layers = n_layers
    if len(self.dense_layers) != n_layers:
      raise ValueError(
          f'{type(self)} got {len(self.dense_layers)=} != {n_layers=}.'
      )
    if isinstance(self.gating, Sequence) and len(self.gating) != 2 * n_layers:
      raise ValueError(
          f'{type(self)} got {self.gating=} that is not a sequence of '
          f'2x {n_layers} + 1 == {2 * n_layers + 1} gating functions/layers.'
      )

  @abc.abstractmethod
  def rearrange_inputs(self, inputs: jax.Array) -> jax.Array:
    """Converts `inputs` from [C, spatial_axes...] to [B, L, C] format.

    B: batch of sequences, L: sequence length, C: channels.

    Args:
      inputs: array to be rearranged.
    """
    ...

  @abc.abstractmethod
  def rearrange_latents(self, latents: jax.Array) -> jax.Array:
    """Converts `latents` from [C, spatial_axes...] to [B, L, C] format.

    B: batch of sequences, L: sequence length, C: channels.
    Args:
      latents: array to be rearranged.
    """
    ...

  @abc.abstractmethod
  def rearrange_outputs(
      self,
      outputs: jax.Array,
      inputs_shape: tuple[int, ...],
  ) -> jax.Array:
    """Converts `outputs` from [B, L, C] back to the input's spatial format.

    Args:
      outputs: Outputs in the [B, L, C] format (Batch, Length, Channels).
      inputs_shape: Original shape of the inputs, used to restore spatial dims.

    Returns:
      `outputs` rearranged to the original input's spatial format.
    """
    ...

  @abc.abstractmethod
  def attention_key_source(
      self,
      x: typing.Array,
      z: typing.Array,
      x_pos_encoding: typing.Array,
      z_pos_encoding: typing.Array,
  ) -> typing.Array:
    """Computes the key source for attention layers.

    `x` are the evolving inputs throughout the blocks; `z` are static latents.

    Args:
      x: Current evolving inputs to the transformer block.
      z: Static latent tokens, if provided.
      x_pos_encoding: Positional encodings for inputs `x`.
      z_pos_encoding: Positional encodings for latents `z`.

    Returns:
      Array to be used as keys for attention.
    """
    ...

  @abc.abstractmethod
  def attention_value_source(
      self,
      x: typing.Array,
      z: typing.Array,
      x_pos_encoding: typing.Array,
      z_pos_encoding: typing.Array,
  ) -> typing.Array:
    """Computes the value source for attention layers.

    `x` are the evolving inputs throughout the blocks; `z` are static latents.

    Args:
      x: Current evolving inputs to the transformer block.
      z: Static latent tokens, if provided.
      x_pos_encoding: Positional encodings for inputs `x`.
      z_pos_encoding: Positional encodings for latents `z`.

    Returns:
      Array to be used as values for attention.
    """
    ...

  @abc.abstractmethod
  def apply_attention(
      self,
      attention: MultiHeadAttention,
      query: typing.Array,
      key: typing.Array,
      value: typing.Array,
      mask: typing.Array,
      query_pos_encoding: typing.Array,
      kv_pos_encoding: typing.Array,
      layer_idx: int,
  ) -> typing.Array:
    """Applies the `attention_module` to the given query, key, and value.

    Subclasses implement the specific attention strategy, potentially adding
    positional biases or performing windowing.

    Args:
      attention: The `MultiHeadAttention` module to apply.
      query: Query tensor in sequence format [B, Lq, C].
      key: Key tensor in sequence format [B, Lkv, C], or None.
      value: Value tensor in sequence format [B, Lkv, C], or None.
      mask: Attention mask, or None.
      query_pos_encoding: Positional encodings for `query`.
      kv_pos_encoding: Positional encodings for `key`/`value`.
      layer_idx: Index of the current attention layer.

    Returns:
      The output of the attention module.
    """
    ...

  def __call__(
      self,
      inputs: typing.Array,
      latents: typing.Array | None = None,
      inputs_pos_encoding: typing.Array | None = None,
      latents_pos_encoding: typing.Array | None = None,
      mask: typing.Array | None = None,
  ) -> typing.Array:
    """Applies a sequence of transformer blocks to the inputs.

    The method processes `inputs` through multiple transformer layers.
    If `latents` are provided, they can be used as a source for keys/values
    in attention, as defined by `attention_key_source` and
    `attention_value_source` in subclasses. Positional encodings, if supplied,
    are passed to attention mechanisms. An optional mask can be used to prevent
    attention to certain positions.

    Args:
      inputs: Input array of shape [C, spatial_axes...].
      latents: Optional latent array with leading channel dimension.
      inputs_pos_encoding: Optional positional encodings for inputs.
      latents_pos_encoding: Optional positional encodings for latents.
      mask: Optional attention mask.

    Returns:
      The processed array, in the same spatial format as `inputs`.
    """
    x = self.rearrange_inputs(inputs)
    z = None if latents is None else self.rearrange_latents(latents)
    x_pos_enc, z_pos_enc = inputs_pos_encoding, latents_pos_encoding
    x_dense = None
    layer_norms = self.layer_norms or [lambda x: x] * self.n_layers
    gates = (
        self.gating
        if isinstance(self.gating, Sequence)
        else [self.gating] * (2 * self.n_layers + 1)
    )
    parts = zip(self.attentions, self.dense_layers, layer_norms, strict=True)
    for i, (attention, dense, maybe_layer_norm) in enumerate(parts):
      x = x if x_dense is None else gates[2 * i](x, x_dense)
      x_norm = maybe_layer_norm(x)
      keys = self.attention_key_source(x_norm, z, x_pos_enc, z_pos_enc)
      values = self.attention_value_source(x_norm, z, x_pos_enc, z_pos_enc)
      x_attn = self.apply_attention(
          attention_module=attention,
          query=x_norm,
          key=keys,
          value=values,
          query_pos_encoding=x_pos_enc,
          kv_pos_encoding=z_pos_enc,
          mask=mask,
          layer_idx=i,
      )
      x = gates[2 * i + 1](x, x_attn)
      x_dense = dense(x)

    x = gates[-1](x, x_dense)
    return self.rearrange_outputs(x, inputs_shape=inputs.shape)

  @classmethod
  def build_args_using_factories(
      cls,
      input_size: int,
      output_size: int | None = None,
      *,
      intermediate_sizes: tuple[int, ...],
      num_heads: int,
      qkv_features: int | None = None,
      use_bias_in_attention: bool = True,
      use_layer_norm: bool = True,
      # activation: Callable[[typing.Array], typing.Array] = jax.nn.gelu,
      gating: Sequence[Gating] | Gating = lambda skip, x: x,
      dense_factory: standard_layers.UnaryLayerFactory,
      w_init=nnx.initializers.xavier_uniform(),
      b_init=nnx.initializers.zeros,
      rngs: nnx.Rngs,
  ):
    """Prepares arguments for `TransformerBase` constructor.

    This method constructs the sequences of attention layers, dense layers,
    layer norms, and gating configurations based on the provided factories
    and parameters. It's primarily a helper for `build_using_factories`.

    Args:
      input_size: Number of input channels.
      output_size: Number of output channels. Defaults to `input_size`.
      intermediate_sizes: Tuple of intermediate channel numbers for each layer.
      num_heads: Number of attention heads.
      qkv_features: Channels for query/key/value. If None and `d_in` is input
        features to attention, it's `d_in // num_heads`.
      use_bias_in_attention: If True, `MultiHeadAttention` includes bias.
      use_layer_norm: If True, `LayerNorm` is applied.
      gating: Gating function(s). Defaults to identity.
      dense_factory: Factory for dense layers post-attention.
      w_init: Kernel initializer for `MultiHeadAttention`.
      b_init: Bias initializer for `MultiHeadAttention`.
      rngs: JAX PRNG keys.

    Returns:
      A tuple containing (attentions, dense_layers, layer_norms, gating)
      suitable for initializing a `TransformerBase` instance.
    """
    if output_size is None:
      output_size = input_size
    input_sizes = (input_size,) + tuple(intermediate_sizes)
    output_sizes = tuple(intermediate_sizes) + (output_size,)
    attentions = []
    dense_layers = []
    layer_norms = []
    for i, (d_in, d_out) in enumerate(zip(input_sizes, output_sizes)):
      if qkv_features is None:
        if d_in % num_heads != 0:
          raise ValueError(
              f'{d_in=} at layer={i} is not divisible by {num_heads=}'
              ' which is required if qkv_features is not specified.'
          )
        qkv_features = d_in // num_heads
      if use_layer_norm:
        layer_norms.append(nnx.LayerNorm(d_in, rngs=rngs))
      attentions.append(
          MultiHeadAttention(
              num_heads=num_heads,
              in_features=d_in,
              qkv_features=qkv_features,
              out_features=d_out,
              use_bias=use_bias_in_attention,
              kernel_init=w_init,
              bias_init=b_init,
              rngs=rngs,
          )
      )
      dense_layers.append(dense_factory(d_out, d_out, rngs=rngs))
    layer_norms = layer_norms if layer_norms else None
    return (attentions, dense_layers, layer_norms, gating)

  @classmethod
  def build_using_factories(
      cls,
      input_size: int,
      output_size: int | None = None,
      *,
      intermediate_sizes: tuple[int, ...],
      num_heads: int,
      qkv_features: int | None = None,
      use_bias_in_attention: bool = True,
      use_layer_norm: bool = True,
      # activation: Callable[[typing.Array], typing.Array] = jax.nn.gelu,
      gating: Sequence[Gating] | Gating = lambda skip, x: x,
      dense_factory: standard_layers.UnaryLayerFactory,
      w_init=nnx.initializers.xavier_uniform(),
      b_init=nnx.initializers.zeros,
      rngs: nnx.Rngs,
  ) -> Self:
    """Generates standard attributes of TransformerBase class.

    Args:
      input_size: number of input channels.
      output_size: number of output channels.
      intermediate_sizes: tuple of intermediate channel numbers.
      num_heads: number of attention heads to use.
      qkv_features: number of channels used for query/key/value representations.
      use_bias_in_attention: whether to include bias term in MultiHeadAttention.
      use_layer_norm: whether to use layer normalization.
      gating: sequence of 2n+1 or single gating function. Default is no gating.
      dense_factory: factory for generating dense layers that follow attention.
      w_init: kernel initializer for the MultiHeadAttention modules.
      b_init: bias initializer for the MultiHeadAttention modules.
      rngs: random number generator for parameter initialization.
    """
    attentions, dense_layers, layer_norms, gating = (
        cls.build_args_using_factories(
            input_size=input_size,
            output_size=output_size,
            intermediate_sizes=intermediate_sizes,
            num_heads=num_heads,
            qkv_features=qkv_features,
            use_bias_in_attention=use_bias_in_attention,
            use_layer_norm=use_layer_norm,
            gating=gating,
            dense_factory=dense_factory,
            w_init=w_init,
            b_init=b_init,
            rngs=rngs,
        )
    )
    return cls(
        attentions=attentions,
        dense_layers=dense_layers,
        layer_norms=layer_norms,
        gating=gating,
    )


class TransformerBlocks(TransformerBase):
  """A standard sequence of Transformer blocks."""

  def rearrange_inputs(self, inputs: jax.Array) -> jax.Array:
    """Transposes inputs for attention: [C, L] -> [L, C]."""
    return jnp.transpose(inputs)

  def rearrange_latents(self, latents: jax.Array) -> jax.Array:
    """Transposes latents for attention: [C, L] -> [L, C]."""
    return jnp.transpose(latents)

  def rearrange_outputs(self, outputs: jax.Array, inputs_shape) -> jax.Array:
    del inputs_shape  # unused.
    return jnp.transpose(outputs)

  def attention_key_source(
      self,
      x: typing.Array,
      z: typing.Array | None,
      x_pos_encoding: typing.Array | None,
      z_pos_encoding: typing.Array | None,
  ) -> typing.Array | None:
    """Returns latents `z` as the key source if available."""
    # TODO(dkochkov): Consider adding class attributes that choose alternatives.
    return z

  def attention_value_source(
      self,
      x: typing.Array,
      z: typing.Array | None,
      x_pos_encoding: typing.Array | None,
      z_pos_encoding: typing.Array | None,
  ) -> typing.Array | None:
    """Returns latents `z` as the value source if available."""
    # TODO(dkochkov): Consider adding class attributes that choose alternatives.
    return z

  def apply_attention(
      self,
      attention: MultiHeadAttention,
      query: typing.Array,
      key: typing.Array | None,
      value: typing.Array | None,
      mask: typing.Array | None,
      query_pos_encoding: typing.Array | None,
      kv_pos_encoding: typing.Array | None,
      layer_idx: int,
  ) -> typing.Array:
    """Applies attention to query, key, value inputs."""
    del layer_idx, query_pos_encoding, kv_pos_encoding  # unused.
    return attention(query, key, value, mask=mask)


@nnx_compat.dataclass
class WindowTransformerBlocks(TransformerBase):
  """Transformer blocks that apply attention within windows."""

  inputs_bc: boundaries.BoundaryCondition = dataclasses.field(kw_only=True)
  kv_bc: boundaries.BoundaryCondition = dataclasses.field(kw_only=True)
  inputs_window_shape: tuple[int, ...]
  kv_window_shape: tuple[int, ...]
  relative_bias_net: standard_layers.UnaryLayer
  shift_windows: bool = False

  def _window_rearrange_args(
      self,
      array_shape: tuple[int, ...],
      window_shape: tuple[int, ...],
  ) -> tuple[str, str, dict[str, int]]:
    """Generates einops patterns and shape dicts for windowing/unwindowing.

    Calculates patterns for rearranging an array with `array_shape` into
    windows of `window_shape`, and vice-versa.

    Args:
      array_shape: Shape of the array to be windowed (e.g., [C, H, W]).
      window_shape: Shape of the windows (e.g., [h_win, w_win]).

    Returns:
      A tuple containing:
        - spatial_pattern: einops pattern for the original spatial layout.
        - window_pattern: einops pattern for the windowed layout.
        - shape_kwargs: Dictionary of dimension sizes for einops.
    """
    spatial_shape = array_shape[1:]  # no windowing on channel dimension.
    windows_divmod = [divmod(x, w) for x, w in zip(spatial_shape, window_shape)]
    if any([x[1] for x in windows_divmod]):
      raise ValueError(f'{spatial_shape=} is incompatible with {window_shape=}')
    n_windows = [x[0] for x in windows_divmod]
    n_names = [f'n{i}' for i in range(len(n_windows))]
    w_names = [f'w{i}' for i in range(len(n_windows))]
    n_w_zip = zip(n_names, w_names)
    # spatial_pattern represent spatial_shape: 'c (n0 w0) (n1 w1)'.
    spatial_pattern = 'c' + ' '.join([f'({n} {w})' for n, w in n_w_zip])
    # window_patter represent grouped shape: '(n0 n1 ...) (w0 w1 ...) c'.
    window_patter = f"({' '.join(n_names)}) ({' '.join(w_names)}) c"
    shape_kwargs = {name: size for name, size in zip(n_names, n_windows)} | {
        name: size for name, size in zip(w_names, window_shape)
    }
    return spatial_pattern, window_patter, shape_kwargs

  def _to_windows(
      self,
      array: typing.Array,
      window_shape: tuple[int, ...],
      bc: boundaries.BoundaryCondition,
      shifts: tuple[int, ...] | None = None,
      return_pad_sizes: bool = False,
  ) -> typing.Array:
    """Pads and rearranges an array into non-overlapping windows.

    Optionally shifts the array before windowing. Can return padding sizes which
    can be needed to trim the windows after shifting.

    Args:
      array: Input array, typically [C, spatial_dims...].
      window_shape: Desired shape of spatial windows.
      bc: Boundary condition for padding.
      shifts: Optional shifts to apply before windowing (e.g., for SWIN).
      return_pad_sizes: If True, also returns the padding sizes applied.

    Returns:
      Windowed array [num_windows, window_volume, C], or
      (windowed_array, pad_sizes) if `return_pad_sizes` is True.
    """
    if shifts is None:
      shifts = tuple(0 for _ in window_shape)
    pad_sizes = [
        ((w - s) % w, (x - s) % w)
        for x, w, s in zip(array.shape[1:], window_shape, shifts)
    ]

    @nnx.vmap(in_axes=(None, 0))  # vmap over channel dimension.
    def pad(bc, x):
      return bc.pad_array(x, pad_sizes)

    padded_array = pad(bc, array)
    space_pattern, window_patter, shape_kwargs = self._window_rearrange_args(
        padded_array.shape, window_shape
    )
    result = einops.rearrange(
        padded_array, f'{space_pattern} -> {window_patter}', **shape_kwargs
    )
    if return_pad_sizes:
      return result, pad_sizes
    return result

  def _from_windows(
      self,
      array: typing.Array,
      window_shape: tuple[int, ...],
      bc: boundaries.BoundaryCondition,
      spatial_shape: tuple[int, ...],
      shifts: tuple[int, ...] | None = None,
  ) -> typing.Array:
    """Rearranges a windowed array back to a spatial shape."""
    if shifts is None:
      shifts = tuple(0 for _ in window_shape)
    space_pattern, window_patter, shape_kwargs = self._window_rearrange_args(
        (0,) + spatial_shape, window_shape
    )
    spatial = einops.rearrange(
        array, f'{window_patter} -> {space_pattern}', **shape_kwargs
    )
    pad_sizes = [
        ((w - s) % w, (x - s) % w)
        for x, w, s in zip(spatial_shape, window_shape, shifts)
    ]

    @nnx.vmap(in_axes=(None, 0))  # vmap over channel dimension.
    def trim(bc, x):
      return bc.trim_array(x, pad_sizes)

    return trim(bc, spatial)

  def _attention_bias(
      self,
      q_pe_windows: jax.Array,
      k_pe_windows: jax.Array,
  ) -> jax.Array | None:
    """Generates the relative position bias values."""
    if self.relative_bias_net is None:
      return None
    # windows have shapes: [B, WINDOW_SIZE, D], D - positional encoding size.
    compute_diffs = lambda x, y: (x[:, None] - y[None, :])
    compute_diffs = jax.vmap(jax.vmap(compute_diffs, in_axes=-1, out_axes=-1))
    pe_window_pairs = compute_diffs(q_pe_windows, k_pe_windows)
    # pe_window_pairs is [B, Q_WINDOW_SIZE, KV_WINDOW_SIZE, D]

    @nnx.vmap(in_axes=(None, 0), out_axes=0)  # over B
    @nnx.vmap(in_axes=(None, 0), out_axes=0)  # over Q_WINDOW_SIZE
    @nnx.vmap(in_axes=(None, 0), out_axes=0)  # over KV_WINDOW_SIZE
    def get_attention_bias(net, x):
      return net(x)

    bias = get_attention_bias(self.relative_bias_net, pe_window_pairs)
    # dot_attention requires shape (B, NUM_HEADS, Q_WINDOW_SIZE, KV_WINDOW_SIZE)
    return bias.transpose(0, 3, 1, 2)

  def rearrange_inputs(self, inputs: jax.Array) -> jax.Array:
    return self._to_windows(inputs, self.inputs_window_shape, self.inputs_bc)

  def rearrange_latents(self, latents: jax.Array) -> jax.Array:
    return self._to_windows(latents, self.kv_window_shape, self.kv_bc)

  def rearrange_outputs(
      self, outputs: jax.Array, inputs_shape: tuple[int, ...]
  ) -> jax.Array:
    return self._from_windows(
        outputs,
        self.inputs_window_shape,
        self.inputs_bc,
        spatial_shape=inputs_shape[1:],
    )

  def attention_key_source(self, x, z, x_pos_encoding, z_pos_encoding):
    return z  # Parameterize by class attributes?

  def attention_value_source(self, x, z, x_pos_encoding, z_pos_encoding):
    return z  # Parameterize by class attributes?

  def apply_attention(
      self,
      attention,
      query,
      key,
      value,
      mask,
      query_pos_encoding,
      kv_pos_encoding,
      layer_idx,
  ):
    if kv_pos_encoding is None:
      kv_pos_encoding = query_pos_encoding

    q_shape, kv_shape = query_pos_encoding.shape, kv_pos_encoding.shape
    # Transform positional encodings to window format.
    if layer_idx % 2 == 0 or not self.shift_windows:
      q_pe_windows, q_pad_sizes = self._to_windows(
          query_pos_encoding,
          self.inputs_window_shape,
          self.inputs_bc,
          return_pad_sizes=True,
      )
      kv_pe_windows = self._to_windows(
          kv_pos_encoding, self.kv_window_shape, self.kv_bc
      )
      attention_bias = self._attention_bias(q_pe_windows, kv_pe_windows)
      result = attention(
          query, key, value, mask=mask, attention_bias=attention_bias
      )
    else:  # if shift_windows is True and odd layer - apply shift to windows.
      q_pe_windows, q_pad_sizes = self._to_windows(
          query_pos_encoding,
          self.inputs_window_shape,
          self.inputs_bc,
          shifts=(w // 2 for w in self.inputs_window_shape),
          return_pad_sizes=True,
      )
      kv_pe_windows = self._to_windows(
          kv_pos_encoding,
          self.kv_window_shape,
          self.kv_bc,
          shifts=(w // 2 for w in self.kv_window_shape),
      )
      attention_bias = self._attention_bias(q_pe_windows, kv_pe_windows)

      query = self._from_windows(
          query,
          self.inputs_window_shape,
          self.inputs_bc,
          spatial_shape=q_shape[1:],
      )
      query = self._to_windows(
          query,
          self.inputs_window_shape,
          self.inputs_bc,
          shifts=(w // 2 for w in self.inputs_window_shape),
      )  # move back to windows, but with a shift.
      if mask is not None:
        mask = self._from_windows(
            mask, self.kv_window_shape, self.kv_bc, spatial_shape=kv_shape[1:]
        )
        mask = self._to_windows(
            mask,
            self.kv_window_shape,
            self.kv_bc,
            shifts=(w // 2 for w in self.kv_window_shape),
        )
      if key is not None:
        key = self._from_windows(
            key, self.kv_window_shape, self.kv_bc, spatial_shape=kv_shape[1:]
        )
        key = self._to_windows(
            key,
            self.kv_window_shape,
            self.kv_bc,
            shifts=(w // 2 for w in self.kv_window_shape),
        )
      if value is not None:
        value = self._from_windows(
            value, self.kv_window_shape, self.kv_bc, spatial_shape=kv_shape[1:]
        )
        value = self._to_windows(
            value,
            self.kv_window_shape,
            self.kv_bc,
            shifts=(w // 2 for w in self.kv_window_shape),
        )
      result = attention(
          query, key, value, mask=mask, attention_bias=attention_bias
      )

    q_padded_shape = tuple(
        s + sum(padding) for s, padding in zip(q_shape[1:], q_pad_sizes)
    )
    result = self._from_windows(
        result,
        self.inputs_window_shape,
        self.inputs_bc,
        spatial_shape=q_padded_shape,
    )

    @nnx.vmap(in_axes=(None, 0))
    def _trim(bc, x):
      return bc.trim_array(x, q_pad_sizes)

    result = _trim(self.inputs_bc, result)
    result = self._to_windows(result, self.inputs_window_shape, self.inputs_bc)
    return result

  @classmethod
  def build_using_factories(
      cls,
      input_size: int,
      output_size: int | None = None,
      *,
      inputs_bc: boundaries.BoundaryCondition,
      kv_bc: boundaries.BoundaryCondition | None = None,
      inputs_window_shape: tuple[int, ...],
      kv_window_shape: tuple[int, ...] | None = None,
      relative_bias_net: standard_layers.UnaryLayer,
      shift_windows: bool = False,
      intermediate_sizes: tuple[int, ...],
      num_heads: int,
      qkv_features: int | None = None,
      use_bias_in_attention: bool = True,
      use_layer_norm: bool = True,
      # activation: Callable[[typing.Array], typing.Array] = jax.nn.gelu,
      gating: Sequence[Gating] | Gating = lambda skip, x: x,
      dense_factory: standard_layers.UnaryLayerFactory,
      w_init=nnx.initializers.xavier_uniform(),
      b_init=nnx.initializers.zeros,
      rngs: nnx.Rngs,
  ) -> Self:
    """Constructs WindowTransformerBlocks parameterized by input/output sizes.

    Args:
      input_size: number of input channels.
      output_size: number of output channels. Defaults to `input_size`.
      inputs_bc: Boundary condition for input windowing (query).
      kv_bc: Boundary condition for key/value. Defaults to `inputs_bc`.
      inputs_window_shape: Shape of windows for inputs (query).
      kv_window_shape: Shape of windows for key/value. Defaults to
        `inputs_window_shape`.
      relative_bias_net: Network for computing relative positional bias from
        positional encodings.
      shift_windows: If True, windows are shifted in alternating layers.
      intermediate_sizes: tuple of intermediate channel numbers.
      num_heads: number of attention heads to use.
      qkv_features: number of channels used for query/key/value representations.
      use_bias_in_attention: whether to include bias term in MultiHeadAttention.
      use_layer_norm: whether to use layer normalization.
      gating: sequence of 2n+1 or single gating function. Default is no gating.
      dense_factory: factory for generating dense layers that follow attention.
      w_init: kernel initializer for the MultiHeadAttention modules.
      b_init: bias initializer for the MultiHeadAttention modules.
      rngs: random number generator for parameter initialization.
    """
    attentions, dense_layers, layer_norms, gating = (
        TransformerBase.build_args_using_factories(
            input_size=input_size,
            output_size=output_size,
            intermediate_sizes=intermediate_sizes,
            num_heads=num_heads,
            qkv_features=qkv_features,
            use_bias_in_attention=use_bias_in_attention,
            use_layer_norm=use_layer_norm,
            gating=gating,
            dense_factory=dense_factory,
            w_init=w_init,
            b_init=b_init,
            rngs=rngs,
        )
    )
    if kv_window_shape is None:
      kv_window_shape = inputs_window_shape
    if kv_bc is None:
      kv_bc = inputs_bc
    return cls(
        attentions=attentions,
        dense_layers=dense_layers,
        layer_norms=layer_norms,
        gating=gating,
        inputs_bc=inputs_bc,
        kv_bc=kv_bc,
        inputs_window_shape=inputs_window_shape,
        kv_window_shape=kv_window_shape,
        relative_bias_net=relative_bias_net,
        shift_windows=shift_windows,
    )


def spherical_harmonic_lon_lat_encodings(
    ylm_transform: spherical_transforms.SphericalHarmonicsTransform,
    l_max: int,
    l_min: int = 0,
):
  """Spherical harmonic positional encodings for lon-lat grids."""

  def _get_ylm(l, m):
    zeros = np.zeros(ylm_transform.modal_grid.shape)
    # TODO(dkochkov): use sel semantic once it is added to coordax.
    zeros[2 * abs(m) + int(m < 0), l] = 1
    return ylm_transform.to_nodal_array(zeros)

  pe_maps = []
  for l in range(l_min, l_max):
    for m in range(-l, l + 1):
      pe_maps.append(_get_ylm(l, m))

  return np.stack(pe_maps, axis=0)


class PositionalEncoder(Protocol):
  """Protocol for positional encoders."""

  def __call__(
      self,
      inputs: cx.Field,
      dims: tuple[str | cx.Coordinate, ...],
      encoding_dim_tag: str | cx.Coordinate | None = None,
  ) -> cx.Field:
    ...


@nnx_compat.dataclass
class SphericalPositionalEncoder(nnx.Module):
  """Module that generates spherical positional encodings."""

  ylm_transformations: spherical_transforms.SphericalHarmonicsTransformations
  l_max: int

  def __call__(
      self,
      inputs: cx.Field,
      dims: tuple[str | cx.Coordinate, ...],
      encoding_dim_tag: str | cx.Coordinate | None = None,
  ) -> cx.Field:
    """Returns positional encodings for `inputs` over dimensions `dims`."""
    lon_lat_dims = ('longitude', 'latitude')
    grid = cx.compose_coordinates(*[inputs.axes.get(d) for d in lon_lat_dims])
    if grid.dims != dims:
      raise ValueError(
          f'SphericalPositionalEncoder generates encoding for {lon_lat_dims=} '
          f'but got a request for {dims=}'
      )
    ylm_transform = self.ylm_transform_group.ylm_transform(grid)
    pe = spherical_harmonic_lon_lat_encodings(ylm_transform, self.l_max)
    return cx.wrap(pe, encoding_dim_tag, grid)
