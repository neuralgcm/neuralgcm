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
import itertools
import math
from typing import Callable, Protocol, Self, Sequence

import coordax as cx
import einops
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental.core import boundaries
from neuralgcm.experimental.core import coordinates
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
    normalize_v: should V normalization be applied (arxiv.org/abs/2503.04598v3).
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
      attention_fn: Callable[..., typing.Array] = nnx.dot_product_attention,
      normalize_qk: bool = False,
      normalize_v: bool = False,
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
    self.normalize_v = normalize_v

    if self.qkv_features % self.num_heads != 0:
      raise ValueError(
          f'Memory dimension ({self.qkv_features}) must be divisible by '
          f"'num_heads' ({self.num_heads})."
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
    if self.normalize_v:
      self.value_ln = nnx.LayerNorm(
          self.head_dim,
          use_bias=False,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          rngs=rngs,
      )
    else:
      self.value_ln = None

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
      inputs_q: typing.Array,
      inputs_k: typing.Array | None = None,
      inputs_v: typing.Array | None = None,
      attention_bias: typing.Array | None = None,
      mask: typing.Array | None = None,
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

    if self.normalize_v:
      # implements HybridNorm in conjunction with normalize_qk.
      assert self.value_ln is not None
      value = self.value_ln(value)

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


class DyT(nnx.Module):
  """Dynamic Tanh normalization layer.

  DyT(x) = tanh(alpha * x) * gamma + beta

  Attributes:
    alpha: Value of the alpha scaling parameter. Initialized to 0.5.
    beta: Value of the beta shift parameter. Initialized to 0.
    gamma: Value of the gamma scaling parameter. Initialized to 1.
  """

  def __init__(
      self,
      d_in: int,
      *,
      alpha_init: float = 0.5,
      rngs: nnx.Rngs,
  ):
    self.alpha = nnx.Param(
        nnx.initializers.constant(alpha_init)(rngs.params(), (1,), jnp.float32)
    )
    self.beta = nnx.Param(
        nnx.initializers.zeros_init()(rngs.params(), (d_in,), jnp.float32)
    )
    self.gamma = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), (d_in,), jnp.float32)
    )

  def __call__(self, x: typing.Array) -> typing.Array:
    return jnp.tanh(self.alpha.value * x) * self.gamma.value + self.beta.value


def no_gating(skip, x):
  del skip  # unused.
  return x


def residual_gating(skip, x):
  return skip + x


NormalizationFactory = Callable[..., nnx.Module]
# Examples include: nnx.LayerNorm, DyT.


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
    activation: Activation function to be applied to inputs of dense layers.
  """

  input_size: int = dataclasses.field(init=False)
  output_size: int = dataclasses.field(init=False)
  n_layers: int = dataclasses.field(init=False)
  attentions: Sequence[MultiHeadAttention]
  dense_layers: Sequence[standard_layers.UnaryLayer]
  pre_norms: Sequence[nnx.Module] | None
  post_norms: Sequence[nnx.Module] | None
  gating: Sequence[Gating] | Gating = dataclasses.field(
      kw_only=True, default=lambda skip, x: x
  )
  activation: Callable[[typing.Array], typing.Array] = dataclasses.field(
      kw_only=True, default=jax.nn.gelu
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
          f'2 * {n_layers} == {2 * n_layers} gating functions/layers.'
      )

  @abc.abstractmethod
  def rearrange_inputs(self, inputs: typing.Array) -> typing.Array:
    """Converts `inputs` from [C, spatial_axes...] to [B, L, C] format.

    B: batch of sequences, L: sequence length, C: channels.

    Args:
      inputs: array to be rearranged.
    """
    ...

  @abc.abstractmethod
  def rearrange_latents(self, latents: typing.Array) -> typing.Array:
    """Converts `latents` from [C, spatial_axes...] to [B, L, C] format.

    B: batch of sequences, L: sequence length, C: channels.
    Args:
      latents: array to be rearranged.
    """
    ...

  @abc.abstractmethod
  def rearrange_outputs(
      self,
      outputs: typing.Array,
      inputs_shape: tuple[int, ...],
  ) -> typing.Array:
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
  ) -> typing.Array | None:
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
  ) -> typing.Array | None:
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
    """Applies the `attention` to the given query, key, and value.

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
    pre_norms = self.pre_norms or [lambda x: x] * 2 * self.n_layers
    post_norms = self.post_norms or [lambda x: x] * 2 * self.n_layers
    gates = (
        self.gating
        if isinstance(self.gating, Sequence)
        else [self.gating] * (2 * self.n_layers)
    )
    parts = zip(self.attentions, self.dense_layers, strict=True)
    for i, (attention, dense) in enumerate(parts):
      x_norm = pre_norms[2 * i](x)
      keys = self.attention_key_source(x_norm, z, x_pos_enc, z_pos_enc)
      values = self.attention_value_source(x_norm, z, x_pos_enc, z_pos_enc)
      x_attn = self.apply_attention(
          attention=attention,
          query=x_norm,
          key=keys,
          value=values,
          query_pos_encoding=x_pos_enc,
          kv_pos_encoding=z_pos_enc,
          mask=mask,
          layer_idx=i,
      )
      x = gates[2 * i](x, self.activation(x_attn))
      x = post_norms[2 * i](x)
      x_dense = dense(pre_norms[2 * i + 1](x))
      x = gates[2 * i + 1](x, x_dense)
      x = post_norms[2 * i + 1](x)

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
      pre_normalization_factory: NormalizationFactory | None = None,
      post_normalization_factory: NormalizationFactory | None = None,
      gating: Sequence[Gating] | Gating | None = lambda skip, x: x,
      dense_factory: standard_layers.UnaryLayerFactory,
      normalize_qk: bool = False,
      normalize_v: bool = False,
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
      qkv_features: Channels for query/key/value.
      use_bias_in_attention: If True, `MultiHeadAttention` includes bias.
      pre_normalization_factory: factory for pre-normalization layers.
      post_normalization_factory: factory for post-normalization layers.
      gating: Gating function(s). Defaults to no gating. If None, a "where
        possible" gating strategy is used.
      dense_factory: Factory for dense layers post-attention.
      normalize_qk: whether to add layer norm prior to computing query and key.
      normalize_v: whether to add layer norm prior to computing value.
      w_init: Kernel initializer for `MultiHeadAttention`.
      b_init: Bias initializer for `MultiHeadAttention`.
      rngs: JAX PRNG keys.

    Returns:
      A tuple with (attentions, dense_layers, pre_norms, post_norms, gating)
      suitable for initializing a `TransformerBase` instance.
    """
    if output_size is None:
      output_size = input_size
    input_sizes = (input_size,) + tuple(intermediate_sizes)
    output_sizes = tuple(intermediate_sizes) + (output_size,)
    attentions = []
    dense_layers = []
    pre_norms = []
    post_norms = []
    possible_gating = []
    for i, (d_in, d_out) in enumerate(zip(input_sizes, output_sizes)):
      if qkv_features is None:
        if d_in % num_heads != 0:
          raise ValueError(
              f'{d_in=} at layer={i} is not divisible by {num_heads=}'
              ' which is required if qkv_features is not specified.'
          )
        qkv_features = d_in
      attentions.append(
          MultiHeadAttention(
              num_heads=num_heads,
              in_features=d_in,
              qkv_features=qkv_features,
              out_features=d_out,
              use_bias=use_bias_in_attention,
              kernel_init=w_init,
              normalize_qk=normalize_qk,
              normalize_v=normalize_v,
              bias_init=b_init,
              rngs=rngs,
          )
      )
      if pre_normalization_factory is not None:
        pre_norms.append(pre_normalization_factory(d_in, rngs=rngs))
        pre_norms.append(pre_normalization_factory(d_out, rngs=rngs))
      if post_normalization_factory is not None:
        post_norms.append(post_normalization_factory(d_out, rngs=rngs))
        post_norms.append(post_normalization_factory(d_out, rngs=rngs))
      possible_gating.append(no_gating if d_in != d_out else residual_gating)
      dense_layers.append(dense_factory(d_out, d_out, rngs=rngs))
      possible_gating.append(residual_gating)  # (attention -> dense) residual.
    pre_norms = pre_norms if pre_normalization_factory is not None else None
    post_norms = post_norms if post_normalization_factory is not None else None
    if gating is None:
      gating = possible_gating
    return (attentions, dense_layers, pre_norms, post_norms, gating)

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
      pre_normalization_factory: NormalizationFactory | None = None,
      post_normalization_factory: NormalizationFactory | None = None,
      activation: Callable[[typing.Array], typing.Array] = jax.nn.gelu,
      gating: Sequence[Gating] | Gating | None = lambda skip, x: x,
      dense_factory: standard_layers.UnaryLayerFactory,
      normalize_qk: bool = False,
      normalize_v: bool = False,
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
      pre_normalization_factory: factory for pre-normalization layers.
      post_normalization_factory: factory for post-normalization layers.
      activation: activation function to be applied to the output of attention.
      gating: sequence of 2n+1 or single gating function. Default is no gating.
      dense_factory: factory for generating dense layers that follow attention.
      normalize_qk: whether to add layer norm prior to computing query and key.
      normalize_v: whether to add layer norm prior to computing value.
      w_init: kernel initializer for the MultiHeadAttention modules.
      b_init: bias initializer for the MultiHeadAttention modules.
      rngs: random number generator for parameter initialization.
    """
    attentions, dense_layers, pre_norms, post_norms, gating = (
        cls.build_args_using_factories(
            input_size=input_size,
            output_size=output_size,
            intermediate_sizes=intermediate_sizes,
            num_heads=num_heads,
            qkv_features=qkv_features,
            use_bias_in_attention=use_bias_in_attention,
            pre_normalization_factory=pre_normalization_factory,
            post_normalization_factory=post_normalization_factory,
            gating=gating,
            dense_factory=dense_factory,
            normalize_qk=normalize_qk,
            normalize_v=normalize_v,
            w_init=w_init,
            b_init=b_init,
            rngs=rngs,
        )
    )
    return cls(
        attentions=attentions,
        dense_layers=dense_layers,
        pre_norms=pre_norms,
        post_norms=post_norms,
        gating=gating,
        activation=activation,
    )


class TransformerBlocks(TransformerBase):
  """A standard sequence of Transformer blocks."""

  def rearrange_inputs(self, inputs: typing.Array) -> typing.Array:
    """Transposes inputs for attention: [C, L] -> [L, C]."""
    return jnp.transpose(inputs)

  def rearrange_latents(self, latents: typing.Array) -> typing.Array:
    """Transposes latents for attention: [C, L] -> [L, C]."""
    return jnp.transpose(latents)

  def rearrange_outputs(
      self, outputs: typing.Array, inputs_shape
  ) -> typing.Array:
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
      raise ValueError(
          f'{spatial_shape=} is incompatible with {window_shape=} as it does '
          'not result in integer number of windows.'
      )
    n_windows = [x[0] for x in windows_divmod]
    n_names = [f'n{i}' for i in range(len(n_windows))]
    w_names = [f'w{i}' for i in range(len(n_windows))]
    n_w_zip = zip(n_names, w_names)
    # spatial_pattern represent spatial_shape: 'c (n0 w0) (n1 w1)'.
    spatial_pattern = 'c ' + ' '.join([f'({n} {w})' for n, w in n_w_zip])
    # window_pattern represent grouped shape: '(n0 n1 ...) (w0 w1 ...) c'.
    window_pattern = f"({' '.join(n_names)}) ({' '.join(w_names)}) c"
    shape_kwargs = {name: size for name, size in zip(n_names, n_windows)} | {
        name: size for name, size in zip(w_names, window_shape)
    }
    return spatial_pattern, window_pattern, shape_kwargs

  def _pad_sizes(
      self,
      spatial_shape: tuple[int, ...],
      window_shape: tuple[int, ...],
      shifts: tuple[int, ...] | None = None,
  ) -> tuple[tuple[int, ...], ...]:
    """Calculates padding sizes for windowing with optional shifts."""
    if shifts is None:
      shifts = tuple(0 for _ in window_shape)
    return tuple(
        (s % w, (w * math.ceil((x + s) / w) - (x + s)) % w)
        for x, w, s in zip(spatial_shape, window_shape, shifts)
    )

  def _to_windows(
      self,
      array: typing.Array,
      window_shape: tuple[int, ...],
      bc: boundaries.BoundaryCondition,
      shifts: tuple[int, ...] | None = None,
  ) -> typing.Array:
    """Pads and rearranges an array into non-overlapping windows.

    If `array` does not correspond to an integer number of windows, then
    additional ghost cells are padded to the array using boundary conditions
    `bc`. If `shifts` are provided, then the origins of windows are altered,
    resulting in different ghost cell padding.

    Args:
      array: Input array, [C, *spatial_dims].
      window_shape: Shape of spatial windows (e.g. [h_win, w_win]).
      bc: Boundary condition for padding.
      shifts: Optional shifts to apply before windowing (e.g., for SWIN).

    Returns:
      Windowed array [num_windows, window_volume, C]
    """
    pad_sizes = self._pad_sizes(array.shape[1:], window_shape, shifts)

    @nnx.vmap(in_axes=(None, 0))  # vmap over channel dimension.
    def pad(bc, x):
      return bc.pad_array(x, pad_sizes)

    padded_array = pad(bc, array)
    space_pattern, window_pattern, shape_kwargs = self._window_rearrange_args(
        padded_array.shape, window_shape
    )
    result = einops.rearrange(
        padded_array, f'{space_pattern} -> {window_pattern}', **shape_kwargs
    )
    return result

  def _from_windows(
      self,
      array: typing.Array,
      window_shape: tuple[int, ...],
      bc: boundaries.BoundaryCondition,
      spatial_shape: tuple[int, ...],
      shifts: tuple[int, ...] | None = None,
  ) -> typing.Array:
    """Rearranges a windowed array back to an original spatial shape.

    If original `spatial_shape`, `window_shape` and `shifts` indicate that
    array required padding to be windowed, then this reconstruction performs
    an additional trim operation to remove the padded ghost cells. This ensures
    the invariant that `_from_windows` returns arrays without additional ghost
    cells and `_to_windows` introduces the minimal number required to perform
    the rearrangement.

    Args:
      array: Windowed array to reshape and maybe trim to `[C, *spatial_shape]`.
      window_shape: Shape of the windows (e.g., [h_win, w_win]).
      bc: Boundary condition describing how to pad/trim ghost cells..
      spatial_shape: Spatial shape of the array without padding (e.g. [H, W]).
      shifts: Optional shifts that were applied to the windowing procedure.

    Returns:
      Array with shape `[C, *spatial_shape]`.
    """
    pad_sizes = self._pad_sizes(spatial_shape, window_shape, shifts)
    spatial_shape = tuple(x + sum(s) for x, s in zip(spatial_shape, pad_sizes))
    space_pattern, window_pattern, shape_kwargs = self._window_rearrange_args(
        (0,) + spatial_shape, window_shape
    )
    spatial = einops.rearrange(
        array, f'{window_pattern} -> {space_pattern}', **shape_kwargs
    )

    @nnx.vmap(in_axes=(None, 0))  # vmap over channel dimension.
    def trim(bc, x):
      return bc.trim_array(x, pad_sizes)

    return trim(bc, spatial)

  def _attention_bias(
      self,
      q_pe_windows: typing.Array,
      k_pe_windows: typing.Array,
  ) -> typing.Array | None:
    """Generates the relative position bias values."""
    if self.relative_bias_net is None:
      return None
    # windows have shapes: [B, WINDOW_SIZE, D], D == positional encoding size.
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

  def rearrange_inputs(self, inputs: typing.Array) -> typing.Array:
    return self._to_windows(inputs, self.inputs_window_shape, self.inputs_bc)

  def rearrange_latents(self, latents: typing.Array) -> typing.Array:
    return self._to_windows(latents, self.kv_window_shape, self.kv_bc)

  def rearrange_outputs(
      self, outputs: typing.Array, inputs_shape: tuple[int, ...]
  ) -> typing.Array:
    """Rearranges outputs back to the original spatial shape."""
    return self._from_windows(
        outputs,
        self.inputs_window_shape,
        self.inputs_bc,
        spatial_shape=inputs_shape[1:],
    )

  def attention_key_source(self, x, z, x_pos_encoding, z_pos_encoding):
    # TODO(dkochkov): Consider adding class attributes that choose alternatives.
    return z

  def attention_value_source(self, x, z, x_pos_encoding, z_pos_encoding):
    # TODO(dkochkov): Consider adding class attributes that choose alternatives.
    return z  # Parameterize by class attributes?

  def apply_attention(
      self,
      attention: MultiHeadAttention,
      query: typing.Array,
      key: typing.Array | None,
      value: typing.Array | None,
      mask: typing.Array | None,
      query_pos_encoding: typing.Array,
      kv_pos_encoding: typing.Array | None,
      layer_idx: int,
  ):
    """Applies `attention` module to input q,k,v,mask over windows.

    This method manages the application of the attention mechanism, including
    handling optional window shifting (e.g., for SWIN transformers) and
    computing relative positional biases based on windowed positional
    encodings.

    Throughout this method, an invariant is maintained: `query`, `key`,
    `value`, and `mask` (if not None) are expected to be in and are returned
    in their windowed format, i.e., [num_windows, window_volume, C].
    The `_to_windows` and `_from_windows` methods handle necessary padding and
    un-padding when data is temporarily brought out of windowed format,
    for example, during window shifting operations. Positional encodings
    (`query_pos_encoding`, `kv_pos_encoding`) are provided in their original
    spatial format and are converted to windowed format internally as needed
    for calculating attention biases.

    Args:
      attention: The MultiHeadAttention module to apply.
      query: Query tensor in windowed format [B, Lq_win, C].
      key: Key tensor in windowed format [B, Lkv_win, C], or None.
      value: Value tensor in windowed format [B, Lkv_win, C], or None.
      mask: Attention mask in windowed format, or None.
      query_pos_encoding: Positional encodings for `query` in original spatial
        format [C, spatial_dims...].
      kv_pos_encoding: Positional encodings for `key`/`value` in original
        spatial format [C, spatial_dims...], or None (uses query_pos_encoding).
      layer_idx: Index of the current attention layer, used for window shifting.

    Returns:
      The output of the attention module, in windowed format.
    """
    if kv_pos_encoding is None:
      kv_pos_encoding = query_pos_encoding

    q_shape, kv_shape = query_pos_encoding.shape, kv_pos_encoding.shape
    # Transform positional encodings to window format.
    if layer_idx % 2 == 0 or not self.shift_windows:
      q_shifts = tuple(0 for _ in self.inputs_window_shape)  # no window shifts.
      q_pe_windows = self._to_windows(
          query_pos_encoding,
          self.inputs_window_shape,
          self.inputs_bc,
      )
      kv_pe_windows = self._to_windows(
          kv_pos_encoding,
          self.kv_window_shape,
          self.kv_bc,
      )
      attention_bias = self._attention_bias(q_pe_windows, kv_pe_windows)
      result = attention(
          query, key, value, mask=mask, attention_bias=attention_bias
      )
    else:  # if shift_windows is True and odd layer - apply shift to windows.
      q_shifts = tuple(w // 2 for w in self.inputs_window_shape)
      q_pe_windows = self._to_windows(
          query_pos_encoding,
          self.inputs_window_shape,
          self.inputs_bc,
          shifts=q_shifts,
      )
      kv_shifts = tuple(w // 2 for w in self.kv_window_shape)
      kv_pe_windows = self._to_windows(
          kv_pos_encoding,
          self.kv_window_shape,
          self.kv_bc,
          shifts=kv_shifts,
      )
      attention_bias = self._attention_bias(q_pe_windows, kv_pe_windows)
      # To obtain proper shifted windows we move to spatial and back to windows.
      # _to_windows and _from_windows manage padding and trimming arising from
      # the addition of the shift argument. The same process is applied to other
      # tensors that are subject to windowing.
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
          shifts=q_shifts,
      )  # move back to windows, but with a shift.
      if mask is not None:
        # adding shifting, see comment above similar transform on query above.
        mask = self._from_windows(
            mask,
            self.kv_window_shape,
            self.kv_bc,
            spatial_shape=kv_shape[1:],
        )
        mask = self._to_windows(
            mask,
            self.kv_window_shape,
            self.kv_bc,
            shifts=kv_shifts,
        )
      if key is not None:
        # adding shifting, see comment above similar transform on query above.
        key = self._from_windows(
            key, self.kv_window_shape, self.kv_bc, spatial_shape=kv_shape[1:]
        )
        key = self._to_windows(
            key,
            self.kv_window_shape,
            self.kv_bc,
            shifts=kv_shifts,
        )
      if value is not None:
        # adding shifting, see comment above similar transform on query above.
        value = self._from_windows(
            value, self.kv_window_shape, self.kv_bc, spatial_shape=kv_shape[1:]
        )
        value = self._to_windows(
            value,
            self.kv_window_shape,
            self.kv_bc,
            shifts=kv_shifts,
        )
      result = attention(
          query, key, value, mask=mask, attention_bias=attention_bias
      )
    # Result should be in a windowed format without shifts with valid unused
    # ghost cells. This is achieved by moving to spatial and back to windowed
    # format. If windowing is not used, this either refreshes the ghost cells
    # or becomes a no-op if no cells were needed.
    result = self._from_windows(
        result,
        self.inputs_window_shape,
        self.inputs_bc,
        spatial_shape=q_shape[1:],
        shifts=q_shifts,
    )
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
      pre_normalization_factory: NormalizationFactory | None = None,
      post_normalization_factory: NormalizationFactory | None = None,
      activation: Callable[[typing.Array], typing.Array] = jax.nn.gelu,
      gating: Sequence[Gating] | Gating | None = lambda skip, x: x,
      dense_factory: standard_layers.UnaryLayerFactory,
      normalize_qk: bool = False,
      normalize_v: bool = False,
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
      pre_normalization_factory: factory for pre-normalization layers.
      post_normalization_factory: factory for post-normalization layers.
      activation: activation function to use in the feed-forward layers.
      gating: sequence of 2n or single gating function. Default is no gating.
      dense_factory: factory for generating dense layers that follow attention.
      normalize_qk: whether to add layer norm prior to computing query and key.
      normalize_v: whether to add layer norm prior to computing value.
      w_init: kernel initializer for the MultiHeadAttention modules.
      b_init: bias initializer for the MultiHeadAttention modules.
      rngs: random number generator for parameter initialization.
    """
    attentions, dense_layers, pre_norms, post_norms, gating = (
        TransformerBase.build_args_using_factories(
            input_size=input_size,
            output_size=output_size,
            intermediate_sizes=intermediate_sizes,
            num_heads=num_heads,
            qkv_features=qkv_features,
            use_bias_in_attention=use_bias_in_attention,
            pre_normalization_factory=pre_normalization_factory,
            post_normalization_factory=post_normalization_factory,
            gating=gating,
            dense_factory=dense_factory,
            normalize_qk=normalize_qk,
            normalize_v=normalize_v,
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
        pre_norms=pre_norms,
        post_norms=post_norms,
        gating=gating,
        activation=activation,
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

  return jnp.stack(pe_maps, axis=0)


class PositionalEncoder(Protocol):
  """Protocol for positional encoders."""

  def __call__(
      self,
      inputs: cx.Field,
      dims: tuple[str | cx.Coordinate, ...],
      encoding_dim_tag: str | cx.Coordinate | None = None,
  ) -> cx.Field:
    ...


def _dim_names(*dims: str | cx.Coordinate) -> tuple[str, ...]:
  """Returns dimension names for a sequence of names and coordinates."""
  dim_tuples = [(c,) if isinstance(c, str) else c.dims for c in dims]
  return tuple(itertools.chain(*dim_tuples))


@nnx_compat.dataclass
class SphericalPositionalEncoder(nnx.Module):
  """Module that generates spherical positional encodings."""

  ylm_mapper: spherical_transforms.YlmMapper
  l_max: int
  l_min: int = 0

  def __call__(
      self,
      inputs: cx.Field,
      dims: tuple[str | cx.Coordinate, ...],
      encoding_dim_tag: str | cx.Coordinate | None = None,
  ) -> cx.Field:
    """Returns positional encodings for `inputs` over dimensions `dims`."""
    lon_lat_dims = ('longitude', 'latitude')
    grid = cx.compose_coordinates(*[inputs.axes.get(d) for d in lon_lat_dims])
    if not isinstance(grid, coordinates.LonLatGrid):
      raise ValueError(
          'SphericalPositionalEncoder generates encoding for LonLatGrid data '
          f'but inputs has coordinates {grid=}'
      )
    l_max, l_min = self.l_max, self.l_min
    ylm_transform = self.ylm_mapper.ylm_transform(grid)
    pe = spherical_harmonic_lon_lat_encodings(ylm_transform, l_max, l_min)
    return cx.wrap(pe, encoding_dim_tag, grid)
